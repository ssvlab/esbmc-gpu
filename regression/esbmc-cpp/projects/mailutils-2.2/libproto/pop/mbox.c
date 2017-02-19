/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#ifdef ENABLE_POP

#include <termios.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdarg.h>

#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/attribute.h>
#include <mailutils/auth.h>
#include <mailutils/body.h>
#include <mailutils/debug.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/header.h>
#include <mailutils/message.h>
#include <mailutils/observer.h>
#include <mailutils/property.h>
#include <mailutils/stream.h>
#include <mailutils/url.h>
#include <mailutils/secret.h>
#include <mailutils/tls.h>
#include <mailutils/md5.h>
#include <mailutils/io.h>
#include <mailutils/mutil.h>
#include <mailutils/cstr.h>
#include <mailutils/cctype.h>

#include <folder0.h>
#include <mailbox0.h>
#include <registrar0.h>
#include <url0.h>

#define PROP_RFC822 1

/* Advance declarations.  */
struct _pop_data;
struct _pop_message;

typedef struct _pop_data * pop_data_t;
typedef struct _pop_message * pop_message_t;

/* The different possible states of a Pop client, Note that POP3 is not
   reentrant i.e. it is only one channel, so it is not possible to start
   Another operation while one is running.  The only resort is to close the
   connection and reopen it again.  This is what we do, the downside is that
   the client as to get the authentication again user/pass.  */
enum pop_state
{
  POP_NO_STATE, POP_STATE_DONE,
  POP_OPEN_CONNECTION,
  POP_GREETINGS,
  POP_CAPA, POP_CAPA_ACK,
  POP_APOP, POP_APOP_ACK,
  POP_DELE, POP_DELE_ACK,
  POP_LIST, POP_LIST_ACK, POP_LIST_RX,
  POP_QUIT, POP_QUIT_ACK,
  POP_NOOP, POP_NOOP_ACK,
  POP_RETR, POP_RETR_ACK, POP_RETR_RX_HDR, POP_RETR_RX_BODY,
  POP_RSET, POP_RSET_ACK,
  POP_STAT, POP_STAT_ACK,
  POP_STLS, POP_STLS_ACK,
  POP_TOP,  POP_TOP_ACK,  POP_TOP_RX,
  POP_UIDL, POP_UIDL_ACK,
  POP_AUTH, POP_AUTH_DONE,
  POP_AUTH_USER, POP_AUTH_USER_ACK,
  POP_AUTH_PASS, POP_AUTH_PASS_ACK
};

/*  POP3 capabilities  */
#define CAPA_TOP             0x00000001
#define CAPA_USER            0x00000002
#define CAPA_UIDL            0x00000004
#define CAPA_RESP_CODES      0x00000008
#define CAPA_LOGIN_DELAY     0x00000010
#define CAPA_PIPELINING      0x00000020
#define CAPA_EXPIRE          0x00000040
#define CAPA_SASL            0x00000080
#define CAPA_STLS            0x00000100
#define CAPA_IMPLEMENTATION  0x00000200

static void pop_destroy        (mu_mailbox_t);
static int pop_capa            (mu_mailbox_t);
static int pop_stls            (mu_mailbox_t);

/*  Functions/Methods that implements the mu_mailbox_t API.  */
static int pop_open            (mu_mailbox_t, int);
static int pop_close           (mu_mailbox_t);
static int pop_get_message     (mu_mailbox_t, size_t, mu_message_t *);
static int pop_messages_count  (mu_mailbox_t, size_t *);
static int pop_messages_recent (mu_mailbox_t, size_t *);
static int pop_message_unseen  (mu_mailbox_t, size_t *);
static int pop_expunge         (mu_mailbox_t);
static int pop_scan            (mu_mailbox_t, size_t, size_t *);
static int pop_is_updated      (mu_mailbox_t);

/* The implementation of mu_message_t */
int _pop_user            (mu_authority_t);
int _pop_apop            (mu_authority_t);
static int pop_get_size        (mu_mailbox_t, mu_off_t *);
/* We use pop_top for retreiving headers.  */
/* static int pop_header_read (mu_header_t, char *, size_t, mu_off_t, size_t *); */
static int pop_body_transport  (mu_stream_t, mu_transport_t *, mu_transport_t *);
static int pop_body_size       (mu_body_t, size_t *);
static int pop_body_lines      (mu_body_t, size_t *);
static int pop_body_read       (mu_stream_t, char *, size_t, mu_off_t, size_t *);
static int pop_message_read    (mu_stream_t, char *, size_t, mu_off_t, size_t *);
static int pop_message_size    (mu_message_t, size_t *);
static int pop_message_transport (mu_stream_t, mu_transport_t *, mu_transport_t *);
static int pop_top             (mu_header_t, char *, size_t, mu_off_t, size_t *);
static int pop_retr            (pop_message_t, char *, size_t, mu_off_t, size_t *);
static int pop_get_transport2   (pop_message_t, mu_transport_t *, mu_transport_t *);
static int pop_get_attribute   (mu_attribute_t, int *);
static int pop_set_attribute   (mu_attribute_t, int);
static int pop_unset_attribute (mu_attribute_t, int);
static int pop_uidl            (mu_message_t, char *, size_t, size_t *);
static int pop_uid             (mu_message_t, size_t *);
static int fill_buffer         (pop_data_t, char *, size_t);
static int pop_sleep           (int);
static int pop_readline        (pop_data_t);
static int pop_read_ack        (pop_data_t);
static int pop_writeline       (pop_data_t, const char *, ...)
                                 MU_PRINTFLIKE(2,3);
static int pop_write           (pop_data_t);
static int pop_get_user        (mu_authority_t);
static int pop_get_passwd      (mu_authority_t);
static char *pop_get_timestamp (pop_data_t);
static int pop_get_md5         (pop_data_t);

/* This structure holds the info for a message. The pop_message_t
   type, will serve as the owner of the mu_message_t and contains the command to
   send to "RETR"eive the specify message.  The problem comes from the header.
   If the  POP server supports TOP, we can cleanly fetch the header.
   But otherwise we use the clumsy approach. .i.e for the header we read 'til
   ^\n then discard the rest, for the body we read after ^\n and discard the
   beginning.  This is a waste, Pop was not conceive for this obviously.  */
struct _pop_message
{
  int inbody;
  int skip_header;
  int skip_body;
  size_t body_size;
  size_t header_size;
  size_t body_lines;
  size_t header_lines;
  size_t mu_message_size;
  size_t num;
  char *uidl; /* Cache the uidl string.  */
  int attr_flags;
  mu_message_t message;
  pop_data_t mpd; /* Back pointer.  */
};

/* Structure to hold things general to the POP mailbox, like its state, how
   many messages we have so far etc ...  */
struct _pop_data
{
  void *func;  /*  Indicate a command is in operation, busy.  */
  size_t id;   /* A second level of distincion, we maybe in the same function
		  but working on a different message.  */
  int pops; /* POPS or POP? */
  char *greeting_banner; /* A greeting banner */
  unsigned long capa; /* Server capabilities */
  enum pop_state state;
  pop_message_t *pmessages;
  size_t pmessages_count;
  size_t messages_count;
  size_t size;

  /* Working I/O buffers.  */
  char *buffer;
  size_t buflen; /* Len of buffer.  */
  char *ptr; /* Points to the end of the buffer i.e the non consume chars.  */
  char *nl;  /* Points to the '\n' char in te string.  */
  mu_off_t offset; /* Dummy, this is use because of the stream buffering.
		   The mu_stream_t maintains and offset and the offset we use must
		   be in sync.  */

  int is_updated;
  char *user;     /* Temporary holders for user and passwd.  */
  mu_secret_t secret;
  char *digest;
  mu_mailbox_t mbox; /* Back pointer.  */
} ;

/* Usefull little Macros, since these are very repetitive.  */

/* Check if we're busy ?  */
/* POP is a one channel download protocol, so if someone
   is trying to execute a command while another is running
   something is seriously incorrect,  So the best course
   of action is to close down the connection and start a new one.
   For example mu_mime_t only reads part of the message.  If a client
   wants to read different part of the message via mime it should
   download it first.  POP does not have the features of IMAP for
   multipart messages.
   Let see a concrete example:
   {
     mu_mailbox_t mbox; mu_message_t msg; mu_stream_t stream; char buffer[105];
     mu_mailbox_create (&mbox, "pop://qnx.com");
     mu_mailbox_get_message (mbox, 1, &msg);
     mu_message_get_stream (msg, &stream);
     while (mu_stream_readline (stream, buffer, sizeof(buffer), NULL) != 0) { ..}
   }
   if in the while of the readline, one try to get another email.  The pop
   server will get seriously confused, and the second message will still
   be the first one,  There is no way to tell POP servers yo! stop/abort.
   The approach is to close the stream and reopen again. So  every time
   we go in to a function our state is preserve by the triplets
   mpd->{func,state,id}.  The macro CHECK_BUSY checks if we are not
   in another operation if not you get access if yes the stream is close
   and pop_open() is recall again for a new connection.
 */
#define CHECK_BUSY(mbox, mpd, function, identity) \
do \
  { \
    int err = mu_monitor_wrlock (mbox->monitor); \
    if (err != 0) \
      return err; \
    if ((mpd->func && mpd->func != function) \
        || (mpd->id && mpd->id != (size_t)identity)) \
      { \
        mpd->id = 0; \
        mpd->func = (void *)pop_open; \
        mpd->state = POP_NO_STATE; \
        mu_monitor_unlock (mbox->monitor); \
        err = pop_open (mbox, mbox->flags); \
        if (err != 0) \
          { \
            return err; \
          } \
      } \
    else \
      { \
        mpd->id = (size_t)identity; \
        mpd->func = func; \
        mu_monitor_unlock (mbox->monitor); \
      } \
  } \
while (0)

/* Clear the state.  */
#define CLEAR_STATE(mpd) \
 mpd->id = 0, mpd->func = NULL, mpd->state = POP_NO_STATE

/* Clear the state and close the stream.  */
#define CHECK_ERROR_CLOSE(mbox, mpd, status) \
do \
  { \
     if (status != 0) \
       { \
          mu_stream_close (mbox->stream); \
          CLEAR_STATE (mpd); \
          mpd->func = (void *)-1; \
          MU_DEBUG1 (mbox->debug, MU_DEBUG_PROT, \
                     "CHECK_ERROR_CLOSE: %s\n", mu_strerror (status));\
          return status; \
       } \
  } \
while (0)

/* If error, clear the state and return.  */
#define CHECK_ERROR(mpd, status) \
do \
  { \
     if (status != 0) \
       { \
          CLEAR_STATE (mpd); \
          mpd->func = (void*)-1; \
          MU_DEBUG1(mpd->mbox->debug, MU_DEBUG_PROT, \
                    "CHECK_ERROR: %s\n", mu_strerror (status));\
          return status; \
       } \
  } \
while (0)

/* Clear the state for non recoverable error.  */
#define CHECK_EAGAIN(mpd, status) \
do \
  { \
    if (status != 0) \
      { \
         if (status != EAGAIN && status != EINPROGRESS && status != EINTR) \
           { \
             CLEAR_STATE (mpd); \
             mpd->func = (void *)-1; \
             MU_DEBUG1(mpd->mbox->debug, MU_DEBUG_PROT, \
                       "CHECK_EAGAIN: %s\n", mu_strerror (status));\
           } \
         return status; \
      } \
   }  \
while (0)


/* Allocate mu_mailbox_t, allocate pop internal structures.  */
static int
_mailbox_pop_and_pops_init (mu_mailbox_t mbox, int pops)
{
  pop_data_t mpd;
  int status = 0;

  /* Allocate specifics for pop data.  */
  mpd = mbox->data = calloc (1, sizeof (*mpd));
  if (mbox->data == NULL)
    return ENOMEM;

  mpd->mbox = mbox;		/* Back pointer.  */
  mpd->state = POP_NO_STATE;	/* Init with no state.  */
  mpd->pops = pops;

  /* Initialize the structure.  */
  mbox->_destroy = pop_destroy;

  mbox->_open = pop_open;
  mbox->_close = pop_close;

  /* Messages.  */
  mbox->_get_message = pop_get_message;
  mbox->_messages_count = pop_messages_count;
  mbox->_messages_recent = pop_messages_recent;
  mbox->_message_unseen = pop_message_unseen;
  mbox->_expunge = pop_expunge;

  mbox->_scan = pop_scan;
  mbox->_is_updated = pop_is_updated;

  mbox->_get_size = pop_get_size;

  /* Set our properties.  */
  {
    mu_property_t property = NULL;
    mu_mailbox_get_property (mbox, &property);
    mu_property_set_value (property, "TYPE", "POP3", 1);
  }

  /* Hack! POP does not really have a folder.  */
  mbox->folder->data = mbox;

  return status;
}

int
_mailbox_pop_init (mu_mailbox_t mbox)
{
  return _mailbox_pop_and_pops_init (mbox, 0);
}

int
_mailbox_pops_init (mu_mailbox_t mbox)
{
  return _mailbox_pop_and_pops_init (mbox, 1);
}

/*  Cleaning up all the ressources associate with a pop mailbox.  */
static void
pop_destroy (mu_mailbox_t mbox)
{
  if (mbox->data)
    {
      pop_data_t mpd = mbox->data;
      size_t i;
      mu_monitor_wrlock (mbox->monitor);
      /* Destroy the pop messages and ressources associated to them.  */
      for (i = 0; i < mpd->pmessages_count; i++)
	{
	  if (mpd->pmessages[i])
	    {
	      mu_message_destroy (&(mpd->pmessages[i]->message),
			       mpd->pmessages[i]);
	      if (mpd->pmessages[i]->uidl)
		free (mpd->pmessages[i]->uidl);
	      free (mpd->pmessages[i]);
	      mpd->pmessages[i] = NULL;
	    }
	}
      if (mpd->greeting_banner)
	free (mpd->greeting_banner);
      if (mpd->buffer)
	free (mpd->buffer);
      if (mpd->pmessages)
	free (mpd->pmessages);
      free (mpd);
      mbox->data = NULL;
      mu_monitor_unlock (mbox->monitor);
    }
}

static int
pop_mbox_uidls (mu_mailbox_t mbox, mu_list_t list)
{
  pop_data_t mpd = mbox->data;
  int status;

  status = pop_writeline (mpd, "UIDL\r\n");
  CHECK_ERROR (mpd, status);
  MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);

  status = pop_write (mpd);
  CHECK_EAGAIN (mpd, status);

  status = pop_read_ack (mpd);
  CHECK_EAGAIN (mpd, status);
  MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);

  if (!mu_c_strncasecmp (mpd->buffer, "+OK", 3))
    {
      do
	{
	  char *p;
	  size_t num;
	  struct mu_uidl *uidl;
	  
	  status = pop_read_ack (mpd);
	  MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);

	  num = strtoul (mpd->buffer, &p, 10);
	  if (*p == 0 || !mu_isblank (*p))
	    continue; /* FIXME: or error? */
	  p = mu_str_skip_class (p, MU_CTYPE_SPACE);
	  mu_rtrim_cset (p, "\r\n");

	  uidl = malloc (sizeof (uidl[0]));
	  if (!uidl)
	    {
	      status = ENOMEM;
	      break;
	    }
	  uidl->msgno = num;
	  strncpy (uidl->uidl, p, MU_UIDL_BUFFER_SIZE);
	  status = mu_list_append (list, uidl);
	}
      while (mpd->nl);
    }
  else
    status = ENOSYS;
  return status;
}

/*
  POP3 CAPA support.
 */

static int
pop_parse_capa (pop_data_t mpd)
{
  int status;
  if (!mu_c_strncasecmp (mpd->buffer, "+OK", 3))
    {
      mpd->capa = 0;
      do
	{
	  status = pop_read_ack (mpd);
	  MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);

	  /* Here we check some common capabilities like TOP, USER, UIDL,
	     and STLS. The rest are ignored. Please note that some
	     capabilities might have an extra arguments. For instance,
	     SASL can have CRAM-MD5 and/or KERBEROS_V4, and etc.
	     This is why I suggest adding (in a future) an extra variable,
	     for example `capa_sasl'. It would hold the following flags:
	     SASL_CRAM_MD5, SASL_KERBEROS_V4, and so on. Also the EXPIRE
	     and LOGIN-DELAY capabilities have an extra arguments!
	     Note that there is no APOP capability, even though APOP
	     is an optional command in POP3. -- W.P. */

	  if (!mu_c_strncasecmp (mpd->buffer, "TOP", 3))
	    mpd->capa |= CAPA_TOP;
	  else if (!mu_c_strncasecmp (mpd->buffer, "USER", 4))
	    mpd->capa |= CAPA_USER;
	  else if (!mu_c_strncasecmp (mpd->buffer, "UIDL", 4))
	    mpd->capa |= CAPA_UIDL;
	  else if (!mu_c_strncasecmp (mpd->buffer, "STLS", 4))
	    mpd->capa |= CAPA_STLS;
	}
      while (mpd->nl);

      if (mpd->capa & CAPA_UIDL)
	mpd->mbox->_get_uidls = pop_mbox_uidls;
  
      return status;
    }
  else
    {
      /* mu_error ("CAPA not implemented"); */ /* FIXME */
      return ENOSYS;
    }
}

static int
pop_capa (mu_mailbox_t mbox)
{
  pop_data_t mpd = mbox->data;
  int status;

  status = pop_writeline (mpd, "CAPA\r\n");
  CHECK_ERROR (mpd, status);
  MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);

  status = pop_write (mpd);
  CHECK_EAGAIN (mpd, status);
  mpd->state = POP_CAPA_ACK;

  /* POP_CAPA_ACK */
  status = pop_read_ack (mpd);
  CHECK_EAGAIN (mpd, status);
  MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);

  return pop_parse_capa (mpd);
}

/* Simple User/pass authentication for pop. We ask for the info
   from the standard input.  */
int
_pop_user (mu_authority_t auth)
{
  mu_folder_t folder = mu_authority_get_owner (auth);
  mu_mailbox_t mbox = folder->data;
  pop_data_t mpd = mbox->data;
  int status;

  switch (mpd->state)
    {
    case POP_AUTH:
      /*  Fetch the user from them.  */
      status = pop_get_user (auth);
      if (status != 0 || mpd->user == NULL || mpd->user[0] == '\0')
	{
	  pop_writeline (mpd, "QUIT\r\n");
	  MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
	  pop_write (mpd);
	  CHECK_ERROR_CLOSE (mbox, mpd, MU_ERR_NOUSERNAME);
	}
      status = pop_writeline (mpd, "USER %s\r\n", mpd->user);
      CHECK_ERROR_CLOSE(mbox, mpd, status);
      MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      free (mpd->user);
      mpd->user = NULL;
      mpd->state = POP_AUTH_USER;

    case POP_AUTH_USER:
      /* Send username.  */
      status = pop_write (mpd);
      CHECK_EAGAIN (mpd, status);
      mpd->state = POP_AUTH_USER_ACK;

    case POP_AUTH_USER_ACK:
      /* Get the user ack.  */
      status = pop_read_ack (mpd);
      CHECK_EAGAIN (mpd, status);
      MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      if (mu_c_strncasecmp (mpd->buffer, "+OK", 3) != 0)
	{
	  mu_observable_t observable = NULL;
	  mu_mailbox_get_observable (mbox, &observable);
	  CLEAR_STATE (mpd);
	  mu_observable_notify (observable, MU_EVT_AUTHORITY_FAILED, NULL);
	  CHECK_ERROR_CLOSE (mbox, mpd, EACCES);
	}
      status = pop_get_passwd (auth);
      if (status != 0 || mpd->secret == NULL)
	{
	  pop_writeline (mpd, "QUIT\r\n");
	  MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
	  pop_write (mpd);
	  CHECK_ERROR_CLOSE (mbox, mpd, MU_ERR_NOPASSWORD);
	}
      status = pop_writeline (mpd, "PASS %s\r\n",
			      mu_secret_password (mpd->secret));
      mu_secret_password_unref (mpd->secret);
      mu_secret_unref (mpd->secret);
      mpd->secret = NULL;
      MU_DEBUG (mbox->debug, MU_DEBUG_PROT, "PASS ***\n");
      CHECK_ERROR_CLOSE (mbox, mpd, status);
      mpd->state = POP_AUTH_PASS;
      /* FIXME: Merge these two cases */
	 
    case POP_AUTH_PASS:
      /* Send passwd.  */
      status = pop_write (mpd);
      CHECK_EAGAIN (mpd, status);
      /* Clear the buffer it contains the passwd.  */
      memset (mpd->buffer, '\0', mpd->buflen);
      mpd->state = POP_AUTH_PASS_ACK;

    case POP_AUTH_PASS_ACK:
      /* Get the ack from passwd.  */
      status = pop_read_ack (mpd);
      CHECK_EAGAIN (mpd, status);
      MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      if (mu_c_strncasecmp (mpd->buffer, "+OK", 3) != 0)
	{
	  mu_observable_t observable = NULL;
	  mu_mailbox_get_observable (mbox, &observable);
	  CLEAR_STATE (mpd);
	  mu_observable_notify (observable, MU_EVT_AUTHORITY_FAILED, NULL);
	  return MU_ERR_AUTH_FAILURE;
	}
      mpd->state = POP_AUTH_DONE;
      break;  /* We're outta here.  */

    default:
      break;
    }
  CLEAR_STATE (mpd);
  return 0;
}

int
_pop_apop (mu_authority_t auth)
{
  mu_folder_t folder = mu_authority_get_owner (auth);
  mu_mailbox_t mbox = folder->data;
  pop_data_t mpd = mbox->data;
  int status;

  switch (mpd->state)
    {
    case POP_AUTH:
      /* Fetch the user from them.  */
      status = pop_get_user (auth);
      if (status != 0 || mpd->user == NULL || mpd->user[0] == '\0')
	{
	  CHECK_ERROR_CLOSE (mbox, mpd, EINVAL);
	}

      /* Fetch the secret from them.  */
      status = pop_get_passwd (auth);
      if (status != 0 || mpd->secret == NULL)
	{
	  CHECK_ERROR_CLOSE (mbox, mpd, EINVAL);
	}

      /* Make the MD5 digest string.  */
      status = pop_get_md5 (mpd);
      if (status != 0)
	{
	  CHECK_ERROR_CLOSE (mbox, mpd, status);
	}
      status = pop_writeline (mpd, "APOP %s %s\r\n", mpd->user, mpd->digest);
      MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      /* We have to obscure the md5 string.  */
      memset (mpd->digest, '\0', strlen (mpd->digest));
      free (mpd->user);
      free (mpd->digest);
      mpd->user = NULL;
      mpd->digest = NULL;
      CHECK_ERROR_CLOSE (mbox, mpd, status);
      mpd->state = POP_APOP;

    case POP_APOP:
      /* Send apop.  */
      status = pop_write (mpd);
      CHECK_EAGAIN (mpd, status);
      /* Clear the buffer it contains the md5.  */
      memset (mpd->buffer, '\0', mpd->buflen);
      mpd->state = POP_APOP_ACK;

    case POP_APOP_ACK:
      status = pop_read_ack (mpd);
      CHECK_EAGAIN (mpd, status);
      MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      if (mu_c_strncasecmp (mpd->buffer, "+OK", 3) != 0)
        {
          mu_observable_t observable = NULL;
          mu_mailbox_get_observable (mbox, &observable);
          CLEAR_STATE (mpd);
          mu_observable_notify (observable, MU_EVT_AUTHORITY_FAILED, NULL);
          CHECK_ERROR_CLOSE (mbox, mpd, EACCES);
        }
      mpd->state = POP_AUTH_DONE;
      break;  /* We're outta here.  */

    default:
      break;
    }
  CLEAR_STATE (mpd);
  return 0;
}

/*
  Client side STLS support.
 */

static int
pop_reader (void *iodata)
{
  int status = 0;
  pop_data_t iop = iodata;
  status = pop_read_ack (iop);
  CHECK_EAGAIN (iop, status);
  MU_DEBUG (iop->mbox->debug, MU_DEBUG_PROT, iop->buffer);
  return status;/*mu_c_strncasecmp (iop->buffer, "+OK", 3) == 0;*/
}

static int
pop_writer (void *iodata, char *buf)
{
  pop_data_t iop = iodata;
  int status;
  
  MU_DEBUG1 (iop->mbox->debug, MU_DEBUG_PROT, "%s\n", buf);
  status = pop_writeline (iop, "%s\r\n", buf);
  CHECK_ERROR (iop, status);
  status = pop_write (iop);
  CHECK_ERROR (iop, status);
  return status;
}

static void
pop_stream_ctl (void *iodata, mu_stream_t *pold, mu_stream_t new)
{
  pop_data_t iop = iodata;
  if (pold)
    *pold = iop->mbox->stream;
  if (new)
    iop->mbox->stream = new;
}

static int
pop_stls (mu_mailbox_t mbox)
{
#ifdef WITH_TLS
  int status;
  pop_data_t mpd = mbox->data;
  char *keywords[] = { "STLS", "CAPA", NULL };

  if (!mu_tls_enable || !(mpd->capa & CAPA_STLS))
    return -1;

  status = mu_tls_begin (mpd, pop_reader, pop_writer,
			 pop_stream_ctl, keywords);

  MU_DEBUG1 (mbox->debug, MU_DEBUG_PROT, "TLS negotiation %s\n",
		  status == 0 ? "succeeded" : "failed");

  if (status == 0)
    pop_parse_capa (mpd);

  return status;
#else
  return -1;
#endif /* WITH_TLS */
}

/* Open the connection to the sever, and send the authentication. */
static int
pop_open (mu_mailbox_t mbox, int flags)
{
  pop_data_t mpd = mbox->data;
  int status;
  const char *host;
  long port = mpd->pops ? MU_POPS_PORT : MU_POP_PORT;

  /* Sanity checks.  */
  if (mpd == NULL)
    return EINVAL;

  /* Fetch the pop server name and the port in the mu_url_t.  */
  status = mu_url_sget_host (mbox->url, &host);
  if (status != 0)
    return status;
  mu_url_get_port (mbox->url, &port);

  mbox->flags = flags;

  /* Do not check for reconnect here.  */
  /* CHECK_BUSY (mbox, mpd, func, 0); */

  /* Enter the pop state machine, and boogy: AUTHORISATION State.  */
  switch (mpd->state)
    {
    case POP_NO_STATE:
      /* Allocate a working io buffer.  */
      if (mpd->buffer == NULL)
	{
	  /* 255 is the limit lenght of a POP3 command according to RFCs.  */
	  mpd->buflen = 255;
	  mpd->buffer = calloc (mpd->buflen + 1, sizeof (char));
	  if (mpd->buffer == NULL)
	    {
	      CHECK_ERROR (mpd, ENOMEM);
	    }
	}
      else
	{
	  /* Clear any residual from a previous connection.  */
	  memset (mpd->buffer, '\0', mpd->buflen);
	}
      mpd->ptr = mpd->buffer;

      /* Create the networking stack.  */
      if (mbox->stream == NULL)
	{
	  status = mu_tcp_stream_create (&mbox->stream, host, port, mbox->flags);
	  CHECK_ERROR (mpd, status);

#ifdef WITH_TLS
	  if (mpd->pops)
	    {
	      mu_stream_t newstr;

	      status = mu_stream_open (mbox->stream);
	      CHECK_EAGAIN (mpd, status);
	      CHECK_ERROR_CLOSE (mbox, mpd, status);

	      status = mu_tls_stream_create_client_from_tcp (&newstr, mbox->stream, 0);
	      if (status != 0)
		{
		  mu_error ("pop_open: mu_tls_stream_create_client_from_tcp: %s",
			    mu_strerror (status));
		  return status;
		}
	      mbox->stream = newstr;
	    }
#endif /* WITH_TLS */

	  /* Using the awkward mu_stream_t buffering.  */
	  mu_stream_setbufsiz (mbox->stream, BUFSIZ);
	}
      else
	{
	  /* This is sudden death: for many pop servers, it is important to
	     let them time to remove locks or move the .user.pop files.  This
	     happen when we do BUSY_CHECK().  For example, the user does not
	     want to read the entire file, and wants start to read a new
	     message, closing the connection and immediately contact the
	     server again, and we'll end up having "-ERR Mail Lock busy" or
	     something similar. To prevent this race condition we sleep 2
	     seconds. */
	  mu_stream_close (mbox->stream);
	  pop_sleep (2);
	}
      mpd->state = POP_OPEN_CONNECTION;

    case POP_OPEN_CONNECTION:
      /* Establish the connection.  */
      MU_DEBUG2 (mbox->debug, MU_DEBUG_PROT, "open (%s:%ld)\n", host, port);
      status = mu_stream_open (mbox->stream);
      CHECK_EAGAIN (mpd, status);
      /* Can't recover bailout.  */
      CHECK_ERROR_CLOSE (mbox, mpd, status);
      mpd->state = POP_GREETINGS;

    case POP_GREETINGS:
      {
	int gblen = 0;
	status = pop_read_ack (mpd);
	CHECK_EAGAIN (mpd, status);
	MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
	if (mu_c_strncasecmp (mpd->buffer, "+OK", 3) != 0)
	  {
	    CHECK_ERROR_CLOSE (mbox, mpd, EACCES);
	  }
	gblen = strlen (mpd->buffer);
	mpd->greeting_banner = calloc (gblen, 1);
	if (mpd->greeting_banner == NULL)
	  {
	    CHECK_ERROR (mpd, ENOMEM);
	  }
	memcpy (mpd->greeting_banner, mpd->buffer, gblen);
	mpd->state = POP_CAPA;
      }

    case POP_CAPA:
    case POP_CAPA_ACK:
      pop_capa (mbox);
      mpd->state = POP_STLS;

    case POP_STLS:
    case POP_STLS_ACK:
      if (!mpd->pops)
	pop_stls (mbox);
      mpd->state = POP_AUTH;

    case POP_AUTH:
    case POP_AUTH_USER:
    case POP_AUTH_USER_ACK:
    case POP_AUTH_PASS:
    case POP_AUTH_PASS_ACK:
    case POP_APOP:
    case POP_APOP_ACK:
      /* Authenticate.  */
      status = mu_authority_authenticate (mbox->folder->authority);
      CHECK_EAGAIN (mpd, status);

    case POP_AUTH_DONE:
      break;

    default:
      /*
	mu_error ("pop_open: unknown state");
      */
      break;
    }/* End AUTHORISATION state. */

  /* Clear any state.  */
  CLEAR_STATE (mpd);
  return 0;
}

/* Send the QUIT and close the socket.  */
static int
pop_close (mu_mailbox_t mbox)
{
  pop_data_t mpd = mbox->data;
  void *func = (void *)pop_close;
  int status;
  size_t i;

  if (mpd == NULL)
    return EINVAL;

  /* Should not check for Busy, we're shuting down anyway.  */
  /* CHECK_BUSY (mbox, mpd, func, 0); */
  mu_monitor_wrlock (mbox->monitor);
  if (mpd->func && mpd->func != func)
    mpd->state = POP_NO_STATE;
  mpd->id = 0;
  mpd->func = func;
  mu_monitor_unlock (mbox->monitor);

  /*  Ok boys, it's a wrap: UPDATE State.  */
  switch (mpd->state)
    {
    case POP_NO_STATE:
      /* Initiate the close.  */
      status = pop_writeline (mpd, "QUIT\r\n");
      CHECK_ERROR (mpd, status);
      MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      mpd->state = POP_QUIT;

    case POP_QUIT:
      /* Send the quit.  */
      status = pop_write (mpd);
      CHECK_EAGAIN (mpd, status);
      mpd->state = POP_QUIT_ACK;

    case POP_QUIT_ACK:
      /* Glob the acknowledge.  */
      status = pop_read_ack (mpd);
      CHECK_EAGAIN (mpd, status);
      MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      /*  Now what ! and how can we tell them about errors ?  So far now
	  lets just be verbose about the error but close the connection
	  anyway.  */
      if (mu_c_strncasecmp (mpd->buffer, "+OK", 3) != 0)
	mu_error ("pop_close: %s", mpd->buffer);
      mu_stream_close (mbox->stream);
      break;

    default:
      /*
	mu_error ("pop_close: unknown state");
      */
      break;
    } /* UPDATE state.  */

  /* Free the messages.  */
  for (i = 0; i < mpd->pmessages_count; i++)
    {
      if (mpd->pmessages[i])
	{
	  mu_message_destroy (&(mpd->pmessages[i]->message),
			   mpd->pmessages[i]);
	  if (mpd->pmessages[i]->uidl)
	    free (mpd->pmessages[i]->uidl);
	  free (mpd->pmessages[i]);
	  mpd->pmessages[i] = NULL;
	}
    }
  /* And clear any residue.  */
  if (mpd->greeting_banner)
    free (mpd->greeting_banner);
  mpd->greeting_banner = NULL;
  if (mpd->pmessages)
    free (mpd->pmessages);
  mpd->pmessages = NULL;
  mpd->pmessages_count = 0;
  mpd->is_updated = 0;
  if (mpd->buffer)
    free (mpd->buffer);
  mpd->buffer = NULL;

  CLEAR_STATE (mpd);
  return 0;
}

/*  Only build/setup the mu_message_t structure for a mesgno. pop_message_t,
    will act as the owner of messages.  */
static int
pop_get_message (mu_mailbox_t mbox, size_t msgno, mu_message_t *pmsg)
{
  pop_data_t mpd = mbox->data;
  mu_message_t msg = NULL;
  pop_message_t mpm;
  int status;
  size_t i;

  /* Sanity.  */
  if (pmsg == NULL || mpd == NULL)
    return EINVAL;

  /* If we did not start a scanning yet do it now.  */
  if (!pop_is_updated (mbox))
    pop_scan (mbox, 1, NULL);

  if (msgno > mpd->messages_count)
    return EINVAL;

  mu_monitor_rdlock (mbox->monitor);
  /* See if we have already this message.  */
  for (i = 0; i < mpd->pmessages_count; i++)
    {
      if (mpd->pmessages[i])
	{
	  if (mpd->pmessages[i]->num == msgno)
	    {
	      *pmsg = mpd->pmessages[i]->message;
	      mu_monitor_unlock (mbox->monitor);
	      return 0;
	    }
	}
    }
  mu_monitor_unlock (mbox->monitor);

  mpm = calloc (1, sizeof (*mpm));
  if (mpm == NULL)
    return ENOMEM;

  /* Back pointer.  */
  mpm->mpd = mpd;
  mpm->num = msgno;

  /* Create the message.  */
  {
    mu_stream_t stream = NULL;
    if ((status = mu_message_create (&msg, mpm)) != 0
	|| (status = mu_stream_create (&stream, mbox->flags, msg)) != 0)
      {
	mu_stream_destroy (&stream, msg);
	mu_message_destroy (&msg, mpm);
	free (mpm);
	return status;
      }
    /* Help for the readline()s  */
    mu_stream_setbufsiz (stream, 128);
    mu_stream_set_read (stream, pop_message_read, msg);
    mu_stream_set_get_transport2 (stream, pop_message_transport, msg);
    mu_message_set_stream (msg, stream, mpm);
    mu_message_set_size (msg, pop_message_size, mpm);
  }

  /* Create the header.  */
  {
    mu_header_t header = NULL;
    if ((status = mu_header_create (&header, NULL, 0,  msg)) != 0)
      {
	mu_message_destroy (&msg, mpm);
	free (mpm);
	return status;
      }
    mu_header_set_fill (header, pop_top, msg);
    mu_message_set_header (msg, header, mpm);
  }

  /* Create the attribute.  */
  {
    mu_attribute_t attribute;
    status = mu_attribute_create (&attribute, msg);
    if (status != 0)
      {
	mu_message_destroy (&msg, mpm);
	free (mpm);
	return status;
      }
    mu_attribute_set_get_flags (attribute, pop_get_attribute, msg);
    mu_attribute_set_set_flags (attribute, pop_set_attribute, msg);
    mu_attribute_set_unset_flags (attribute, pop_unset_attribute, msg);
    mu_message_set_attribute (msg, attribute, mpm);
  }

  /* Create the body and its stream.  */
  {
    mu_body_t body = NULL;
    mu_stream_t stream = NULL;
    if ((status = mu_body_create (&body, msg)) != 0
	|| (status = mu_stream_create (&stream, mbox->flags, body)) != 0)
      {
	mu_body_destroy (&body, msg);
	mu_stream_destroy (&stream, body);
	mu_message_destroy (&msg, mpm);
	free (mpm);
	return status;
      }
    /* Helps for the readline()s  */
    mu_stream_setbufsiz (stream, 128);
    mu_stream_set_read (stream, pop_body_read, body);
    mu_stream_set_get_transport2 (stream, pop_body_transport, body);
    mu_body_set_size (body, pop_body_size, msg);
    mu_body_set_lines (body, pop_body_lines, msg);
    mu_body_set_stream (body, stream, msg);
    mu_message_set_body (msg, body, mpm);
  }

  /* Set the UIDL call on the message. */
  if (mpd->capa & CAPA_UIDL)
    mu_message_set_uidl (msg, pop_uidl, mpm);
  
  /* Set the UID on the message. */
  mu_message_set_uid (msg, pop_uid, mpm);

  /* Add it to the list.  */
  mu_monitor_wrlock (mbox->monitor);
  {
    pop_message_t *m ;
    m = realloc (mpd->pmessages, (mpd->pmessages_count + 1)*sizeof (*m));
    if (m == NULL)
      {
	mu_message_destroy (&msg, mpm);
	free (mpm);
	mu_monitor_unlock (mbox->monitor);
	return ENOMEM;
      }
    mpd->pmessages = m;
    mpd->pmessages[mpd->pmessages_count] = mpm;
    mpd->pmessages_count++;
  }
  mu_monitor_unlock (mbox->monitor);

  /* Save The message pointer.  */
  mu_message_set_mailbox (msg, mbox, mpm);
  *pmsg = mpm->message = msg;

  return 0;
}

/* FIXME: Should use strtoumax ideally */
static int
parse_answer0 (const char *buffer, size_t *n1, size_t *n2)
{
  char *p;
  unsigned long m;
  if (strlen (buffer) < 3 || memcmp (buffer, "+OK", 3))
    return 1;
  m = *n1 = strtoul (buffer + 3, &p, 10);
  if (!mu_isspace (*p) || m != *n1)
    return 1;
  m = *n2 = strtoul (p, &p, 10);
  if (!(*p == 0 || mu_isspace (*p)) || m != *n2)
    return 1;
  return 0;
}

/* FIXME: Should use strtoumax ideally */
static int
parse_answer1 (const char *buffer, size_t *n1, char *buf, size_t bufsize)
{
  char *p;
  unsigned long m;
  if (strlen (buffer) < 3 || memcmp (buffer, "+OK", 3))
    return 1;
  m = *n1 = strtoul (buffer + 3, &p, 0);
  if (!mu_isspace (*p) || m != *n1)
    return 1;
  while (*p && mu_isspace (*p))
    p++;
  if (strlen (p) >= bufsize)
    return 1;
  strcpy (buf, p);
  return 0;
}
  
/* There is no such thing in pop all messages should be consider recent.
   FIXME: We could cheat and peek at the status if it was not strip
   by the server ...  */
static int
pop_messages_recent (mu_mailbox_t mbox, size_t *precent)
{
  return pop_messages_count (mbox, precent);
}

/* There is no such thing in pop all messages should be consider unseen.
   FIXME: We could cheat and peek at the status if it was not strip
   by the server ...  */
static int
pop_message_unseen (mu_mailbox_t mbox, size_t *punseen)
{
  size_t count = 0;
  int status = pop_messages_count (mbox, &count);
  if (status != 0)
    return status;
  if (punseen)
    *punseen = (count > 0) ? 1 : 0;
  return 0;
}

/*  How many messages we have.  Done with STAT.  */
static int
pop_messages_count (mu_mailbox_t mbox, size_t *pcount)
{
  pop_data_t mpd = mbox->data;
  int status;
  void *func = (void *)pop_messages_count;

  if (mpd == NULL)
    return EINVAL;

  /* Do not send a STAT if we know the answer.  */
  if (pop_is_updated (mbox))
    {
      if (pcount)
	*pcount = mpd->messages_count;
      return 0;
    }

  /* Flag busy.  */
  CHECK_BUSY (mbox, mpd, func, 0);

  /* TRANSACTION state.  */
  switch (mpd->state)
    {
    case POP_NO_STATE:
      status = pop_writeline (mpd, "STAT\r\n");
      CHECK_ERROR (mpd, status);
      MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      mpd->state = POP_STAT;

    case POP_STAT:
      /* Send the STAT.  */
      status = pop_write (mpd);
      CHECK_EAGAIN (mpd, status);
      mpd->state = POP_STAT_ACK;

    case POP_STAT_ACK:
      /* Get the ACK.  */
      status = pop_read_ack (mpd);
      CHECK_EAGAIN (mpd, status);
      MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      break;

    default:
      /*
	mu_error ("pop_messages_count: unknown state");
      */
      break;
    }


  /* Parse the answer.  */

  status = parse_answer0 (mpd->buffer, &mpd->messages_count, &mpd->size);
  /*  Clear the state _after_ the scanf, since another thread could
      start writing over mpd->buffer.  */
  CLEAR_STATE (mpd);

  if (status)
    return EIO;

  if (pcount)
    *pcount = mpd->messages_count;
  mpd->is_updated = 1;
  return 0;
}

/* Update and scanning.  */
static int
pop_is_updated (mu_mailbox_t mbox)
{
  pop_data_t mpd = mbox->data;
  if (mpd == NULL)
    return 0;
  return mpd->is_updated;
}

/* We just simulate by sending a notification for the total msgno.  */
/* FIXME is message is set deleted should we sent a notif ?  */
static int
pop_scan (mu_mailbox_t mbox, size_t msgno, size_t *pcount)
{
  int status;
  size_t i;
  size_t count = 0;

  status = pop_messages_count (mbox, &count);
  if (pcount)
    *pcount = count;
  if (status != 0)
    return status;
  if (mbox->observable == NULL)
    return 0;
  for (i = msgno; i <= count; i++)
    {
      size_t tmp = i;
      if (mu_observable_notify (mbox->observable, MU_EVT_MESSAGE_ADD,
				&tmp) != 0)
	break;
      if (((i +1) % 10) == 0)
	{
	  mu_observable_notify (mbox->observable, MU_EVT_MAILBOX_PROGRESS,
				NULL);
	}
    }
  return 0;
}

/* This is where we actually send the DELE command. Meaning that when
   the attribute on the message is set deleted the comand DELE is not
   sent right away and if we did there is no way to mark a message undeleted
   beside closing down the connection without going to the update state via
   QUIT.  So DELE is send only when in expunge.  */
static int
pop_expunge (mu_mailbox_t mbox)
{
  pop_data_t mpd = mbox->data;
  size_t i;
  mu_attribute_t attr;
  int status;
  void *func = (void *)pop_expunge;

  if (mpd == NULL)
    return EINVAL;

  /* Busy ?  */
  CHECK_BUSY (mbox, mpd, func, 0);

  for (i = (int)mpd->id; i < mpd->pmessages_count; mpd->id = ++i)
    {
      if (mu_message_get_attribute (mpd->pmessages[i]->message, &attr) == 0)
	{
	  if (mu_attribute_is_deleted (attr))
	    {
	      switch (mpd->state)
		{
		case POP_NO_STATE:
		  status = pop_writeline (mpd, "DELE %lu\r\n",
					  (unsigned long)
					    mpd->pmessages[i]->num);
		  CHECK_ERROR (mpd, status);
		  MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
		  mpd->state = POP_DELE;

		case POP_DELE:
		  /* Send DELETE.  */
		  status = pop_write (mpd);
		  CHECK_EAGAIN (mpd, status);
		  mpd->state = POP_DELE_ACK;

		case POP_DELE_ACK:
		  /* Ack Delete.  */
		  status = pop_read_ack (mpd);
		  CHECK_EAGAIN (mpd, status);
		  MU_DEBUG (mbox->debug, MU_DEBUG_PROT, mpd->buffer);
		  if (mu_c_strncasecmp (mpd->buffer, "+OK", 3) != 0)
		    {
		      CHECK_ERROR (mpd, ERANGE);
		    }
		  mpd->state = POP_NO_STATE;
		  break;

		default:
		  /* mu_error ("pop_expunge: unknown state); */
		  break;
		} /* switch (state) */
	    } /* if mu_attribute_is_deleted() */
	} /* mu_message_get_attribute() */
    } /* for */
  CLEAR_STATE (mpd);
  /* Invalidate.  But Really they should shutdown the channel POP protocol
     is not meant for this like IMAP.  */
  mpd->is_updated = 0;
  return 0;
}

/* Mailbox size ? It is part of the STAT command */
static int
pop_get_size (mu_mailbox_t mbox, mu_off_t *psize)
{
  pop_data_t mpd = mbox->data;
  int status = 0;

  if (mpd == NULL)
    return EINVAL;

  if (!pop_is_updated (mbox))
    status = pop_messages_count (mbox, &mpd->size);
  if (psize)
    *psize = mpd->size;
  return status;
}

/* Form the RFC:
   "It is important to note that the octet count for a message on the
   server host may differ from the octet count assigned to that message
   due to local conventions for designating end-of-line.  Usually,
   during the AUTHORIZATION state of the POP3 session, the POP3 server
   can calculate the size of each message in octets when it opens the
   maildrop.  For example, if the POP3 server host internally represents
   end-of-line as a single character, then the POP3 server simply counts
   each occurrence of this character in a message as two octets."

   This is not perfect if we do not know the number of lines in the message
   then the octets returned will not be correct so we do our best.
 */
static int
pop_message_size (mu_message_t msg, size_t *psize)
{
  pop_message_t mpm = mu_message_get_owner (msg);
  pop_data_t mpd;
  int status = 0;
  void *func = (void *)pop_message_size;
  size_t num;

  if (mpm == NULL)
    return EINVAL;

  /* Did we have it already ?  */
  if (mpm->mu_message_size != 0)
    {
      *psize = mpm->mu_message_size;
      return 0;
    }

  mpd = mpm->mpd;
  /* Busy ? */
  CHECK_BUSY (mpd->mbox, mpd, func, msg);

  /* Get the size.  */
  switch (mpd->state)
    {
    case POP_NO_STATE:
      status = pop_writeline (mpd, "LIST %lu\r\n", (unsigned long) mpm->num);
      CHECK_ERROR (mpd, status);
      MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      mpd->state = POP_LIST;

    case POP_LIST:
      /* Send the LIST.  */
      status = pop_write (mpd);
      CHECK_EAGAIN (mpd, status);
      mpd->state = POP_LIST_ACK;

    case POP_LIST_ACK:
      /* Resp from LIST. */
      status = pop_read_ack (mpd);
      CHECK_EAGAIN (mpd, status);
      MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      break;

    default:
      /*
	mu_error ("pop_message_size: unknown state");
      */
      break;
    }

  /* FIXME */
  status = parse_answer0 (mpd->buffer, &num, &mpm->mu_message_size);
  CLEAR_STATE (mpd);

  if (status != 0)
    return MU_ERR_PARSE;

  /* The size of the message is with the extra '\r' octet for everyline.
     Substract to get, hopefully, a good count.  */
  if (psize)
    *psize = mpm->mu_message_size - (mpm->header_lines + mpm->body_lines);
  return 0;
}

/* Another source of trouble, POP only gives the size of the message
   not the size of subparts like headers, body etc .. Again we're doing
   our best with what we know but the only way to get a precise number
   is by dowloading the whole message.  */
static int
pop_body_size (mu_body_t body, size_t *psize)
{
  mu_message_t msg = mu_body_get_owner (body);
  pop_message_t mpm = mu_message_get_owner (msg);

  if (mpm == NULL)
    return EINVAL;

  /* Did we have it already ?  */
  if (mpm->body_size != 0)
    {
      *psize = mpm->body_size;
    }
  else if (mpm->mu_message_size != 0)
    {
      /* Take a guest.  */
      *psize = mpm->mu_message_size - mpm->header_size - mpm->body_lines;
    }
  else
    *psize = 0;

  return 0;
}

/* Not know until the whole message get downloaded.  */
static int
pop_body_lines (mu_body_t body, size_t *plines)
{
  mu_message_t msg = mu_body_get_owner (body);
  pop_message_t mpm = mu_message_get_owner (msg);
  if (mpm == NULL)
    return EINVAL;
  if (plines)
    *plines = mpm->body_lines;
  return 0;
}

/* Pop does not have any command for this, We fake by reading the "Status: "
   header.  But this is hackish some POP server(Qpopper) skip it.  Also
   because we call mu_header_get_value the function may return EAGAIN... uncool.
   To put it another way, many servers simply remove the "Status:" header
   field, when you dowload a message, so a message will always look like
   new even if you already read it.  There is also no way to set an attribute
   on remote mailbox via the POP server and many server once you do a RETR
   and in some cases a TOP will mark the message as read; "Status: RO"
   or maybe worst some ISP configure there servers to delete after
   the RETR some go as much as deleting after the TOP, since technicaly
   you can download a message via TOP without RET'reiving it.  */
static int
pop_get_attribute (mu_attribute_t attr, int *pflags)
{
  mu_message_t msg = mu_attribute_get_owner (attr);
  pop_message_t mpm = mu_message_get_owner (msg);
  char hdr_status[64];
  mu_header_t header = NULL;

  if (mpm == NULL || pflags == NULL)
    return EINVAL;
  if (mpm->attr_flags == 0)
    {
      hdr_status[0] = '\0';
      mu_message_get_header (mpm->message, &header);
      mu_header_get_value (header, "Status", hdr_status, sizeof hdr_status, NULL);
      mu_string_to_flags (hdr_status, &(mpm->attr_flags));
    }
  *pflags = mpm->attr_flags;
  return 0;
}

static int
pop_set_attribute (mu_attribute_t attr, int flags)
{
  mu_message_t msg = mu_attribute_get_owner (attr);
  pop_message_t mpm = mu_message_get_owner (msg);

  if (mpm == NULL)
    return EINVAL;
  mpm->attr_flags |= flags;
  return 0;
}

static int
pop_unset_attribute (mu_attribute_t attr, int flags)
{
  mu_message_t msg = mu_attribute_get_owner (attr);
  pop_message_t mpm = mu_message_get_owner (msg);

  if (mpm == NULL)
    return EINVAL;
  mpm->attr_flags &= ~flags;
  return 0;
}

/* Stub to call the fd from body object.  */
static int
pop_body_transport (mu_stream_t stream, mu_transport_t *ptr, mu_transport_t *ptr2)
{
  mu_body_t body = mu_stream_get_owner (stream);
  mu_message_t msg = mu_body_get_owner (body);
  pop_message_t mpm = mu_message_get_owner (msg);
  return pop_get_transport2 (mpm, ptr, ptr2);
}

/* Stub to call the fd from message object.  */
static int
pop_message_transport (mu_stream_t stream, mu_transport_t *ptr, mu_transport_t *ptr2)
{
  mu_message_t msg = mu_stream_get_owner (stream);
  pop_message_t mpm = mu_message_get_owner (msg);
  return pop_get_transport2 (mpm, ptr, ptr2);
}

static int
pop_get_transport2 (pop_message_t mpm, mu_transport_t *ptr, mu_transport_t *ptr2)
{
  if (mpm && mpm->mpd && mpm->mpd->mbox)
	return mu_stream_get_transport2 (mpm->mpd->mbox->stream, ptr, ptr2);
  return EINVAL;
}

static int
pop_uid (mu_message_t msg,  size_t *puid)
{
  pop_message_t mpm = mu_message_get_owner (msg);
  if (puid)
    *puid = mpm->num;
  return 0;
}

/* Get the UIDL.  The client should be prepared, since it may fail.  UIDL is
   optional on many POP servers.
   FIXME:  We should check the "mpd->capa & CAPA_UIDL" and fall back to
   a md5 scheme ? Or maybe check for "X-UIDL" a la Qpopper ?  */
static int
pop_uidl (mu_message_t msg, char *buffer, size_t buflen, size_t *pnwriten)
{
  pop_message_t mpm = mu_message_get_owner (msg);
  pop_data_t mpd;
  int status = 0;
  void *func = (void *)pop_uidl;
  size_t num;
  char uniq[MU_UIDL_BUFFER_SIZE];

  if (mpm == NULL)
    return EINVAL;

  /* Is it cached ?  */
  if (mpm->uidl)
    {
      size_t len = strlen (mpm->uidl);
      if (buffer)
	{
	  buflen--; /* Leave space for the null.  */
	  buflen = (len > buflen) ? buflen : len;
	  memcpy (buffer, mpm->uidl, buflen);
	  buffer[buflen] = '\0';
	}
      else
	buflen = len;
      if (pnwriten)
	*pnwriten = buflen;
      return 0;
    }

  mpd = mpm->mpd;

  /* Busy ? */
  CHECK_BUSY (mpd->mbox, mpd, func, 0);

  /* Get the UIDL.  */
  switch (mpd->state)
    {
    case POP_NO_STATE:
      status = pop_writeline (mpd, "UIDL %lu\r\n", (unsigned long) mpm->num);
      CHECK_ERROR (mpd, status);
      MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      mpd->state = POP_UIDL;

    case POP_UIDL:
      /* Send the UIDL.  */
      status = pop_write (mpd);
      CHECK_EAGAIN (mpd, status);
      mpd->state = POP_UIDL_ACK;

    case POP_UIDL_ACK:
      /* Resp from UIDL. */
      status = pop_read_ack (mpd);
      CHECK_EAGAIN (mpd, status);
      MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      break;

    default:
      /*
	mu_error ("pop_uidl: unknown state");
      */
      break;
    }

  /* FIXME:  I should cache the result.  */
  *uniq = '\0';
  status = parse_answer1 (mpd->buffer, &num, uniq, sizeof uniq);
  if (status)
    {
      status = MU_ERR_PARSE;
      buflen = 0;
    }
  else
    {
      num = strlen (uniq);
      uniq[num - 1] = '\0'; /* Nuke newline.  */
      if (buffer)
	{
	  buflen--; /* Leave space for the null.  */
	  buflen = (buflen < num) ? buflen : num;
	  memcpy (buffer, uniq, buflen);
	  buffer [buflen] = '\0';
	}
      else
	buflen = num - 1; /* Do not count newline.  */
      mpm->uidl = strdup (uniq);
      status = 0;
    }

  CLEAR_STATE (mpd);

  if (pnwriten)
    *pnwriten = buflen;
  return status;
}

/* How we retrieve the headers.  If it fails we jump to the pop_retr()
   code i.e. send a RETR and skip the body, ugly.
   NOTE: if the offset is different, flag an error, offset is meaningless
   on a socket but we better warn them, some stuff like mu_mime_t may try to
   read ahead, for example for the headers.  */
static int
pop_top (mu_header_t header, char *buffer, size_t buflen,
	 mu_off_t offset, size_t *pnread)
{
  mu_message_t msg = mu_header_get_owner (header);
  pop_message_t mpm = mu_message_get_owner (msg);
  pop_data_t mpd;
  size_t nread = 0;
  int status = 0;
  void *func = (void *)pop_top;

  if (mpm == NULL)
    return EINVAL;

  mpd = mpm->mpd;

  /* Busy ? */
  CHECK_BUSY (mpd->mbox, mpd, func, msg);

  /* We start fresh then reset the sizes.  */
  if (mpd->state == POP_NO_STATE)
    mpm->header_size = 0;

  /* Throw an error if trying to seek back.  */
  if ((size_t)offset < mpm->header_size)
    return ESPIPE;

  /* Get the header.  */
  switch (mpd->state)
    {
    case POP_NO_STATE:
      if (mpd->capa & CAPA_TOP)
        {
	  status = pop_writeline (mpd, "TOP %lu 0\r\n",
				  (unsigned long) mpm->num);
	  CHECK_ERROR (mpd, status);
	  MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);
	  mpd->state = POP_TOP;
	}
      else /* Fall back to RETR call.  */
        {
	  mpd->state = POP_NO_STATE;
	  mpm->skip_header = 0;
	  mpm->skip_body = 1;
	  return pop_retr (mpm, buffer, buflen, offset, pnread);
        }

    case POP_TOP:
      /* Send the TOP.  */
      status = pop_write (mpd);
      CHECK_EAGAIN (mpd, status);
      mpd->state = POP_TOP_ACK;

    case POP_TOP_ACK:
      /* Ack from TOP. */
      status = pop_read_ack (mpd);
      CHECK_EAGAIN (mpd, status);
      MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      if (mu_c_strncasecmp (mpd->buffer, "+OK", 3) != 0)
	CHECK_ERROR (mpd, EINVAL);
      mpd->state = POP_TOP_RX;

    case POP_TOP_RX:
      /* Get the header.  */
      do
	{
	  /* Seek in position.  */
	  ssize_t pos = offset - mpm->header_size;
	  /* Do we need to fill up.  */
	  if (mpd->nl == NULL || mpd->ptr == mpd->buffer)
	    {
	      status = pop_readline (mpd);
	      CHECK_EAGAIN (mpd, status);
	      mpm->header_lines++;
	    }
	  /* If we have to skip some data to get to the offset.  */
	  if (pos > 0)
	    nread = fill_buffer (mpd, NULL, pos);
	  else
	    nread = fill_buffer (mpd, buffer, buflen);
	  mpm->header_size += nread;
	}
      while (nread > 0 && (size_t)offset > mpm->header_size);
      break;

    default:
      /* Probaly TOP was not supported so we have fall back to RETR.  */
      mpm->skip_header = 0;
      mpm->skip_body = 1;
      return pop_retr (mpm, buffer, buflen, offset, pnread);
    } /* switch (state) */

  if (nread == 0)
    {
      CLEAR_STATE (mpd);
    }
  if (pnread)
    *pnread = nread;
  return 0;
}

/* This is no longer use, see pop_top to retreive headers, we still
   keep it around for debugging purposes.  */
#if 0
/* Stub to call pop_retr ().   Call form the stream object of the header.  */
static int
pop_header_read (mu_header_t header, char *buffer, size_t buflen, mu_off_t offset,
		 size_t *pnread)
{
  mu_message_t msg = mu_header_get_owner (header);
  pop_message_t mpm = mu_message_get_owner (msg);
  pop_data_t mpd;
  void *func = (void *)pop_header_read;

  if (mpm == NULL)
    return EINVAL;

  mpd = mpm->mpd;

  /* Busy ? */
  CHECK_BUSY (mpd->mbox, mpd, func, msg);

  /* We start fresh then reset the sizes.  */
  if (mpd->state == POP_NO_STATE)
    mpm->header_size = mpm->inbody = 0;

  /* Throw an error if trying to seek back.  */
  if ((size_t)offset < mpm->header_size)
    return ESPIPE;

  mpm->skip_header = 0;
  mpm->skip_body = 1;
  return pop_retr (mpm, buffer, buflen, offset, pnread);
}
#endif

/* Stub to call pop_retr (). Call from the stream object of the body.  */
static int
pop_body_read (mu_stream_t is, char *buffer, size_t buflen, mu_off_t offset,
	       size_t *pnread)
{
  mu_body_t body = mu_stream_get_owner (is);
  mu_message_t msg = mu_body_get_owner (body);
  pop_message_t mpm = mu_message_get_owner (msg);
  pop_data_t mpd;
  void *func = (void *)pop_body_read;

  if (mpm == NULL)
    return EINVAL;

  mpd = mpm->mpd;

  /* Busy ? */
  CHECK_BUSY (mpd->mbox, mpd, func, msg);

  /* We start fresh then reset the sizes.  */
  if (mpd->state == POP_NO_STATE)
    mpm->body_size = mpm->inbody = 0;

  /* Can not seek back this a stream socket.  */
  if ((size_t)offset < mpm->body_size)
    return ESPIPE;

  mpm->skip_header = 1;
  mpm->skip_body = 0;
  return pop_retr (mpm, buffer, buflen, offset, pnread);
}

/* Stub to call pop_retr (), calling from the stream object of a message.  */
static int
pop_message_read (mu_stream_t is, char *buffer, size_t buflen, mu_off_t offset,
		  size_t *pnread)
{
  mu_message_t msg = mu_stream_get_owner (is);
  pop_message_t mpm = mu_message_get_owner (msg);
  pop_data_t mpd;
  void *func = (void *)pop_message_read;

  if (mpm == NULL)
    return EINVAL;

  mpd = mpm->mpd;

  /* Busy ? */
  CHECK_BUSY (mpd->mbox, mpd, func, msg);

  /* We start fresh then reset the sizes.  */
  if (mpd->state == POP_NO_STATE)
    mpm->header_size = mpm->body_size = mpm->inbody = 0;

  /* Can not seek back this is a stream socket.  */
  if ((size_t)offset < (mpm->body_size + mpm->header_size))
    return ESPIPE;

  mpm->skip_header = mpm->skip_body = 0;
  return pop_retr (mpm, buffer, buflen, offset, pnread);
}

/* Little helper to fill the buffer without overflow.  */
static int
fill_buffer (pop_data_t mpd, char *buffer, size_t buflen)
{
  int nleft, n, nread = 0;

  /* How much we can copy ?  */
  n = mpd->ptr - mpd->buffer;
  nleft = buflen - n;

 /* We got more then requested.  */
  if (nleft < 0)
    {
      size_t sentinel;
      nread = buflen;
      sentinel = mpd->ptr - (mpd->buffer + nread);
      if (buffer)
	memcpy (buffer, mpd->buffer, nread);
      memmove (mpd->buffer, mpd->buffer + nread, sentinel);
      mpd->ptr = mpd->buffer + sentinel;
    }
  else
    {
      /* Drain our buffer.  */;
      nread = n;
      if (buffer)
	memcpy (buffer, mpd->buffer, nread);
      mpd->ptr = mpd->buffer;
    }

  return nread;
}

/* The heart of most funtions.  Send the RETR and skip different parts.  */
static int
pop_retr (pop_message_t mpm, char *buffer, size_t buflen,  
          mu_off_t offset MU_ARG_UNUSED, size_t *pnread)
{
  pop_data_t mpd;
  size_t nread = 0;
  int status = 0;
  size_t oldbuflen = buflen;

  mpd = mpm->mpd;

  if (pnread)
    *pnread = nread;

  /*  Take care of the obvious.  */
  if (buffer == NULL || buflen == 0)
    {
      CLEAR_STATE (mpd);
      return 0;
    }

  /* pop_retr() is not call directly so we assume that the locks were set.  */

  switch (mpd->state)
    {
    case POP_NO_STATE:
      mpm->body_lines = mpm->body_size = 0;
      status = pop_writeline (mpd, "RETR %lu\r\n",
			      (unsigned long) mpm->num);
      MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);
      CHECK_ERROR (mpd, status);
      mpd->state = POP_RETR;

    case POP_RETR:
      /* Send the RETR command.  */
      status = pop_write (mpd);
      CHECK_EAGAIN (mpd, status);
      mpd->state = POP_RETR_ACK;

    case POP_RETR_ACK:
      /* RETR ACK.  */
      status = pop_read_ack (mpd);
      CHECK_EAGAIN (mpd, status);
      MU_DEBUG (mpd->mbox->debug, MU_DEBUG_PROT, mpd->buffer);

      if (mu_c_strncasecmp (mpd->buffer, "+OK", 3) != 0)
	{
	  CHECK_ERROR (mpd, EACCES);
	}
      mpd->state = POP_RETR_RX_HDR;

    case POP_RETR_RX_HDR:
      /* Skip/Take the header.  */
      while (!mpm->inbody)
        {
	  /* Do we need to fill up.  */
	  if (mpd->nl == NULL || mpd->ptr == mpd->buffer)
	    {
	      status = pop_readline (mpd);
	      if (status != 0)
		{
		  /* Do we have something in the buffer flush it first.  */
		  if (buflen != oldbuflen)
		    return 0;
		  CHECK_EAGAIN (mpd, status);
		}
	      mpm->header_lines++;
	    }
	  /* Oops !! Hello houston we have a major problem here.  */
	  if (mpd->buffer[0] == '\0')
	    {
	      /* Still Do the right thing.  */
	      if (buflen != oldbuflen)
		{
		  CLEAR_STATE (mpd);
		}
	      else
		mpd->state = POP_STATE_DONE;
	      return 0;
	    }
	  /* The problem is that we are using RETR instead of TOP to retreive
	     headers, i.e the server contacted does not support it.  So we
	     have to make sure that the next line really means end of the
	     headers.  Still some cases we may loose.  But 99.9% of POPD
	     encounter support TOP.  In the 0.1% case get GNU pop3d, or the
	     hack below will suffice.  */
	  if (mpd->buffer[0] == '\n' && mpd->buffer[1] == '\0')
	    mpm->inbody = 1; /* break out of the while.  */
	  if (!mpm->skip_header)
	    {
	      ssize_t pos = offset  - mpm->header_size;
	      if (pos > 0)
		{
		  nread = fill_buffer (mpd, NULL, pos);
		  mpm->header_size += nread;
		}
	      else
		{
		  nread = fill_buffer (mpd, buffer, buflen);
		  mpm->header_size += nread;
		  if (pnread)
		    *pnread += nread;
		  buflen -= nread ;
		  if (buflen > 0)
		    buffer += nread;
		  else
		    return 0;
		}
	    }
	  else
	    mpd->ptr = mpd->buffer;
	}
      mpd->state = POP_RETR_RX_BODY;

    case POP_RETR_RX_BODY:
      /* Start/Take the body.  */
      while (mpm->inbody)
	{
	  /* Do we need to fill up.  */
	  if (mpd->nl == NULL || mpd->ptr == mpd->buffer)
	    {
	      status = pop_readline (mpd);
	      if (status != 0)
		{
		  /* Flush The Buffer ?  */
		  if (buflen != oldbuflen)
		    return 0;
		  CHECK_EAGAIN (mpd, status);
		}
	      mpm->body_lines++;
	    }

	    if (mpd->buffer[0] == '\0')
	      mpm->inbody = 0; /* Breakout of the while.  */

	    if (!mpm->skip_body)
	      {
		/* If we did not skip the header, it means that we are
		   downloading the entire message and the mu_header_size should be
		   part of the offset count.  */
		ssize_t pos = offset - (mpm->body_size + ((mpm->skip_header) ?
					0 : mpm->header_size));
		if (pos > 0)
		  {
		    nread = fill_buffer (mpd, NULL, pos);
		    mpm->body_size += nread;
		  }
		else
		  {
		    nread = fill_buffer (mpd, buffer, buflen);
		    mpm->body_size += nread;
		    if (pnread)
		      *pnread += nread;
		    buflen -= nread ;
		    if (buflen > 0)
		      buffer += nread;
		    else
		      return 0;
		  }
	      }
	    else
	      {
		mpm->body_size += (mpd->ptr - mpd->buffer);
		mpd->ptr = mpd->buffer;
	      }
	  }
      mpm->mu_message_size = mpm->body_size + mpm->header_size;
      mpd->state = POP_STATE_DONE;
      /* Return here earlier, because we want to return nread = 0 to notify
	 the callee that we've finish, since there is already data
	 we have to return them first and _then_ tell them its finish.  If we
	 don't we will start over again by sending another RETR.  */
      if (buflen != oldbuflen)
	return 0;

    case POP_STATE_DONE:
      /* A convenient break, this is here we can return 0, we're done.  */

    default:
      /* mu_error ("pop_retr: unknown state"); */
      break;
    } /* Switch state.  */

  CLEAR_STATE (mpd);
  mpm->skip_header = mpm->skip_body = 0;
  return 0;
}

/* Extract the User from the URL or the ticket.  */
static int
pop_get_user (mu_authority_t auth)
{
  mu_folder_t folder = mu_authority_get_owner (auth);
  mu_mailbox_t mbox = folder->data;
  pop_data_t mpd = mbox->data;
  mu_ticket_t ticket = NULL;
  int status;
  /*  Fetch the user from them.  */

  mu_authority_get_ticket (auth, &ticket);
  if (mpd->user)
    {
      free (mpd->user);
      mpd->user = NULL;
    }
  /* Was it in the URL? */
  status = mu_url_aget_user (mbox->url, &mpd->user);
  if (status == MU_ERR_NOENT)
    status = mu_ticket_get_cred (ticket, mbox->url, "Pop User: ",
				 &mpd->user, NULL);
  if (status == MU_ERR_NOENT || mpd->user == NULL)
    return MU_ERR_NOUSERNAME;  
  return status;
}

/* Extract the User from the URL or the ticket.  */
static int
pop_get_passwd (mu_authority_t auth)
{
  mu_folder_t folder = mu_authority_get_owner (auth);
  mu_mailbox_t mbox = folder->data;
  pop_data_t mpd = mbox->data;
  mu_ticket_t ticket = NULL;
  int status;

  mu_authority_get_ticket (auth, &ticket);
  /* Was it in the URL? */
  status = mu_url_get_secret (mbox->url, &mpd->secret);
  if (status == MU_ERR_NOENT)
    status = mu_ticket_get_cred (ticket, mbox->url, "Pop Passwd: ",
				 NULL, &mpd->secret);
  if (status == MU_ERR_NOENT || !mpd->secret)
    /* FIXME: Is this always right? The user might legitimately have
       no password */
    return MU_ERR_NOPASSWORD;
  return 0;
}


static char *
pop_get_timestamp (pop_data_t mpd)
{
  char *right, *left;
  char *timestamp = NULL;
  size_t len;

  len = strlen (mpd->greeting_banner);
  right = memchr (mpd->greeting_banner, '<', len);
  if (right)
    {
      len = len - (right - mpd->greeting_banner);
      left = memchr (right, '>', len);
      if (left)
	{
	  len = left - right + 1;
	  timestamp = calloc (len + 1, 1);
	  if (timestamp != NULL)
	    {
	      memcpy (timestamp, right, len);
	    }
	}
    }
  return timestamp;
}

/*  Make the MD5 string.  */
static int
pop_get_md5 (pop_data_t mpd)
{
  struct mu_md5_ctx md5context;
  unsigned char md5digest[16];
  char digest[64]; /* Really it just has to be 32 + 1(null).  */
  char *tmp;
  size_t n;
  char *timestamp;

  timestamp = pop_get_timestamp (mpd);
  if (timestamp == NULL)
    return EINVAL;

  mu_md5_init_ctx (&md5context);
  mu_md5_process_bytes (timestamp, strlen (timestamp), &md5context);
  mu_md5_process_bytes (mu_secret_password (mpd->secret),
			mu_secret_length (mpd->secret),
			&md5context);
  mu_secret_password_unref (mpd->secret);
  mu_secret_unref (mpd->secret);
  mpd->secret = NULL;
  mu_md5_finish_ctx (&md5context, md5digest);
  
  for (tmp = digest, n = 0; n < 16; n++, tmp += 2)
    sprintf (tmp, "%02x", md5digest[n]);
  *tmp = '\0';
  free (timestamp);
  mpd->digest = strdup (digest);
  return 0;
}

/* GRRRRR!!  We can not use sleep in the library since this we'll
   muck up any alarm() done by the user.  */
static int
pop_sleep (int seconds)
{
  struct timeval tval;
  tval.tv_sec = seconds;
  tval.tv_usec = 0;
  return select (1, NULL, NULL, NULL, &tval);
}

/* C99 says that a conforming implementation of snprintf () should return the
   number of char that would have been call but many old GNU/Linux && BSD
   implementations return -1 on error.  Worse QnX/Neutrino actually does not
   put the terminal null char.  So let's try to cope.  */
static int
pop_writeline (pop_data_t mpd, const char *format, ...)
{
  int len;
  va_list ap;
  int done = 1;

  if (mpd->buffer == NULL)
    return EINVAL;
  va_start(ap, format);
  do
    {
      len = vsnprintf (mpd->buffer, mpd->buflen - 1, format, ap);
      if (len < 0 || len >= (int)mpd->buflen
	  || !memchr (mpd->buffer, '\0', len + 1))
	{
	  mpd->buflen *= 2;
	  mpd->buffer = realloc (mpd->buffer, mpd->buflen);
	  if (mpd->buffer == NULL)
	    return ENOMEM;
	  done = 0;
	}
      else
	done = 1;
    }
  while (!done);
  va_end(ap);
  mpd->ptr = mpd->buffer + len;
  return 0;
}

/* A socket may write less then expected and we have to cope with nonblocking.
   if the write failed we keep track and restart where left.  */
static int
pop_write (pop_data_t mpd)
{
  int status = 0;
  if (mpd->ptr > mpd->buffer)
    {
      size_t len;
      size_t n = 0;
      len = mpd->ptr - mpd->buffer;
      status = mu_stream_write (mpd->mbox->stream, mpd->buffer, len, 0, &n);
      if (status == 0)
	{
	  memmove (mpd->buffer, mpd->buffer + n, len - n);
	  mpd->ptr -= n;
	}
    }
  else
    mpd->ptr = mpd->buffer;
  return status;
}

/* Call readline and reset the mpd->ptr to the buffer, signalling that we have
   done the read to completion. */
static int
pop_read_ack (pop_data_t mpd)
{
  int status = pop_readline (mpd);
  if (status == 0)
    mpd->ptr = mpd->buffer;
  return status;
}

/* Read a complete line form the pop server. Transform CRLF to LF, remove
   the stuff byte termination octet ".", put a null in the buffer
   when done.  */
static int
pop_readline (pop_data_t mpd)
{
  size_t n = 0;
  size_t total = mpd->ptr - mpd->buffer;
  int status;

  /* Must get a full line before bailing out.  */
  do
    {
      status = mu_stream_readline (mpd->mbox->stream, mpd->buffer + total,
				mpd->buflen - total,  mpd->offset, &n);
      if (status != 0)
	return status;

      /* The server went away:  It maybe a timeout and some pop server
	 does not send the -ERR.  Consider this like an error.  */
      if (n == 0)
	return EIO;

      total += n;
      mpd->offset += n;
      mpd->nl = memchr (mpd->buffer, '\n', total);
      if (mpd->nl == NULL)  /* Do we have a full line.  */
	{
	  /* Allocate a bigger buffer ?  */
	  if (total >= mpd->buflen -1)
	    {
	      mpd->buflen *= 2;
	      mpd->buffer = realloc (mpd->buffer, mpd->buflen + 1);
	      if (mpd->buffer == NULL)
		return ENOMEM;
	    }
	}
      mpd->ptr = mpd->buffer + total;
    }
  while (mpd->nl == NULL);

  /* When examining a multi-line response, the client checks to see if the
     line begins with the termination octet "."(DOT). If yes and if octets
     other than CRLF follow, the first octet of the line (the termination
     octet) is stripped away.  */
  if (total >= 3  && mpd->buffer[0] == '.')
    {
      if (mpd->buffer[1] != '\r' && mpd->buffer[2] != '\n')
	{
	  memmove (mpd->buffer, mpd->buffer + 1, total - 1);
	  mpd->ptr--;
	  mpd->nl--;
	}
      /* And if CRLF immediately follows the termination character, then the
	 response from the POP server is ended and the line containing
	 ".CRLF" is not considered part of the multi-line response.  */
      else if (mpd->buffer[1] == '\r' && mpd->buffer[2] == '\n')
	{
	  mpd->buffer[0] = '\0';
	  mpd->ptr = mpd->buffer;
	  mpd->nl = NULL;
	}
    }
  /* \r\n --> \n\0, conversion.  */
  if (mpd->nl > mpd->buffer)
    {
      *(mpd->nl - 1) = '\n';
      *(mpd->nl) = '\0';
      mpd->ptr = mpd->nl;
    }
  return 0;
}

#endif
