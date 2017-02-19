/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2006, 2007, 2009, 2010
   Free Software Foundation, Inc.

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

/** @file smtp.c
@brief an SMTP mailer
*/

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#ifdef ENABLE_SMTP

#include <errno.h>
#include <netdb.h>
#include <pwd.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mailutils/address.h>
#include <mailutils/debug.h>
#include <mailutils/errno.h>
#include <mailutils/header.h>
#include <mailutils/body.h>
#include <mailutils/message.h>
#include <mailutils/mime.h>
#include <mailutils/mutil.h>
#include <mailutils/observer.h>
#include <mailutils/property.h>
#include <mailutils/stream.h>
#include <mailutils/url.h>
#include <mailutils/tls.h>
#include <mailutils/md5.h>
#include <mailutils/io.h>
#include <mailutils/secret.h>
#include <mailutils/cctype.h>
#include <mailutils/cstr.h>

#include <mailer0.h>
#include <url0.h>
#include <registrar0.h>

static int      _mailer_smtp_init (mu_mailer_t);

static int
_url_smtp_init (mu_url_t url)
{
  /* host isn't optional */
  if (!url->host)
    return EINVAL;

  /* accept url->user, pass, and auth
     for the ESMTP authentication */

  /* all other fields must be NULL */
  if (url->path || url->qargc)
    return EINVAL;

  if (url->port == 0)
    url->port = MU_SMTP_PORT;

  return 0;
}

static struct _mu_record _smtp_record = {
  MU_SMTP_PRIO,
  MU_SMTP_SCHEME,
  _url_smtp_init,		/* url init.  */
  _mu_mailer_mailbox_init,	/* Mailbox init.  */
  _mailer_smtp_init,		/* Mailer init.  */
  _mu_mailer_folder_init,	/* Folder init.  */
  NULL,				/* No need for a back pointer.  */
  NULL,				/* _is_scheme method.  */
  NULL,				/* _get_url method.  */
  NULL,				/* _get_mailbox method.  */
  NULL,				/* _get_mailer method.  */
  NULL				/* _get_folder method.  */
};

/* We export : url parsing and the initialisation of
   the mailbox, via the register entry/record.  */
mu_record_t     mu_smtp_record = &_smtp_record;

struct _smtp
{
  mu_mailer_t     mailer;
  char           *mailhost;
  char           *localhost;

  /* IO buffering. */
  char           *buffer;	/* Must be freed. */
  size_t          buflen;

  char           *ptr;
  char           *nl;
  off_t           s_offset;

  enum smtp_state
  {
    SMTP_NO_STATE, SMTP_OPEN, SMTP_GREETINGS, SMTP_EHLO, SMTP_EHLO_ACK,
    SMTP_HELO, SMTP_HELO_ACK, SMTP_QUIT, SMTP_QUIT_ACK, SMTP_ENV_FROM,
    SMTP_ENV_RCPT, SMTP_MAIL_FROM, SMTP_MAIL_FROM_ACK, SMTP_RCPT_TO,
    SMTP_RCPT_TO_ACK, SMTP_DATA, SMTP_DATA_ACK, SMTP_SEND, SMTP_SEND_ACK,
    SMTP_SEND_DOT, SMTP_STARTTLS, SMTP_STARTTLS_ACK, SMTP_AUTH, SMTP_AUTH_ACK,
  }
  state;

  int             extended;
  unsigned long   capa;		/* Server capabilities */
  size_t          max_size;	/* Maximum message size the server is willing
				   to accept */
  unsigned long   auth_mechs;	/* Available ESMTP AUTH mechanisms */

  const char     *mail_from;
  mu_address_t    rcpt_to;	/* Destroy this if not the same as argto below. */
  mu_address_t    rcpt_bcc;
  size_t          rcpt_to_count;
  size_t          rcpt_bcc_count;
  size_t          rcpt_index;
  size_t          rcpt_count;
  int             bccing;
  mu_message_t    msg;		/* Destroy this if not same argmsg. */

  off_t           offset;

  /* The mu_mailer_send_message() args. */
  mu_message_t    argmsg;
  mu_address_t    argfrom;
  mu_address_t    argto;
};

typedef struct _smtp *smtp_t;

/* ESMTP capabilities */
#define CAPA_STARTTLS        0x00000001
#define CAPA_8BITMIME        0x00000002
#define CAPA_SIZE            0x00000004
#define CAPA_AUTH            0x00000008

/* ESMTP AUTH mechanisms */
#define AUTH_LOGIN           0x00000001
#define AUTH_PLAIN           0x00000002
#define AUTH_CRAM_MD5        0x00000004
#define AUTH_DIGEST_MD5      0x00000008
#define AUTH_GSSAPI          0x00000010
#define AUTH_EXTERNAL        0x00000020

struct auth_mech_record
{
  unsigned long   id;
  char           *name;
};

static struct auth_mech_record auth_mech_list[] = {
  {AUTH_LOGIN, "login"},
  {AUTH_PLAIN, "plain"},
  {AUTH_CRAM_MD5, "cram-md5"},
  {AUTH_DIGEST_MD5, "digest-md5"},
  {AUTH_GSSAPI, "gssapi"},
  {AUTH_EXTERNAL, "external"},
  {0, NULL},
};

static void     smtp_destroy (mu_mailer_t);
static int      smtp_open (mu_mailer_t, int);
static int      smtp_close (mu_mailer_t);
static int      smtp_send_message (mu_mailer_t, mu_message_t, mu_address_t,
				   mu_address_t);
static int      smtp_writeline (smtp_t smtp, const char *format, ...);
static int      smtp_readline (smtp_t);
static int      smtp_read_ack (smtp_t);
static int      smtp_parse_ehlo_ack (smtp_t);
static int      smtp_write (smtp_t);
static int      smtp_starttls (smtp_t);
static int      smtp_auth (smtp_t);

static int      _smtp_set_rcpt (smtp_t, mu_message_t, mu_address_t);

/* Useful little macros, since these are very repetitive. */

static void
CLEAR_STATE (smtp_t smtp)
{
  smtp->ptr = smtp->buffer;
  smtp->nl = NULL;
  smtp->s_offset = 0;

  smtp->state = SMTP_NO_STATE;

  smtp->extended = 0;

  if (smtp->mail_from)
    smtp->mail_from = NULL;

  if (smtp->rcpt_to != smtp->argto)
    mu_address_destroy (&smtp->rcpt_to);

  smtp->rcpt_to = NULL;

  mu_address_destroy (&smtp->rcpt_bcc);

  smtp->rcpt_to_count = 0;
  smtp->rcpt_bcc_count = 0;
  smtp->rcpt_index = 0;
  smtp->rcpt_count = 0;
  smtp->bccing = 0;

  if (smtp->msg != smtp->argmsg)
    mu_message_destroy (&smtp->msg, NULL);

  smtp->msg = NULL;

  smtp->offset = 0;

  smtp->argmsg = NULL;
  smtp->argfrom = NULL;
  smtp->argto = NULL;
}

/* If we are resuming, we should be resuming the SAME operation
   as that which is ongoing. Check this. */
static int
smtp_check_send_resumption (smtp_t smtp,
			    mu_message_t msg, mu_address_t from,
			    mu_address_t to)
{
  if (smtp->state == SMTP_NO_STATE)
    return 0;

  /* FIXME: state should be one of the "send" states if its not
     "no state" */
  if (msg != smtp->argmsg)
    return MU_ERR_BAD_RESUMPTION;

  if (from != smtp->argfrom)
    return MU_ERR_BAD_RESUMPTION;

  if (to != smtp->argto)
    return MU_ERR_BAD_RESUMPTION;

  return 0;
}

#define CHECK_SEND_RESUME(smtp, msg, from, to) \
do { \
  if((status = smtp_check_send_resumption(smtp, msg, from, to)) != 0) \
    return status; \
} while (0)

/* Clear the state and close the stream. */
#define CHECK_ERROR_CLOSE(mailer, smtp, status) \
do \
  { \
     if (status != 0) \
       { \
          mu_stream_close (mailer->stream); \
          CLEAR_STATE (smtp); \
          return status; \
       } \
  } \
while (0)

/* Clear the state. */
#define CHECK_ERROR(smtp, status) \
do \
  { \
     if (status != 0) \
       { \
          CLEAR_STATE (smtp); \
          return status; \
       } \
  } \
while (0)

/* Clear the state for non recoverable error.  */
#define CHECK_EAGAIN(smtp, status) \
do \
  { \
    if (status != 0) \
      { \
         if (status != EAGAIN && status != EINPROGRESS && status != EINTR) \
           { \
             CLEAR_STATE (smtp); \
           } \
         return status; \
      } \
   }  \
while (0)

static int
_mailer_smtp_init (mu_mailer_t mailer)
{
  smtp_t          smtp;

  /* Allocate memory specific to smtp mailer.  */
  smtp = mailer->data = calloc (1, sizeof (*smtp));
  if (mailer->data == NULL)
    return ENOMEM;

  smtp->mailer = mailer;	/* Back pointer.  */
  smtp->state = SMTP_NO_STATE;

  mailer->_destroy = smtp_destroy;
  mailer->_open = smtp_open;
  mailer->_close = smtp_close;
  mailer->_send_message = smtp_send_message;

  /* Set our properties.  */
  {
    mu_property_t   property = NULL;

    mu_mailer_get_property (mailer, &property);
    mu_property_set_value (property, "TYPE", "SMTP", 1);
  }

  return 0;
}

static void
smtp_destroy (mu_mailer_t mailer)
{
  smtp_t          smtp = mailer->data;

  CLEAR_STATE (smtp);

  /* Not our responsability to close.  */

  if (smtp->mailhost)
    free (smtp->mailhost);
  if (smtp->localhost)
    free (smtp->localhost);
  if (smtp->buffer)
    free (smtp->buffer);

  free (smtp);

  mailer->data = NULL;
}

/** Open an SMTP mailer.
An SMTP mailer must be opened before any messages can be sent.
@param mailer the mailer created by smtp_create()
@param flags the mailer flags
*/
static int
smtp_open (mu_mailer_t mailer, int flags)
{
  smtp_t          smtp = mailer->data;
  int             status;
  long            port;

  /* Sanity checks.  */
  if (!smtp)
    return EINVAL;

  mailer->flags = flags;

  if ((status = mu_url_get_port (mailer->url, &port)) != 0)
    return status;

  switch (smtp->state)
    {
    case SMTP_NO_STATE:
      if (smtp->mailhost)
	{
	  free (smtp->mailhost);
	  smtp->mailhost = NULL;
	}

      /* Fetch the mailer server name and the port in the mu_url_t.  */
      if ((status = mu_url_aget_host (mailer->url, &smtp->mailhost)) != 0)
	return status;

      if (smtp->localhost)
	{
	  free (smtp->localhost);
	  smtp->localhost = NULL;
	}
      /* Fetch our local host name.  */

      status = mu_get_host_name (&smtp->localhost);

      if (status != 0)
	{
	  /* gethostname failed, abort.  */
	  free (smtp->mailhost);
	  smtp->mailhost = NULL;
	  return status;
	}

      /* allocate a working io buffer.  */
      if (smtp->buffer == NULL)
	{
	  smtp->buflen = 512;	/* Initial guess.  */
	  smtp->buffer = malloc (smtp->buflen + 1);
	  if (smtp->buffer == NULL)
	    {
	      CHECK_ERROR (smtp, ENOMEM);
	    }
	  smtp->ptr = smtp->buffer;
	}

      /* Create a TCP stack if one is not given.  */
      if (mailer->stream == NULL)
	{
	  status =
	    mu_tcp_stream_create (&mailer->stream, smtp->mailhost, port,
				  mailer->flags);
	  CHECK_ERROR (smtp, status);
	  mu_stream_setbufsiz (mailer->stream, BUFSIZ);
	}
      CHECK_ERROR (smtp, status);
      smtp->state = SMTP_OPEN;

    case SMTP_OPEN:
      MU_DEBUG2 (mailer->debug, MU_DEBUG_PROT,
		 "smtp_open (host: %s port: %ld)\n", smtp->mailhost, port);
      status = mu_stream_open (mailer->stream);
      CHECK_EAGAIN (smtp, status);
      smtp->state = SMTP_GREETINGS;

    case SMTP_GREETINGS:
      /* Swallow the greetings.  */
      status = smtp_read_ack (smtp);
      CHECK_EAGAIN (smtp, status);

      if (smtp->buffer[0] != '2')
	{
	  mu_stream_close (mailer->stream);
	  return EACCES;
	}

    ehlo:
      status = smtp_writeline (smtp, "EHLO %s\r\n", smtp->localhost);
      CHECK_ERROR (smtp, status);

      smtp->state = SMTP_EHLO;

    case SMTP_EHLO:
      /* We first try Extended SMTP.  */
      status = smtp_write (smtp);
      CHECK_EAGAIN (smtp, status);
      smtp->state = SMTP_EHLO_ACK;

    case SMTP_EHLO_ACK:
      status = smtp_parse_ehlo_ack (smtp);
      CHECK_EAGAIN (smtp, status);

      if (smtp->buffer[0] != '2')
	{
	  smtp->extended = 0;
	  status = smtp_writeline (smtp, "HELO %s\r\n", smtp->localhost);
	  CHECK_ERROR (smtp, status);
	  smtp->state = SMTP_HELO;
	}
      else
	{
	  smtp->extended = 1;

	  if (smtp->capa & CAPA_STARTTLS)
	    smtp->state = SMTP_STARTTLS;
	  else if (smtp->capa & CAPA_AUTH && mailer->url->user)
	    {
	      smtp->state = SMTP_AUTH;
	    }
	  else
	    break;
	}

    case SMTP_STARTTLS:
    case SMTP_STARTTLS_ACK:
      if ((smtp->capa & CAPA_STARTTLS) && smtp_starttls (smtp) == 0)
	goto ehlo;
      
    case SMTP_AUTH:
    case SMTP_AUTH_ACK:
      if (smtp->capa & CAPA_AUTH)
	{
	  smtp_auth (smtp);
	  break;
	}

    case SMTP_HELO:
      if (!smtp->extended)	/* FIXME: this will always be false! */
	{
	  status = smtp_write (smtp);
	  CHECK_EAGAIN (smtp, status);
	}
      smtp->state = SMTP_HELO_ACK;

    case SMTP_HELO_ACK:
      if (!smtp->extended)
	{
	  status = smtp_read_ack (smtp);
	  CHECK_EAGAIN (smtp, status);

	  if (smtp->buffer[0] != '2')
	    {
	      mu_stream_close (mailer->stream);
	      CLEAR_STATE (smtp);
	      return EACCES;
	    }
	}

    default:
      break;
    }

  CLEAR_STATE (smtp);

  return 0;
}

static int
smtp_close (mu_mailer_t mailer)
{
  smtp_t          smtp = mailer->data;
  int             status;

  switch (smtp->state)
    {
    case SMTP_NO_STATE:
      status = smtp_writeline (smtp, "QUIT\r\n");
      CHECK_ERROR (smtp, status);

      smtp->state = SMTP_QUIT;

    case SMTP_QUIT:
      status = smtp_write (smtp);
      CHECK_EAGAIN (smtp, status);
      smtp->state = SMTP_QUIT_ACK;

    case SMTP_QUIT_ACK:
      status = smtp_read_ack (smtp);
      CHECK_EAGAIN (smtp, status);

    default:
      break;
    }
  smtp->state = SMTP_NO_STATE;
  return mu_stream_close (mailer->stream);
}

#ifdef WITH_TLS
/*
  Client side STARTTLS support.
 */

static int
smtp_reader (void *iodata)
{
  int             status = 0;
  smtp_t          iop = iodata;

  status = smtp_read_ack (iop);
  CHECK_EAGAIN (iop, status);
  return status;
}

static int
smtp_writer (void *iodata, char *buf)
{
  smtp_t          iop = iodata;
  int             status;

  if (mu_c_strncasecmp (buf, "EHLO", 4) == 0)
    status = smtp_writeline (iop, "%s %s\r\n", buf, iop->localhost);
  else
    status = smtp_writeline (iop, "%s\r\n", buf);
  CHECK_ERROR (iop, status);
  status = smtp_write (iop);
  CHECK_EAGAIN (iop, status);
  return status;
}

static void
smtp_stream_ctl (void *iodata, mu_stream_t * pold, mu_stream_t new)
{
  smtp_t          iop = iodata;

  if (pold)
    *pold = iop->mailer->stream;
  if (new)
    iop->mailer->stream = new;
}
#endif

static int
smtp_starttls (smtp_t smtp)
{
#ifdef WITH_TLS
  int             status;
  mu_mailer_t     mailer = smtp->mailer;
  char           *keywords[] = { "STARTTLS", NULL };

  if (!mu_tls_enable || !(smtp->capa & CAPA_STARTTLS))
    return -1;

  smtp->capa = 0;
  smtp->auth_mechs = 0;
  status = mu_tls_begin (smtp, smtp_reader, smtp_writer,
			 smtp_stream_ctl, keywords);

  MU_DEBUG1 (mailer->debug, MU_DEBUG_PROT, "TLS negotiation %s\n",
	     status == 0 ? "succeeded" : "failed");

  return status;
#else
  return -1;
#endif /* WITH_TLS */
}

static void
cram_md5 (char *secret, unsigned char *challenge, size_t challenge_len, 
          unsigned char *digest)
{
  struct mu_md5_ctx context;
  unsigned char   ipad[64];
  unsigned char   opad[64];
  int             secret_len;
  int             i;

  if (secret == 0 || challenge == 0)
    return;

  secret_len = strlen (secret);
  memset (ipad, 0, sizeof (ipad));
  memset (opad, 0, sizeof (opad));

  if (secret_len > 64)
    {
      mu_md5_init_ctx (&context);
      mu_md5_process_bytes ((unsigned char *) secret, secret_len, &context);
      mu_md5_finish_ctx (&context, ipad);
      mu_md5_finish_ctx (&context, opad);
    }
  else
    {
      memcpy (ipad, secret, secret_len);
      memcpy (opad, secret, secret_len);
    }

  for (i = 0; i < 64; i++)
    {
      ipad[i] ^= 0x36;
      opad[i] ^= 0x5c;
    }

  mu_md5_init_ctx (&context);
  mu_md5_process_bytes (ipad, sizeof (ipad), &context);
  mu_md5_process_bytes (challenge, challenge_len, &context);
  mu_md5_finish_ctx (&context, digest);

  mu_md5_init_ctx (&context);
  mu_md5_process_bytes (opad, sizeof (opad), &context);
  mu_md5_process_bytes (digest, 16, &context);
  mu_md5_finish_ctx (&context, digest);
}

static int
smtp_auth (smtp_t smtp)
{
  int             status;
  mu_mailer_t     mailer = smtp->mailer;
  struct auth_mech_record *mechs = auth_mech_list;
  const char     *chosen_mech_name = NULL;
  int             chosen_mech_id = 0;

  status = mu_url_sget_auth (mailer->url, &chosen_mech_name);
  if (status != MU_ERR_NOENT)
    {
      for (; mechs->name; mechs++)
	{
	  if (!mu_c_strcasecmp (mechs->name, chosen_mech_name))
	    {
	      chosen_mech_id = mechs->id;
	      break;
	    }
	}
    }
  if (chosen_mech_id)
    {
      if (smtp->auth_mechs & chosen_mech_id)
	{
	  smtp->auth_mechs = 0;
	  smtp->auth_mechs |= chosen_mech_id;
	}
      else
	{
	  MU_DEBUG1 (mailer->debug, MU_DEBUG_ERROR,
		     "mailer does not support AUTH '%s' mechanism\n",
		     chosen_mech_name);
	  return -1;
	}
    }

#if 0 && defined(WITH_GSASL)

  /* FIXME: Add GNU SASL support. */

#else

  /* Provide basic AUTH mechanisms when GSASL is not enabled. */

  if (smtp->auth_mechs & AUTH_CRAM_MD5)
    {
      int             i;
      char           *p, *buf = NULL;
      const char     *user = NULL;
      mu_secret_t     secret;
      unsigned char  *chl;
      size_t          chlen, buflen = 0, b64buflen = 0;
      unsigned char  *b64buf = NULL;
      unsigned char   digest[16];
      static char     ascii_digest[33];

      memset (digest, 0, 16);

      status = mu_url_sget_user (mailer->url, &user);
      if (status == MU_ERR_NOENT)
	return -1;

      status = mu_url_get_secret (mailer->url, &secret);
      if (status == MU_ERR_NOENT)
	{
	  MU_DEBUG (mailer->debug, MU_DEBUG_ERROR,
		    "AUTH CRAM-MD5 mechanism requires giving a password\n");
	  return -1;
	}

      status = smtp_writeline (smtp, "AUTH CRAM-MD5\r\n");
      CHECK_ERROR (smtp, status);
      status = smtp_write (smtp);
      CHECK_EAGAIN (smtp, status);
      status = smtp_read_ack (smtp);
      CHECK_EAGAIN (smtp, status);

      if (strncmp (smtp->buffer, "334 ", 4))
	{
	  MU_DEBUG (mailer->debug, MU_DEBUG_ERROR,
		    "mailer rejected the AUTH CRAM-MD5 command\n");
	  return -1;
	}

      p = strchr (smtp->buffer, ' ') + 1;
      mu_rtrim_cset (p, "\r\n");
      mu_base64_decode ((unsigned char*) p, strlen (p), &chl, &chlen);

      cram_md5 ((char *) mu_secret_password (secret), chl, chlen, digest);
      mu_secret_password_unref (secret);
      free (chl);

      for (i = 0; i < 16; i++)
	sprintf (ascii_digest + 2 * i, "%02x", digest[i]);

      mu_asnprintf (&buf, &buflen, "%s %s", user, ascii_digest);
      buflen = strlen (buf);
      mu_base64_encode ((unsigned char*) buf, buflen, &b64buf, &b64buflen);
      free (buf);

      status = smtp_writeline (smtp, "%s\r\n", b64buf);
      CHECK_ERROR (smtp, status);
      status = smtp_write (smtp);
      CHECK_EAGAIN (smtp, status);
      status = smtp_read_ack (smtp);
      CHECK_EAGAIN (smtp, status);
    }

  else if (smtp->auth_mechs & AUTH_PLAIN)
    {
      int             c;
      char           *buf = NULL;
      unsigned char  *b64buf = NULL;
      size_t          buflen = 0, b64buflen = 0;
      const char     *user = NULL;
      mu_secret_t     secret;

      status = mu_url_sget_user (mailer->url, &user);
      if (status == MU_ERR_NOENT)
	return -1;

      status = mu_url_get_secret (mailer->url, &secret);
      if (status == MU_ERR_NOENT)
	{
	  MU_DEBUG (mailer->debug, MU_DEBUG_ERROR,
		    "AUTH PLAIN mechanism requires giving a password\n");
	  return -1;
	}

      mu_asnprintf (&buf, &buflen, "^%s^%s",
		    user, mu_secret_password (secret));
      mu_secret_password_unref (secret);
      buflen = strlen (buf);
      for (c = buflen - 1; c >= 0; c--)
	{
	  if (buf[c] == '^')
	    buf[c] = '\0';
	}
      mu_base64_encode ((unsigned char*) buf, buflen, &b64buf, &b64buflen);
      free (buf);

      status = smtp_writeline (smtp, "AUTH PLAIN %s\r\n", b64buf);
      CHECK_ERROR (smtp, status);
      status = smtp_write (smtp);
      CHECK_EAGAIN (smtp, status);
      status = smtp_read_ack (smtp);
      CHECK_EAGAIN (smtp, status);
    }

#endif /* not WITH_GSASL */
  return 0;
}

static int
message_set_header_value (mu_message_t msg, const char *field,
			  const char *value)
{
  int             status = 0;
  mu_header_t     hdr = NULL;

  if ((status = mu_message_get_header (msg, &hdr)))
    return status;

  if ((status = mu_header_set_value (hdr, field, value, 1)))
    return status;

  return status;
}

static int
message_has_bcc (mu_message_t msg)
{
  int             status;
  mu_header_t     header = NULL;
  size_t          bccsz = 0;

  if ((status = mu_message_get_header (msg, &header)))
    return status;

  status = mu_header_get_value (header, MU_HEADER_BCC, NULL, 0, &bccsz);

  /* MU_ERR_NOENT, or there was a Bcc: field. */
  return status == MU_ERR_NOENT ? 0 : 1;
}

/*

The smtp mailer doesn't deal with mail like:

To: public@com, pub2@com
Bcc: hidden@there, two@there

It just sends the message to all the addresses, making the
"blind" cc not particularly blind.

The correct algorithm is

- open smtp connection
- look as msg, figure out addrto&cc, and addrbcc
- deliver to the to & cc addresses:
  - if there are bcc addrs, remove the bcc field
  - send the message to to & cc addrs:
  mail from: me
  rcpt to: public@com
  rcpt to: pub2@com
  data
  ...

- deliver to the bcc addrs:

  for a in (bccaddrs)
  do
    - add header field to msg,  bcc: $a
    - send the msg:
    mail from: me
    rcpt to: $a
    data
    ...
  done

- quit smtp connection

*/

static int
smtp_send_message (mu_mailer_t mailer, mu_message_t argmsg,
		   mu_address_t argfrom, mu_address_t argto)
{
  smtp_t          smtp = NULL;
  int             status;

  if (mailer == NULL)
    return EINVAL;

  smtp = mailer->data;
  if (!smtp)
    return EINVAL;

  CHECK_SEND_RESUME (smtp, argmsg, argfrom, argto);

  switch (smtp->state)
    {
    case SMTP_NO_STATE:
      if (argmsg == NULL || argfrom == NULL)
	return EINVAL;

      smtp->argmsg = smtp->msg = argmsg;
      smtp->argfrom = argfrom;
      smtp->argto = argto;

      status = mu_address_sget_email (smtp->argfrom, 1, &smtp->mail_from);
      CHECK_ERROR (smtp, status);

      status = _smtp_set_rcpt (smtp, smtp->argmsg, smtp->argto);
      CHECK_ERROR (smtp, status);

      /* Clear the Bcc: field if we found one. */
      if (message_has_bcc (smtp->argmsg))
	{
	  smtp->msg = NULL;
	  status = mu_message_create_copy (&smtp->msg, smtp->argmsg);
	  CHECK_ERROR (smtp, status);

	  status = message_set_header_value (smtp->msg, MU_HEADER_BCC, NULL);
	  CHECK_ERROR (smtp, status);
	}

      /* Begin bccing if there are not To: recipients. */
      if (smtp->rcpt_to_count == 0)
	smtp->bccing = 1;

      smtp->rcpt_index = 1;

      smtp->state = SMTP_ENV_FROM;

    case SMTP_ENV_FROM:
    ENV_FROM:
      {
	size_t          size;

	if ((smtp->capa & CAPA_SIZE)
	    && mu_message_size (smtp->msg, &size) == 0)
	  status = smtp_writeline (smtp, "MAIL FROM:<%s> SIZE=%lu\r\n",
				   smtp->mail_from, size);
	else
	  status = smtp_writeline (smtp, "MAIL FROM:<%s>\r\n",
				   smtp->mail_from);
	CHECK_ERROR (smtp, status);
	smtp->state = SMTP_MAIL_FROM;
      }

      /* We use a goto, since we may have multiple messages,
         we come back here and doit all over again ... Not pretty.  */
    case SMTP_MAIL_FROM:
      status = smtp_write (smtp);
      CHECK_EAGAIN (smtp, status);
      smtp->state = SMTP_MAIL_FROM_ACK;

    case SMTP_MAIL_FROM_ACK:
      status = smtp_read_ack (smtp);
      CHECK_EAGAIN (smtp, status);
      if (smtp->buffer[0] != '2')
	{
	  mu_stream_close (mailer->stream);
	  CLEAR_STATE (smtp);
	  return EACCES;
	}

      /* We use a goto, since we may have multiple recipients,
         we come back here and do it all over again ... Not pretty. */
    case SMTP_ENV_RCPT:
    ENV_RCPT:
      {
	mu_address_t    addr = smtp->rcpt_to;
	const char     *to = NULL;

	if (smtp->bccing)
	  addr = smtp->rcpt_bcc;
	status = mu_address_sget_email (addr, smtp->rcpt_index, &to);

	CHECK_ERROR (smtp, status);

	/* Add the Bcc: field back in for recipient. */
	if (smtp->bccing)
	  {
	    status = message_set_header_value (smtp->msg, MU_HEADER_BCC, to);
	    CHECK_ERROR (smtp, status);
	  }

	status = smtp_writeline (smtp, "RCPT TO:<%s>\r\n", to);

	CHECK_ERROR (smtp, status);

	smtp->state = SMTP_RCPT_TO;
	smtp->rcpt_index++;
      }

    case SMTP_RCPT_TO:
      status = smtp_write (smtp);
      CHECK_EAGAIN (smtp, status);
      smtp->state = SMTP_RCPT_TO_ACK;

    case SMTP_RCPT_TO_ACK:
      status = smtp_read_ack (smtp);
      CHECK_EAGAIN (smtp, status);
      if (smtp->buffer[0] != '2')
	{
	  mu_stream_close (mailer->stream);
	  CLEAR_STATE (smtp);
	  return MU_ERR_SMTP_RCPT_FAILED;
	}
      /* Redo the receipt sequence for every To: and Cc: recipient. */
      if (!smtp->bccing && smtp->rcpt_index <= smtp->rcpt_to_count)
	goto ENV_RCPT;

      /* We are done with the rcpt. */
      status = smtp_writeline (smtp, "DATA\r\n");
      CHECK_ERROR (smtp, status);
      smtp->state = SMTP_DATA;

    case SMTP_DATA:
      status = smtp_write (smtp);
      CHECK_EAGAIN (smtp, status);
      smtp->state = SMTP_DATA_ACK;

    case SMTP_DATA_ACK:
      status = smtp_read_ack (smtp);
      CHECK_EAGAIN (smtp, status);
      if (smtp->buffer[0] != '3')
	{
	  mu_stream_close (mailer->stream);
	  CLEAR_STATE (smtp);
	  return EACCES;
	}
      smtp->offset = 0;
      smtp->state = SMTP_SEND;

      if ((smtp->mailer->flags & MAILER_FLAG_DEBUG_DATA) == 0)
	MU_DEBUG (smtp->mailer->debug, MU_DEBUG_PROT, "> (data...)\n");

    case SMTP_SEND:
      {
	mu_stream_t     stream;
	size_t          n = 0;
	char            data[256] = "";
	mu_header_t     hdr;
	mu_body_t       body;
	int             found_nl;

	/* We may be here after an EAGAIN so check if we have something
	   in the buffer and flush it.  */
	status = smtp_write (smtp);
	CHECK_EAGAIN (smtp, status);

	mu_message_get_header (smtp->msg, &hdr);
	mu_header_get_stream (hdr, &stream);
	while ((status = mu_stream_readline (stream, data, sizeof (data),
					     smtp->offset, &n)) == 0 && n > 0)
	  {
	    int             nl;

	    found_nl = (n == 1 && data[0] == '\n');
	    if ((nl = (data[n - 1] == '\n')))
	      data[n - 1] = '\0';
	    if (data[0] == '.')
	      {
		status = smtp_writeline (smtp, ".%s", data);
		CHECK_ERROR (smtp, status);
	      }
	    else if (mu_c_strncasecmp (data, MU_HEADER_FCC,
				       sizeof (MU_HEADER_FCC) - 1))
	      {
		status = smtp_writeline (smtp, "%s", data);
		CHECK_ERROR (smtp, status);
		status = smtp_write (smtp);
		CHECK_EAGAIN (smtp, status);
	      }
	    else
	      nl = 0;

	    if (nl)
	      {
		status = smtp_writeline (smtp, "\r\n");
		CHECK_ERROR (smtp, status);
		status = smtp_write (smtp);
		CHECK_EAGAIN (smtp, status);
	      }
	    smtp->offset += n;
	  }

	if (!found_nl)
	  {
	    status = smtp_writeline (smtp, "\r\n");
	    CHECK_ERROR (smtp, status);
	    status = smtp_write (smtp);
	    CHECK_EAGAIN (smtp, status);
	  }

	mu_message_get_body (smtp->msg, &body);
	mu_body_get_stream (body, &stream);
	smtp->offset = 0;
	while ((status = mu_stream_readline (stream, data, sizeof (data) - 1,
					     smtp->offset, &n)) == 0 && n > 0)
	  {
	    if (data[n - 1] == '\n')
	      data[n - 1] = '\0';
	    if (data[0] == '.')
	      status = smtp_writeline (smtp, ".%s\r\n", data);
	    else
	      status = smtp_writeline (smtp, "%s\r\n", data);
	    CHECK_ERROR (smtp, status);
	    status = smtp_write (smtp);
	    CHECK_EAGAIN (smtp, status);
	    smtp->offset += n;
	  }

	smtp->offset = 0;
	status = smtp_writeline (smtp, ".\r\n");
	CHECK_ERROR (smtp, status);
	smtp->state = SMTP_SEND_DOT;
      }

    case SMTP_SEND_DOT:
      status = smtp_write (smtp);
      CHECK_EAGAIN (smtp, status);
      smtp->state = SMTP_SEND_ACK;

    case SMTP_SEND_ACK:
      status = smtp_read_ack (smtp);
      CHECK_EAGAIN (smtp, status);
      if (smtp->buffer[0] != '2')
	{
	  mu_stream_close (mailer->stream);
	  CLEAR_STATE (smtp);
	  return EACCES;
	}

      /* Decide whether we need to loop again, to deliver to Bcc:
         recipients. */
      if (!smtp->bccing)
	{
	  smtp->bccing = 1;
	  smtp->rcpt_index = 1;
	}
      if (smtp->rcpt_index <= smtp->rcpt_bcc_count)
	goto ENV_FROM;

      mu_observable_notify (mailer->observable, MU_EVT_MAILER_MESSAGE_SENT,
			    argmsg);

    default:
      break;
    }
  CLEAR_STATE (smtp);
  return 0;
}

int
smtp_address_add (mu_address_t * paddr, const char *value)
{
  mu_address_t    addr = NULL;
  int             status;

  status = mu_address_create (&addr, value);
  if (status)
    return status;
  status = mu_address_union (paddr, addr);
  mu_address_destroy (&addr);
  return status;
}

static int
_smtp_property_is_set (smtp_t smtp, const char *name)
{
  mu_property_t   property = NULL;

  mu_mailer_get_property (smtp->mailer, &property);
  return mu_property_is_set (property, name);
}

static int
_smtp_set_rcpt (smtp_t smtp, mu_message_t msg, mu_address_t to)
{
  int             status = 0;
  mu_header_t     header = NULL;
  char           *value;

  /* Get RCPT_TO from TO, or the message. */

  if (to)
    {
      /* Use the specified mu_address_t. */
      if ((status = mu_mailer_check_to (to)) != 0)
	{
	  MU_DEBUG (smtp->mailer->debug, MU_DEBUG_ERROR,
		    "mu_mailer_send_message(): explicit to not valid\n");
	  return status;
	}
      smtp->rcpt_to = to;
      mu_address_get_count (smtp->rcpt_to, &smtp->rcpt_to_count);

      if (status)
	return status;
    }

  if (!to || _smtp_property_is_set (smtp, "READ_RECIPIENTS"))
    {
      if ((status = mu_message_get_header (msg, &header)))
	return status;

      status = mu_header_aget_value (header, MU_HEADER_TO, &value);

      if (status == 0)
	{
	  smtp_address_add (&smtp->rcpt_to, value);
	  free (value);
	}
      else if (status != MU_ERR_NOENT)
	goto end;

      status = mu_header_aget_value (header, MU_HEADER_CC, &value);

      if (status == 0)
	{
	  smtp_address_add (&smtp->rcpt_to, value);
	  free (value);
	}
      else if (status != MU_ERR_NOENT)
	goto end;

      status = mu_header_aget_value (header, MU_HEADER_BCC, &value);
      if (status == 0)
	{
	  smtp_address_add (&smtp->rcpt_bcc, value);
	  free (value);
	}
      else if (status != MU_ERR_NOENT)
	goto end;

      /* If to or bcc is present, the must be OK. */
      if (smtp->rcpt_to && (status = mu_mailer_check_to (smtp->rcpt_to)))
	goto end;

      if (smtp->rcpt_bcc && (status = mu_mailer_check_to (smtp->rcpt_bcc)))
	goto end;
    }

end:

  if (status)
    {
      mu_address_destroy (&smtp->rcpt_to);
      mu_address_destroy (&smtp->rcpt_bcc);
    }
  else
    {
      if (smtp->rcpt_to)
	mu_address_get_count (smtp->rcpt_to, &smtp->rcpt_to_count);

      if (smtp->rcpt_bcc)
	mu_address_get_count (smtp->rcpt_bcc, &smtp->rcpt_bcc_count);

      if (smtp->rcpt_to_count + smtp->rcpt_bcc_count == 0)
	status = MU_ERR_MAILER_NO_RCPT_TO;
    }

  return status;
}

/* C99 says that a conforming implementations of snprintf ()
   should return the number of char that would have been call
   but many GNU/Linux && BSD implementations return -1 on error.
   Worse QNX/Neutrino actually does not put the terminal
   null char.  So let's try to cope.  */
static int
smtp_writeline (smtp_t smtp, const char *format, ...)
{
  int             len;
  va_list         ap;
  int             done = 1;

  va_start (ap, format);
  do
    {
      len = vsnprintf (smtp->buffer, smtp->buflen - 1, format, ap);
      if (len < 0 || (len >= (int) smtp->buflen)
	  || !memchr (smtp->buffer, '\0', len + 1))
	{
	  char           *buffer = NULL;
	  size_t          buflen = smtp->buflen * 2;

	  buffer = realloc (smtp->buffer, buflen);
	  if (smtp->buffer == NULL)
	    return ENOMEM;
	  smtp->buffer = buffer;
	  smtp->buflen = buflen;
	  done = 0;
	}
      else
	done = 1;
    }
  while (!done);

  va_end (ap);

  smtp->ptr = smtp->buffer + len;

  if ((smtp->state != SMTP_SEND && smtp->state != SMTP_SEND_DOT)
      || smtp->mailer->flags & MAILER_FLAG_DEBUG_DATA)
    {
      while (len > 0 && mu_isblank (smtp->buffer[len - 1]))
	len--;
      MU_DEBUG2 (smtp->mailer->debug, MU_DEBUG_PROT, "> %.*s\n", len,
		 smtp->buffer);
    }

  return 0;
}

static int
smtp_write (smtp_t smtp)
{
  int             status = 0;
  size_t          len;

  if (smtp->ptr > smtp->buffer)
    {
      len = smtp->ptr - smtp->buffer;
      status = mu_stream_write (smtp->mailer->stream, smtp->buffer, len,
				0, &len);
      if (status == 0)
	{
	  memmove (smtp->buffer, smtp->buffer + len, len);
	  smtp->ptr -= len;
	}
    }
  else
    {
      smtp->ptr = smtp->buffer;
      len = 0;
    }
  return status;
}

static int
smtp_read_ack (smtp_t smtp)
{
  int             status;
  int             multi;

  do
    {
      multi = 0;
      status = smtp_readline (smtp);
      if ((smtp->ptr - smtp->buffer) > 4 && smtp->buffer[3] == '-')
	multi = 1;
      if (status == 0)
	smtp->ptr = smtp->buffer;
    }
  while (multi && status == 0);

  if (status == 0)
    smtp->ptr = smtp->buffer;
  return status;
}

static int
smtp_parse_ehlo_ack (smtp_t smtp)
{
  int             status;
  int             multi;

  do
    {
      multi = 0;
      status = smtp_readline (smtp);
      if ((smtp->ptr - smtp->buffer) > 4 && smtp->buffer[3] == '-')
	multi = 1;
      if (status == 0 && memcmp (smtp->buffer, "250", 3) == 0)
	{
	  char *capa_str = smtp->buffer + 4;

	  smtp->ptr = smtp->buffer;

	  if (!mu_c_strncasecmp (capa_str, "STARTTLS", 8))
	    smtp->capa |= CAPA_STARTTLS;
	  else if (!mu_c_strncasecmp (capa_str, "SIZE", 4))
	    {
	      char  *p;
	      size_t n;
	      
	      smtp->capa |= CAPA_SIZE;

	      n = strtoul (capa_str + 5, &p, 10);
	      if (*p != '\n')
		MU_DEBUG1 (smtp->mailer->debug, MU_DEBUG_ERROR,
			   "suspicious size capability: %s",
			   smtp->buffer);
	      else
		smtp->max_size = n;
	    }
	  else if (!mu_c_strncasecmp (capa_str, "AUTH", 4))
	    {
	      char           *name, *s;

	      smtp->capa |= CAPA_AUTH;

	      for (name = strtok_r (capa_str + 5, " ", &s); name;
		   name = strtok_r (NULL, " ", &s))
		{
		  struct auth_mech_record *mechs = auth_mech_list;

		  mu_rtrim_cset (name, "\r\n");
		  for (; mechs->name; mechs++)
		    {
		      if (!mu_c_strcasecmp (mechs->name, name))
			{
			  smtp->auth_mechs |= mechs->id;
			  break;
			}
		    }
		}
	    }

	}
    }
  while (multi && status == 0);

  if (status == 0)
    smtp->ptr = smtp->buffer;
  return status;
}

/* Read a complete line form the pop server. Transform CRLF to LF,
   put a null in the buffer when done.  */
static int
smtp_readline (smtp_t smtp)
{
  size_t          n = 0;
  size_t          total = smtp->ptr - smtp->buffer;
  int             status;

  /* Must get a full line before bailing out.  */
  do
    {
      status = mu_stream_readline (smtp->mailer->stream, smtp->buffer + total,
				   smtp->buflen - total, smtp->s_offset, &n);
      if (status != 0)
	return status;

      /* Server went away, consider this like an error.  */
      if (n == 0)
	return EIO;

      total += n;
      smtp->s_offset += n;
      smtp->nl = memchr (smtp->buffer, '\n', total);
      if (smtp->nl == NULL)	/* Do we have a full line.  */
	{
	  /* Allocate a bigger buffer ?  */
	  if (total >= smtp->buflen - 1)
	    {
	      smtp->buflen *= 2;
	      smtp->buffer = realloc (smtp->buffer, smtp->buflen + 1);
	      if (smtp->buffer == NULL)
		return ENOMEM;
	    }
	}
      smtp->ptr = smtp->buffer + total;
    }
  while (smtp->nl == NULL);

  /* \r\n --> \n\0  */
  if (smtp->nl > smtp->buffer)
    {
      *(smtp->nl - 1) = '\n';
      *(smtp->nl) = '\0';
      smtp->ptr = smtp->nl;
    }

  MU_DEBUG1 (smtp->mailer->debug, MU_DEBUG_PROT, "< %s", smtp->buffer);

  return 0;
}

#else
#include <stdio.h>
#include <registrar0.h>
mu_record_t     mu_smtp_record = NULL;
mu_record_t     mu_remote_smtp_record = NULL;
#endif
