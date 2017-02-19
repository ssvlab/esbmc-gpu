/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#include "maidag.h"
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <signal.h>
#include <mu_umaxtostr.h>

mu_list_t lmtp_groups;
static int lmtp_transcript;

void
lmtp_reply (FILE *fp, char *code, char *enh, char *fmt, ...)
{
  va_list ap;
  char *str;
  
  va_start (ap, fmt);
  vasprintf (&str, fmt, ap);
  va_end (ap);

  if (lmtp_transcript)
    {
      if (enh)
	mu_diag_output (MU_DIAG_INFO, "LMTP reply: %s %s %s", code, enh, str);
      else
	mu_diag_output (MU_DIAG_INFO, "LMTP reply: %s %s", code, str);
    }
  
  if (!str)
    {
      mu_error (_("not enough memory"));
      exit (EX_TEMPFAIL);
    }

  while (*str)
    {
      char *end = strchr (str, '\n');

      if (end)
	{
	  size_t len = end - str;
	  fprintf (fp, "%s-", code);
	  if (enh)
	    fprintf (fp, "%s ", enh);
	  fprintf (fp, "%.*s\r\n", (int) len, str);
	  for (str = end; *str && *str == '\n'; str++);
	}
      else
	{
	  fprintf (fp, "%s ", code);
	  if (enh)
	    fprintf (fp, "%s ", enh);
	  fprintf (fp, "%s\r\n", str);
	  str += strlen (str);
	}
    }
}

void
xlatnl (char *arg)
{
  size_t len = strlen (arg);
  if (len > 0 && arg[len-1] == '\n')
    {
      len--;
      if (len > 0 && arg[len-1] == '\r')
	{
	  arg[len-1] = '\n';
	  arg[len] = 0;
	}
    }
}


enum lmtp_state
  {
    state_none,
    
    state_init,    
    state_lhlo,    
    state_mail,    
    state_rcpt,    
    state_data,    
    state_quit,    
    state_dot,

    state_end
  };

#define NSTATE ((int) state_end + 1)

enum lmtp_command
  {
    cmd_unknown,
    cmd_lhlo,
    cmd_mail,    
    cmd_rcpt,    
    cmd_data,
    cmd_quit,
    cmd_rset,
    cmd_help,
    cmd_dot
  };

#define NCMD ((int)cmd_dot + 1)

#define SNO state_none
#define SIN state_init
#define SHL state_lhlo
#define SML state_mail     
#define SRC state_rcpt     
#define SDA state_data     
#define SQT state_quit     
#define SDT state_dot     
#define SEN state_end

int transtab[NCMD][NSTATE] = {
/* state_     SNO  SIN  SHL  SML  SRC  SDA  SQT  SDT  SEN */
/* unkn */  { SNO, SNO, SNO, SNO, SNO, SNO, SNO, SNO, SEN },
/* lhlo */  { SNO, SHL, SNO, SNO, SNO, SNO, SNO, SNO, SNO },
/* mail */  { SNO, SNO, SML, SNO, SNO, SNO, SNO, SNO, SNO },
/* rcpt */  { SNO, SNO, SNO, SRC, SRC, SNO, SNO, SNO, SNO },
/* data */  { SNO, SNO, SNO, SNO, SDA, SNO, SNO, SNO, SNO },
/* quit */  { SNO, SEN, SEN, SEN, SEN, SEN, SEN, SEN, SEN },
/* rset */  { SNO, SIN, SIN, SIN, SIN, SIN, SIN, SIN, SNO },
/* help */  { SNO, SIN, SHL, SML, SRC, SDT, SQT, SDT, SEN },
/* dot  */  { SNO, SNO, SNO, SNO, SNO, SQT, SNO, SNO, SNO },
};


/* Delivery data */
char *lhlo_domain;     /* Sender domain */
char *mail_from;       /* Sender address */
mu_list_t rcpt_list;   /* Recipient addresses */
struct mail_tmp *mtmp; /* Temporary mail storage */
mu_mailbox_t mbox;     /* Collected mail body */


int
cfun_unknown (FILE *out, char *arg)
{
  lmtp_reply (out, "500", "5.5.1", "Command unrecognized");
  return 0;
}


static void
add_default_domain (char *str, int len, char **pret)
{
  *pret = malloc (len + 1 + strlen (lhlo_domain) + 1);
  if (!*pret)
    {
      mu_error (_("not enough memory"));
      exit (EX_SOFTWARE);
    }
  memcpy (*pret, str, len);
  (*pret)[len] = '@';
  strcpy (*pret + len + 1, lhlo_domain);
}

#define MAILER_DAEMON "MAILER-DAEMON"

int
check_address (char *arg, int with_domain, char **pret)
{
  if (strchr (arg, '@') == 0)
    {
      char *addr = NULL;
      size_t addrlen = 0;
      
      if (*arg == '<')
	{
	  size_t len = strlen (arg);
	  if (arg[len - 1] == '>')
	    {
	      if (len == 2) /* null address */
		{
		  if (!with_domain)
		    /* Null address is only legal in mail from */
		    return 1;
		  addr = MAILER_DAEMON;
		  addrlen = sizeof MAILER_DAEMON - 1;
		}
	      else
		{
		  addr = arg + 1;
		  addrlen = len - 2;
		}
	    }
	  else
	    return 1;
	}
      else
	{
	  addr = arg;
	  addrlen = strlen (arg);
	}

      if (with_domain)
	add_default_domain (addr, addrlen, pret);
      else
	{
	  *pret = malloc (addrlen + 1);
	  memcpy (*pret, addr, addrlen);
	  (*pret)[addrlen] = 0;
	}
    }
  else
    {
      mu_address_t addr;
      int rc = mu_address_create (&addr, arg);
      if (rc)
	return 1;
      if (with_domain)
	mu_address_aget_email (addr, 1, pret);
      else
	mu_address_aget_local_part (addr, 1, pret);
      mu_address_destroy (&addr);
    }
  return 0;
}


int
cfun_mail_from (FILE *out, char *arg)
{
  if (*arg == 0)
    {
      lmtp_reply (out, "501", "5.5.2", "Syntax error");
      return 1;
    }

  if (check_address (arg, 1, &mail_from))
    {
      lmtp_reply (out, "553", "5.1.8", "Address format error");
      return 1;
    }
  lmtp_reply (out, "250", "2.1.0", "Go ahead");
  return 0;
}


int
cfun_rcpt_to (FILE *out, char *arg)
{
  char *user;
  struct mu_auth_data *auth;
  
  if (*arg == 0)
    {
      lmtp_reply (out, "501", "5.5.2", "Syntax error");
      return 1;
    }

  /* FIXME: Check if domain is OK */
  if (check_address (arg, 0, &user))
    {
      lmtp_reply (out, "553", "5.1.8", "Address format error");
      return 1;
    }
  auth = mu_get_auth_by_name (user);
  if (!auth)
    {
      lmtp_reply (out, "550", "5.1.1", "User unknown");
      free (user);
      return 1;
    }
  mu_auth_data_free (auth);
  if (!rcpt_list)
    {
      mu_list_create (&rcpt_list);
      mu_list_set_destroy_item (rcpt_list, mu_list_free_item);
    }
  mu_list_append (rcpt_list, user);
  lmtp_reply (out, "250", "2.1.5", "Go ahead");
  return 0;
}  


int
cfun_data (FILE *out, char *arg)
{
  if (*arg)
    {
      lmtp_reply (out, "501", "5.5.2", "Syntax error");
      return 1;
    }

  if (mail_tmp_begin (&mtmp, mail_from))
    {
      /* FIXME: codes */
      lmtp_reply (out, "450", "4.1.0", "Temporary failure, try again later");
      return 1;
    }
  lmtp_reply (out, "354", NULL, "Go ahead");
  return 0;
}


int
dot_temp_fail (void *item, void *cbdata)
{
  char *name = item;
  FILE *out = cbdata;
  lmtp_reply (out, "450", "4.1.0", "%s: temporary failure", name);
  return 0;
}

int
dot_deliver (void *item, void *cbdata)
{
  char *name = item;
  FILE *out = cbdata;
  char *errp = NULL;
  mu_message_t msg;
  int status;
  
  if ((status = mu_mailbox_get_message (mbox, 1, &msg)) != 0)
    {
      mu_error (_("cannot get message from the temporary mailbox: %s"),
		mu_strerror (status));
      lmtp_reply (out, "450", "4.1.0",
		  "%s: temporary failure, try again later",
		  name);
      return 0;
    }

  switch (deliver (msg, name, &errp))
    {
    case 0:
      lmtp_reply (out, "250", "2.0.0", "%s: delivered", name);
      break;

    case EX_UNAVAILABLE:
      if (errp)
	lmtp_reply (out, "553", "5.1.8", "%s", errp);
      else
	lmtp_reply (out, "553", "5.1.8", "%s: delivery failed", name);
      break;

    default:
      if (errp)
	lmtp_reply (out, "450", "4.1.0", "%s", errp);
      else
	lmtp_reply (out, "450", "4.1.0",
		    "%s: temporary failure, try again later",
		    name);
      break;
    }
  free (errp);
  return 0;
}

int
cfun_dot (FILE *out, char *arg)
{
  if (!mtmp)
    mu_list_do (rcpt_list, dot_temp_fail, out);
  else
    {
      int rc = mail_tmp_finish (mtmp, &mbox);
      if (rc)
	mu_list_do (rcpt_list, dot_temp_fail, out);
      else
	{
	  mu_list_do (rcpt_list, dot_deliver, out);
	  mail_tmp_destroy (&mtmp);
	  mu_mailbox_destroy (&mbox);
	}
    }
  free (mail_from);
  mu_list_destroy (&rcpt_list);
  return 0;
}
  

int
cfun_rset (FILE *out, char *arg)
{
  free (lhlo_domain);
  free (mail_from);
  mu_list_destroy (&rcpt_list);
  mail_tmp_destroy (&mtmp);
  mu_mailbox_destroy (&mbox);
  lmtp_reply (out, "250", "2.0.0", "OK, forgotten");
  return 0;
}


char *capa_str = "ENHANCEDSTATUSCODES\n\
PIPELINING\n\
8BITMIME\n\
HELP";

int
cfun_lhlo (FILE *out, char *arg)
{
  if (*arg == 0)
    {
      lmtp_reply (out, "501", "5.0.0", "Syntax error");
      return 1;
    }
  lhlo_domain = strdup (arg);
  lmtp_reply (out, "250", NULL, "Hello\n");
  lmtp_reply (out, "250", NULL, capa_str);
  return 0;
}


int
cfun_quit (FILE *out, char *arg)
{
  lmtp_reply (out, "221", "2.0.0", "Bye");
  return 0;
}


int
cfun_help (FILE *out, char *arg)
{
  lmtp_reply (out, "200", "2.0.0", "Man, help yourself");
  return 0;
}


struct command_tab
{
  char *cmd_verb;
  int cmd_len;
  enum lmtp_command cmd_code;
  int (*cmd_fun) (FILE *, char *);
} command_tab[] = {
#define S(s) #s, (sizeof #s - 1)
  { S(lhlo), cmd_lhlo, cfun_lhlo },
  { S(mail from:), cmd_mail, cfun_mail_from },   
  { S(rcpt to:), cmd_rcpt, cfun_rcpt_to },   
  { S(data), cmd_data, cfun_data },
  { S(quit), cmd_quit, cfun_quit },
  { S(rset), cmd_rset, cfun_rset },
  { S(help), cmd_help, cfun_help },
  { S(.), cmd_dot, cfun_dot },
  { NULL, 0, cmd_unknown, cfun_unknown }
};

struct command_tab *
getcmd (char *buf, char **sp)
{
  struct command_tab *cp;
  size_t len = strlen (buf);
  for (cp = command_tab; cp->cmd_verb; cp++)
    {
      if (cp->cmd_len <= len
	  && mu_c_strncasecmp (cp->cmd_verb, buf, cp->cmd_len) == 0)
	{
	  *sp = buf + cp->cmd_len;
	  return cp;
	}
    }
  return cp;
}

static char *
to_fgets (char *buf, size_t size, FILE *fp, unsigned int timeout)
{
  char *p;
  alarm (timeout);
  p = fgets (buf, size, fp);
  alarm (0);
  return p;
}

int
lmtp_loop (FILE *in, FILE *out, unsigned int timeout)
{
  char buf[1024];
  enum lmtp_state state = state_init;

  setvbuf (in, NULL, _IOLBF, 0);
  setvbuf (out, NULL, _IOLBF, 0);

  lmtp_reply (out, "220", NULL, "At your service");
  while (to_fgets (buf, sizeof buf, in, timeout))
    {
      if (state == state_data
	  && !(buf[0] == '.'
	       && (buf[1] == '\n' || (buf[1] == '\r' && buf[2] == '\n'))))
	{
	  /* This is a special case */
	  if (mtmp)
	    {
	      size_t len;
	      int rc;

	      xlatnl (buf);
	      len = strlen (buf);
	      if ((rc = mail_tmp_add_line (mtmp, buf, len)))
		{
		  mail_tmp_destroy (&mtmp);
		  /* Wait the dot to report the error */
		}
	    }
	}
      else
	{
	  char *sp;
	  struct command_tab *cp = getcmd (buf, &sp);
	  enum lmtp_command cmd = cp->cmd_code;
	  enum lmtp_state next_state = transtab[cmd][state];

	  mu_rtrim_cset (sp, "\r\n");

	  if (lmtp_transcript)
	    mu_diag_output (MU_DIAG_INFO, "LMTP receive: %s", buf);
	      
	  if (next_state != state_none)
	    {
	      if (cp->cmd_fun)
		{
		  sp = mu_str_skip_class (sp, MU_CTYPE_SPACE);
		  if (cp->cmd_fun (out, sp))
		    continue;
		}
	      state = next_state;
	    }
	  else
	    lmtp_reply (out, "503", "5.0.0", "Syntax error");
	}
      
      if (state == state_end)
	break;
    }
  return 0;
}

typedef union
{
  struct sockaddr sa;
  struct sockaddr_in s_in;
  struct sockaddr_un s_un;
} all_addr_t;

int
lmtp_connection (int fd, struct sockaddr *sa, int salen, void *data,
		 mu_ip_server_t srv, time_t timeout, int transcript)
{
  lmtp_transcript = transcript;
  lmtp_loop (fdopen (fd, "r"), fdopen (fd, "w"), timeout);
  return 0;
}

static int
lmtp_set_privs ()
{
  gid_t gid;
  
  if (lmtp_groups)
    {
      gid_t *gidset = NULL;
      size_t size = 0;
      size_t j = 0;
      mu_iterator_t itr;
      int rc;

      mu_list_count (lmtp_groups, &size);
      gidset = calloc (size, sizeof (gidset[0]));
      if (!gidset)
	{
	  mu_error (_("not enough memory"));
	  return EX_UNAVAILABLE;
	}
      if (mu_list_get_iterator (lmtp_groups, &itr) == 0)
	{
	  for (mu_iterator_first (itr);
	       !mu_iterator_is_done (itr); mu_iterator_next (itr)) 
	    mu_iterator_current (itr,
				 (void **)(gidset + j++));
	  mu_iterator_destroy (&itr);
	}
      gid = gidset[0];
      rc = setgroups (j, gidset);
      free (gidset);
      if (rc)
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "setgroups", NULL, errno);
	  return EX_UNAVAILABLE;
	}
    }
  else
    {
      struct group *gr = getgrnam ("mail");
      if (gr == NULL)
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "getgrnam", "mail", errno);
	  return EX_UNAVAILABLE;
	}
      gid = gr->gr_gid;
    }
  if (setgid (gid) == -1)
    {
      mu_diag_funcall (MU_DIAG_ERROR, "setgid", "mail", errno);
      return EX_UNAVAILABLE;
    }
  return 0;
}
		
int
maidag_lmtp_server ()
{
  int rc = lmtp_set_privs ();
  if (rc)
    return rc;

  if (mu_m_server_mode (server) == MODE_DAEMON)
    {
      int status;
      mu_m_server_begin (server);
      status = mu_m_server_run (server);
      mu_m_server_end (server);
      mu_m_server_destroy (&server);
      if (status)
	return EX_CONFIG;
    }
  else 
    return lmtp_loop (stdin, stdout, 0);
}
