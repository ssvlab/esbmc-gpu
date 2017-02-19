/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2008, 2009, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library.  If not,
   see <http://www.gnu.org/licenses/>. */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#ifdef ENABLE_PROG
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>

#include <mailutils/address.h>
#include <mailutils/argcv.h>
#include <mailutils/debug.h>
#include <mailutils/errno.h>
#include <mailutils/header.h>
#include <mailutils/message.h>
#include <mailutils/observer.h>
#include <mailutils/progmailer.h>
#include <mailutils/vartab.h>

#include <url0.h>
#include <mailer0.h>
#include <registrar0.h>

static int _url_prog_init     (mu_url_t);

static struct _mu_record _prog_record =
{
  MU_PROG_PRIO,
  MU_PROG_SCHEME,
  _url_prog_init,    /* url init.  */
  _mu_mailer_mailbox_init,  /* Mailbox entry.  */
  _mu_mailer_prog_init, /* Mailer entry.  */
  _mu_mailer_folder_init, /* Folder entry.  */
  NULL, /* No need for a back pointer.  */
  NULL, /* _is_scheme method.  */
  NULL, /* _get_url method.  */
  NULL, /* _get_mailbox method.  */
  NULL, /* _get_mailer method.  */
  NULL  /* _get_folder method.  */
};

mu_record_t mu_prog_record = &_prog_record;


static int
_url_prog_uplevel (const mu_url_t orig, mu_url_t *up)
{
  return MU_ERR_NOENT;
}

static int
_url_prog_init (mu_url_t url)
{
  /* not valid in a prog url */
  if (url->secret || url->auth || url->host || url->port)
    return EINVAL;
  if (!url->path)
    return EINVAL;
  url->_uplevel = _url_prog_uplevel;
  return 0;
}


static void prog_destroy (mu_mailer_t);
static int prog_open (mu_mailer_t, int);
static int prog_close (mu_mailer_t);
static int prog_send_message (mu_mailer_t, mu_message_t, mu_address_t,
			      mu_address_t);

int
_mu_mailer_prog_init (mu_mailer_t mailer)
{
  int status;
  mu_progmailer_t pm;

  status = mu_progmailer_create (&pm);
  if (status)
    return status;

  mailer->data = pm;
  mailer->_destroy = prog_destroy;
  mailer->_open = prog_open;
  mailer->_close = prog_close;
  mailer->_send_message = prog_send_message;
  
  return 0;
}

static void
prog_destroy (mu_mailer_t mailer)
{
  mu_progmailer_destroy ((mu_progmailer_t*)&mailer->data);
}

static int
prog_open (mu_mailer_t mailer, int flags)
{
  mu_progmailer_t pm = mailer->data;
  int status;
  const char *path;
  
  /* Sanity checks.  */
  if (pm == NULL)
    return EINVAL;

  mailer->flags = flags;

  if ((status = mu_url_sget_path (mailer->url, &path)))
    return status;
  
  if (access (path, X_OK) == -1)
    return errno;

  mu_progmailer_set_debug (pm, mailer->debug);
  status = mu_progmailer_set_command (pm, path);
  MU_DEBUG1 (mailer->debug, MU_DEBUG_TRACE, "prog (%s)\n", path);
  return status;
}

static int
prog_close (mu_mailer_t mailer)
{
  return mu_progmailer_close (mailer->data);
}

static int
_expand_sender (const char *name, void *data, char **p)
{
  mu_address_t addr = data;
  char *email;
  int status = mu_address_aget_email (addr, 1, &email);

  if (status != 0)
    return status;
  *p = email;
  return 0;
}

struct ex_rcpt
{
  mu_message_t msg;
  mu_address_t addr;
  char *string;
};

static int
address_add (mu_address_t *paddr, const char *value)
{
  mu_address_t addr = NULL;
  int status;
  
  status = mu_address_create (&addr, value);
  if (status)
    return status;
  status = mu_address_union (paddr, addr);
  mu_address_destroy (&addr);
  return status;
}

static int
message_read_rcpt (mu_message_t msg, mu_address_t *paddr)
{
  mu_header_t header = NULL;
  const char *value;
  int status;
  
  status = mu_message_get_header (msg, &header);
  if (status)
    return status;
  
  status = mu_header_sget_value (header, MU_HEADER_TO, &value);

  if (status == 0)
    address_add (paddr, value);
  else if (status != MU_ERR_NOENT)
    return status;

  status = mu_header_sget_value (header, MU_HEADER_CC, &value);
  if (status == 0)
    address_add (paddr, value);
  else if (status != MU_ERR_NOENT)
    return status;

  status = mu_header_sget_value (header, MU_HEADER_BCC, &value);
  if (status == 0)
    address_add (paddr, value);
  else if (status != MU_ERR_NOENT)
    return status;
  return 0;
}

static int
_expand_rcpt (const char *name, void *data, char **p)
{
  struct ex_rcpt *exrcpt = data;
  int status;

  if (!exrcpt->string)
    {
      size_t i, count = 0;
      size_t len = 0;
      char *str;
      mu_address_t tmp_addr = NULL, addr;
      
      if (exrcpt->addr)
	addr = exrcpt->addr;
      else
	{
	  status = message_read_rcpt (exrcpt->msg, &tmp_addr);
	  if (status)
	    {
	      mu_address_destroy (&tmp_addr);
	      return status;
	    }
	  addr = tmp_addr;
	}
	    
      mu_address_get_count (addr, &count);
      for (i = 1; i <= count; i++)
	{
	  const char *email;
	  if (i > 1)
	    len++;
	  if ((status = mu_address_sget_email (addr, i, &email)) != 0)
	    {
	      mu_address_destroy (&tmp_addr);
	      return status;
	    }
	  len += strlen (email);
	}

      str = malloc (len + 1);
      if (!str)
	{
	  mu_address_destroy (&tmp_addr);
	  return ENOMEM;
	}
      exrcpt->string = str;
      
      for (i = 1; i <= count; i++)
	{
	  const char *email;
	  if (i > 1)
	    *str++ = ' ';
	  if (mu_address_sget_email (addr, i, &email))
	    continue;
	  strcpy (str, email);
	  str += strlen (email);
	}
      *str = 0;
      mu_address_destroy (&tmp_addr);
    }  
  *p = exrcpt->string;
  return 0;
}

void
_free_rcpt (void *data, char *value)
{
  free (value);
}

static int
url_to_argv (mu_url_t url, mu_message_t msg,
	     mu_address_t from, mu_address_t to,
	     int *pargc, char ***pargv)
{
  int rc;
  mu_vartab_t vtab;
  struct ex_rcpt ex_rcpt;
  char **query;
  size_t i;
  size_t argc;
  char **argv;
  
  ex_rcpt.msg = msg;
  ex_rcpt.addr = to;
  ex_rcpt.string = NULL;
  mu_vartab_create (&vtab);
  mu_vartab_define_exp (vtab, "sender", _expand_sender, NULL, from);
  mu_vartab_define_exp (vtab, "rcpt", _expand_rcpt, _free_rcpt, &ex_rcpt);

  rc = mu_url_sget_query (url, &argc, &query);
  if (rc)
    return rc;

  argv = calloc (argc + 1, sizeof (argv[0]));
  if (!argv)
    return ENOMEM;

  for (i = 0; i < argc; i++)
    {
      if ((rc = mu_vartab_expand (vtab, query[i], &argv[i])))
	{
	  mu_argcv_free (i, argv);
	  mu_vartab_destroy (&vtab);
	  return rc;
	}
    }
  argv[i] = NULL;
  
  mu_vartab_destroy (&vtab);

  *pargc = argc;
  *pargv = argv;
  return 0;
}

static int
prog_send_message (mu_mailer_t mailer, mu_message_t msg, mu_address_t from,
		   mu_address_t to)
{
  mu_progmailer_t pm = mailer->data;
  int argc;
  char **argv;
  int status;
  const char *command;

  status = mu_url_sget_path (mailer->url, &command);
  if (status && status != MU_ERR_NOENT)
    {
      MU_DEBUG1 (mailer->debug, MU_DEBUG_ERROR,
		 "cannot get path from URL: %s\n",
		 mu_strerror (status));
      return status;
    }
  status = mu_progmailer_set_command (pm, command);
  if (status)
    {
      MU_DEBUG1 (mailer->debug, MU_DEBUG_ERROR,
		 "cannot set progmailer command: %s\n",
		 mu_strerror (status));
      return status;
    }
      
  status = url_to_argv (mailer->url, msg, from, to, &argc, &argv);
  if (status)
    {
      MU_DEBUG1 (mailer->debug, MU_DEBUG_ERROR,
		 "cannot convert URL to command line: %s\n",
		 mu_strerror (status));
      return status;
    }

  mu_progmailer_set_debug (pm, mailer->debug);
  status = mu_progmailer_open (pm, argv);
  if (status == 0)
    {
      status = mu_progmailer_send (pm, msg);
      if (status == 0)
	mu_observable_notify (mailer->observable, MU_EVT_MAILER_MESSAGE_SENT,
			      msg);
      else
	MU_DEBUG1 (mailer->debug, MU_DEBUG_ERROR,
		   "progmailer error: %s\n",
		   mu_strerror (status));
    }
  
  mu_argcv_free (argc, argv);
  return status;
}

#else
#include <stdio.h>
#include <registrar0.h>
mu_record_t mu_prog_record = NULL;
#endif
