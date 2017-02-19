/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

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
#include <stdlib.h>
#include <string.h>
#include <mailutils/types.h>
#include <mailutils/gocs.h>
#include <mailutils/mailbox.h>
#include <mailutils/locker.h>
#include <mailutils/mutil.h>
#include <mailutils/mailer.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/nls.h>
#include <mailutils/debug.h>
#include <mailutils/syslog.h>
#include <mailutils/registrar.h>
#include <syslog.h>

int mu_load_user_rcfile = 1;
int mu_load_site_rcfile = 1;
char *mu_load_rcfile = NULL;


int
mu_gocs_dummy (enum mu_gocs_op op, void *data)
{
  return 0;
}

int
mu_gocs_mailbox_init (enum mu_gocs_op op, void *data)
{
  int rc;
  struct mu_gocs_mailbox *p = data;

  if (op == mu_gocs_op_set && p)
    {
      if (p->mail_spool)
	{
	  rc = mu_set_mail_directory (p->mail_spool);
	  if (rc)
	    mu_error (_("cannot set mail directory name to `%s': %s"),
		      p->mail_spool, mu_strerror (rc));
	  free (p->mail_spool);
	  p->mail_spool = NULL;
	}
      if (p->mailbox_pattern)
	{
	  rc = mu_set_mailbox_pattern (p->mailbox_pattern);
	  if (rc)
	    mu_error (_("cannot set mailbox pattern to `%s': %s"),
		      p->mailbox_pattern, mu_strerror (rc));
	  free (p->mailbox_pattern);
	  p->mailbox_pattern = NULL;
	}
      if (p->mailbox_type)
	{
	  if (mu_registrar_set_default_scheme (p->mailbox_type))
	    mu_error (_("invalid mailbox type: %s"), p->mailbox_type);
	  free (p->mailbox_type);
	  p->mailbox_type = NULL;
	}
    }
  return 0;
}

int
mu_gocs_locking_init (enum mu_gocs_op op, void *data)
{
  struct mu_gocs_locking *p = data;
  
  if (!(op == mu_gocs_op_set && p))
    return 0;

  if (p->lock_flags)
    {
      int flags = 0;
      char *s;
      
      for (s = p->lock_flags; *s; s++)
	{
	  switch (*s)
	    {
	    case 'E':
	      flags |= MU_LOCKER_EXTERNAL;
	      break;
	      
	    case 'R':
	      flags |= MU_LOCKER_RETRY;
	      break;
	      
	    case 'T':
	      flags |= MU_LOCKER_TIME;
	      break;
	      
	    case 'P':
	      flags |= MU_LOCKER_PID;
	      break;
	      
	    default:
	      mu_error (_("invalid lock flag `%c'"), *s);
	    }
	}
      mu_locker_set_default_flags (flags, mu_locker_assign);
      free (p->lock_flags);
      p->lock_flags = NULL;
    }

  if (p->lock_retry_count)
    {
      mu_locker_set_default_retry_count (p->lock_retry_count);
      mu_locker_set_default_flags (MU_LOCKER_RETRY, mu_locker_set_bit);
      p->lock_retry_count = 0;
    }

  if (p->lock_retry_timeout)
    {
      mu_locker_set_default_retry_timeout (p->lock_retry_timeout);
      mu_locker_set_default_flags (MU_LOCKER_RETRY, mu_locker_set_bit);
      p->lock_retry_timeout = 0;
    }

  if (p->lock_expire_timeout)
    {
      mu_locker_set_default_expire_timeout (p->lock_expire_timeout);
      mu_locker_set_default_flags (MU_LOCKER_EXTERNAL, mu_locker_set_bit);
      p->lock_expire_timeout = 0;
    }

  if (p->external_locker)
    {
      mu_locker_set_default_external_program (p->external_locker);
      mu_locker_set_default_flags (MU_LOCKER_TIME, mu_locker_set_bit);
      free (p->external_locker);
      p->external_locker = NULL;
    }
  return 0;
}

int
mu_gocs_source_email_init (enum mu_gocs_op op, void *data)
{
  struct mu_gocs_source_email *p = data;
  int rc;

  if (!(op == mu_gocs_op_set && p))
    return 0;
  
  if (p->address)
    {
      if ((rc = mu_set_user_email (p->address)) != 0)
	mu_error (_("invalid email address `%s': %s"),
		  p->address, mu_strerror (rc));
      free (p->address);
      p->address = NULL;
    }

  if (p->domain)
    {
      if ((rc = mu_set_user_email_domain (p->domain)) != 0)
	mu_error (_("invalid email domain `%s': %s"),
		  p->domain, mu_strerror (rc));

      free (p->domain);
      p->domain = NULL;
    }
  return 0;
}

int
mu_gocs_mailer_init (enum mu_gocs_op op, void *data)
{
  struct mu_gocs_mailer *p = data;
  int rc;

  if (!(op == mu_gocs_op_set && p))
    return 0;
  
  if (p->mailer)
    {
      if ((rc = mu_mailer_set_url_default (p->mailer)) != 0)
	mu_error (_("invalid mailer URL `%s': %s"),
		  p->mailer, mu_strerror (rc));
      free (p->mailer);
      p->mailer = NULL;
    }
  return 0;
}

int
mu_gocs_logging_init (enum mu_gocs_op op, void *data)
{
  struct mu_gocs_logging *p = data;

  if (op == mu_gocs_op_set)
    {
      if (!p)
	{
	  static struct mu_gocs_logging default_gocs_logging = { LOG_FACILITY };
	  p = &default_gocs_logging;
	}
  
      if (p->facility)
	{
	  mu_log_facility = p->facility;
	  mu_debug_default_printer = mu_debug_syslog_printer;
	}
      else
	mu_debug_default_printer = mu_debug_stderr_printer;

      if (p->tag)
	mu_log_tag = strdup (p->tag);
    }
  return 0;
}

int
mu_gocs_debug_init (enum mu_gocs_op op, void *data)
{
  if (op == mu_gocs_op_set && data)
    {
      struct mu_gocs_debug *p = data;
      if (p->string && p->errpfx)
	{
	  mu_global_debug_from_string (p->string, p->errpfx);
	  free (p->errpfx);
	}
      if (p->line_info >= 0)
	mu_debug_line_info = p->line_info;
    }
  return 0;
}


struct mu_gocs_entry
{
  const char *name;
  mu_gocs_init_fp init;
};

#define MAX_GOCS 512

static struct mu_gocs_entry _gocs_table[MAX_GOCS];

void
mu_gocs_register (const char *capa, mu_gocs_init_fp init)
{
  int i;
  for (i = 0; _gocs_table[i].name; i++)
    if (i == MAX_GOCS-1)
      {
	mu_error (_("gocs table overflow"));
	abort ();
      }
  _gocs_table[i].name = capa;
  _gocs_table[i].init = init;
}

int
mu_gocs_enumerate (mu_list_action_t action, void *data)
{
  int i;
  
  for (i = 0; _gocs_table[i].name; i++)
    {
      int rc = action ((void*) _gocs_table[i].name, data);
      if (rc)
	return rc;
    }
  return 0;
}

static mu_gocs_init_fp
find_init_function (struct mu_gocs_entry *tab, const char *capa)
{
  for (; tab->name; tab++)
    if (strcmp (tab->name, capa) == 0)
      return tab->init;
  return NULL;
}

static struct mu_gocs_entry std_gocs_table[] = {
  { "common", mu_gocs_dummy },
  { "license", mu_gocs_dummy },
  { "mailbox", mu_gocs_mailbox_init },
  { "locking", mu_gocs_locking_init },
  { "address", mu_gocs_source_email_init },
  { "mailer", mu_gocs_mailer_init },
  { "logging", mu_gocs_logging_init },
  { "debug", mu_gocs_debug_init },
  { "auth", mu_gocs_dummy },
  { NULL }
};

void
mu_gocs_register_std (const char *name)
{
  mu_gocs_init_fp init = find_init_function (std_gocs_table, name);
  if (!init)
    {
      mu_error (_("INTERNAL ERROR at %s:%d: unknown standard capability `%s'"),
		__FILE__, __LINE__, name);
      abort ();
    }
  mu_gocs_register (name, init);
}


struct mu_gocs_data
{
  char *capa;
  void *data;
};

static mu_list_t /* of struct mu_gocs_data */ data_list;

static int
_gocs_comp (const void *a, const void *b)
{
  const struct mu_gocs_data *da = a, *db = b;
  return !(strcmp (da->capa, db->capa) == 0 && da->data == db->data);
}

void
mu_gocs_store (char *capa, void *data)
{
  struct mu_gocs_data *s;
  if (!data_list)
    {
      mu_list_create (&data_list);
      mu_list_set_destroy_item (data_list, mu_list_free_item);
      mu_list_set_comparator (data_list, _gocs_comp);
    }
  s = malloc (sizeof *s);
  if (!s)
    {
      mu_error ("%s", mu_strerror (ENOMEM));
      exit (1);
    }
  s->capa = capa;
  s->data = data;
  if (mu_list_locate (data_list, s, NULL) == 0)
    free (s);
  else
    mu_list_prepend (data_list, s);
}

int
_gocs_flush (void *item, void *data)
{
  struct mu_gocs_data *s = item;
  mu_gocs_init_fp initfun = find_init_function (_gocs_table, s->capa);

  if (!initfun)
    {
      mu_error (_("INTERNAL ERROR at %s:%d: unknown capability `%s'"),
		__FILE__, __LINE__, s->capa);
      abort ();
    }

  if (initfun (mu_gocs_op_set, s->data))
    {
      mu_error (_("initialization of GOCS `%s' failed"), s->capa);
      return 1;
    }
  
  return 0;
}

void
mu_gocs_flush ()
{
  int i;
  mu_list_do (data_list, _gocs_flush, NULL);

  for (i = 0; _gocs_table[i].name; i++)
    _gocs_table[i].init (mu_gocs_op_flush, NULL);
}
