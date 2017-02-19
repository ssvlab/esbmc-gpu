/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2008, 2009, 2010 Free Software Foundation, Inc.

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

/* Implementation of ID extension (RFC 2971) */

#include "imap4d.h"
#include <sys/utsname.h>

mu_list_t imap4d_id_list;

static int
eat_args (imap4d_tokbuf_t tok)
{
  int n = IMAP4_ARG_1;
  char *p;

  p = imap4d_tokbuf_getarg (tok, n++);
  if (!p)
    return RESP_BAD;
  if (mu_c_strcasecmp (p, "NIL") == 0)
    {
      if (imap4d_tokbuf_getarg (tok, n))
	return RESP_BAD;
      return RESP_OK;
    }
  else if (p[0] != '(')
    return RESP_BAD;

  /* Collect arguments */
  while ((p = imap4d_tokbuf_getarg (tok, n++)))
    {
      if (p[0] == ')')
	{
	  if (imap4d_tokbuf_getarg (tok, n))
	    return RESP_BAD;
	  return RESP_OK;
	}
    }
  return RESP_BAD;
}

struct id_value
{
  char *name;
  char *value;
  const char *(*fun) (struct id_value *idv);
};

static const char *
get_os (struct id_value *idv)
{
  struct utsname uts;
  uname(&uts);
  return idv->value = mu_strdup (uts.sysname);
}

static const char *
get_os_version (struct id_value *idv)
{
  struct utsname uts;
  uname(&uts);
  return idv->value = mu_strdup (uts.version);
}

static const char *
get_command (struct id_value *idv MU_ARG_UNUSED)
{
  return mu_program_name;
}

static char *
build_str (char **argv)
{
  size_t size = 0;
  int i, j;
  char *buf, *p;
  
  for (j = 0; argv[j]; j++)
    {
      size_t len = strlen (argv[j]);
      if (size + len + 1 > 1024)
	break;
      size += len + 1;
    }
  
  buf = mu_alloc (size);
  for (i = 0, p = buf; argv[i];)
    {
      strcpy (p, argv[i]);
      p += strlen (p);
      if (++i < j)
	*p++ = ' ';
      else
	break;
    }
  *p = 0;
  return buf;
}  

static const char *
get_arguments (struct id_value *idv)
{
  return idv->value = build_str (imap4d_argv + 1);
}

static const char *
get_environment (struct id_value *idv)
{
  extern char **environ;
  return idv->value = build_str (environ);
}
			 
static struct id_value id_tab[] = {
  { "name", PACKAGE_NAME },
  { "version", PACKAGE_VERSION },
  { "vendor", "GNU" },
  { "support-url", "http://www.gnu.org/software/mailutils" },
  { "address",
    "51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA" },
#if 0
  /* FIXME */
  { "date", NULL },
#endif
  { "os", NULL, get_os },
  { "os-version", NULL, get_os_version },
  { "command", NULL, get_command },
  { "arguments", NULL, get_arguments },
  { "environment", NULL, get_environment },
  { NULL }
};

static const char *
get_id_value (const char *name)
{
  struct id_value *idv;
  const char *val = NULL;
  
  for (idv = id_tab; idv->name; idv++)
    {
      if (strcmp (idv->name, name) == 0)
	{
	  if (idv->value)
	    val = idv->value;
	  else if (idv->fun)
	    val = idv->fun (idv);
	  break;
	}
    }
  return val;
}

int
imap4d_id (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  int rc = eat_args (tok);
  if (rc != RESP_OK)
    return util_finish (command, rc, "Syntax error");
  if (imap4d_id_list)
    {
      mu_iterator_t itr;
      int i;
      int outcnt = 0;
      
      mu_list_get_iterator (imap4d_id_list, &itr);
      for (i = 0, mu_iterator_first (itr);
	   i < 30 && !mu_iterator_is_done (itr);
	   i++, mu_iterator_next (itr))
	{
	  const char *p, *q;
	  size_t len;
	  
	  mu_iterator_current (itr, (void**)&p);
	  len = strcspn (p, "=");
	  if (p[len])
	    q = p + len + 1;
	  else
	    q = get_id_value (p);

	  if (q)
	    {
	      if (outcnt++ == 0)
		util_send ("* ID (");
	      else
		util_send (" ");
	      util_send ("\"%*.*s\" \"%s\"", (int) len, (int) len, p, q);
	    }
	}
      mu_iterator_destroy (&itr);
      if (outcnt)
	util_send (")\r\n");
    }
  return util_finish (command, RESP_OK, "Completed");
}
