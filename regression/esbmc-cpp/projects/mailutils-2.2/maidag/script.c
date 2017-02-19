/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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

struct script_tab
{
  char *lang;
  char *suf;
  maidag_script_fun fun;
};

struct script_tab script_tab[] = {
#ifdef WITH_PYTHON
  { "python", "py\0pyc\0", python_check_msg },
#endif
  { "sieve",  "sv\0siv\0sieve\0", sieve_check_msg },
#ifdef WITH_GUILE
  { "scheme", "scm\0", scheme_check_msg },
#endif
  { NULL }
};

maidag_script_fun
script_lang_handler (const char *lang)
{
  struct script_tab *p;

  for (p = script_tab; p->lang; p++)
    if (strcmp (p->lang, lang) == 0)
      return p->fun;
  return NULL;
}

maidag_script_fun
script_suffix_handler (const char *name)
{
  struct script_tab *p;
  char *suf;
  
  suf = strrchr (name, '.');
  if (!suf)
    return NULL;
  suf++;
  
  for (p = script_tab; p->lang; p++)
    {
      char *s;

      for (s = p->suf; *s; s += strlen (s) + 1)
	if (strcmp (s, suf) == 0)
	  return p->fun;
    }
  return NULL;
}



int
script_register (const char *pattern)
{
  maidag_script_fun fun;
  struct maidag_script *scr;
  
  if (script_handler)
    fun = script_handler;
  else
    {
      fun = script_suffix_handler (pattern);
      if (!fun)
	return EINVAL;
    }

  scr = malloc (sizeof (*scr));
  if (!scr)
    return MU_ERR_FAILURE;
  
  scr->fun = fun;
  scr->pat = pattern;

  if (!script_list)
    {
      if (mu_list_create (&script_list))
	return MU_ERR_FAILURE;
    }

  if (mu_list_append (script_list, scr))
    return MU_ERR_FAILURE;

  return 0;
}


struct apply_script_closure
{
  struct mu_auth_data *auth;
  mu_message_t msg;
};

static int
apply_script (void *item, void *data)
{
  struct maidag_script *scr = item;
  struct apply_script_closure *clos = data;
  char *progfile;
  int rc;
  struct stat st;
  
  progfile = mu_expand_path_pattern (scr->pat, clos->auth->name);
  if (stat (progfile, &st))
    {
      if (debug_level > 2)
	mu_diag_funcall (MU_DIAG_DEBUG, "stat", progfile, errno);
      else if (errno != ENOENT)
	mu_diag_funcall (MU_DIAG_NOTICE, "stat", progfile, errno);
      free (progfile);
      return 0;
    }

  rc = scr->fun (clos->msg, clos->auth, progfile);
  free (progfile);

  if (rc == 0)
    {
      mu_attribute_t attr;
      mu_message_get_attribute (clos->msg, &attr);
      rc = mu_attribute_is_deleted (attr);
    }
  
  return rc;
}
  
int
script_apply (mu_message_t msg, struct mu_auth_data *auth)
{
  int rc = 0;
  
  if (script_list)
    {
      mu_attribute_t attr;
      struct apply_script_closure clos;

      clos.auth = auth;
      clos.msg = msg;

      mu_message_get_attribute (msg, &attr);
      mu_attribute_unset_deleted (attr);
      if (switch_user_id (auth, 1) == 0)
	{
	  chdir (auth->dir);
	  rc = mu_list_do (script_list, apply_script, &clos);
	  chdir ("/");
	  switch_user_id (auth, 0);
	}
    }
  return rc;
}
