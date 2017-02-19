/* This file is part of GNU Mailutils
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 3, or (at
   your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#define MU_CFG_COMPATIBILITY /* This source uses deprecated cfg interfaces */
#include "mailutils/libcfg.h"
#include <string.h>
#include <stdlib.h>

struct mu_cfg_capa *cfg_capa_table[] = {
#define S(c) &__mu_common_cat3__(mu_,c,_cfg_capa)
  S (auth),
  S (mailbox),
  S (locking),
  S (address),
  S (mailer),
  S (logging),
  S (gsasl),
  S (pam),
  S (radius),
  S (sql),
  S (tls),
  S (virtdomain),
  S (sieve),
  S (ldap),
  S (debug),
  NULL
};

struct mu_cfg_capa *
find_cfg_capa (const char *name)
{
  int i;

  for (i = 0; cfg_capa_table[i]; i++)
    if (strcmp (cfg_capa_table[i]->name, name) == 0)
      return cfg_capa_table[i];
  return NULL;
}

static int
reserved_name (const char *name)
{
  static char *reserved[] = { "common", "license", NULL };
  char **p;
  for (p = reserved; *p; p++)
    if (strcmp (name, *p) == 0)
      return 1;
  return 0;
}

void
mu_libcfg_init (char **cnames)
{
  int i;
  for (i = 0; cnames[i]; i++)
    {
      if (!reserved_name (cnames[i]))
	{
	  struct mu_cfg_capa *cp = find_cfg_capa (cnames[i]);
	  if (!cp)
	    mu_error (_("unknown configuration group requested `%s'"),
		      cnames[i]);
	  else
	    mu_config_register_section (NULL, cp->name, NULL,
					cp->parser, cp->cfgparam);
	}
    }
}

int
mu_libcfg_parse_config (mu_cfg_tree_t **ptree)
{
  int flags = 0;
  int rc = 0;
  mu_cfg_tree_t *tree = NULL, *tmp;
  
  if (mu_cfg_parser_verbose)
    flags |= MU_PARSE_CONFIG_VERBOSE;
  if (mu_cfg_parser_verbose > 1)
    flags |= MU_PARSE_CONFIG_DUMP;
  
  if (mu_load_site_rcfile)
    {
      rc = mu_cfg_parse_file (&tmp, MU_CONFIG_FILE, flags);

      if (rc == ENOMEM)
	{
	  mu_error ("%s", mu_strerror (rc));
	  return rc;
	}
      else if (rc == 0)
	{
	  mu_cfg_tree_postprocess (tmp, flags | MU_PARSE_CONFIG_GLOBAL);
	  mu_cfg_tree_union (&tree, &tmp);
	}
    }

  if (mu_load_user_rcfile && mu_program_name)
    {
      size_t size = 3 + strlen (mu_program_name) + 1;
      char *file_name = malloc (size);
      if (file_name)
	{
	  strcpy (file_name, "~/.");
	  strcat (file_name, mu_program_name);

	  rc = mu_cfg_parse_file (&tmp, file_name, flags);
	  if (rc == ENOMEM)
	    {
	      mu_error ("%s", mu_strerror (rc));
	      mu_cfg_destroy_tree (&tree);
	      return rc;
	    }
	  else if (rc == 0)
	    {
	      mu_cfg_tree_postprocess (tmp, flags);
	      mu_cfg_tree_union (&tree, &tmp);
	    }
	  else if (rc == ENOENT)
	    rc = 0;
	  free (file_name);
	}
    }

  if (mu_load_rcfile)
    {
      rc = mu_cfg_parse_file (&tmp, mu_load_rcfile, flags);
      if (rc)
	{
	  mu_error (_("errors parsing file %s: %s"), mu_load_rcfile,
		    mu_strerror (rc));
	  mu_cfg_destroy_tree (&tree);
	  return rc;
	}
      else
	{
	  mu_cfg_tree_postprocess (tmp, flags);
	  mu_cfg_tree_union (&tree, &tmp);
	}
    }

  *ptree = tree;
  return rc;
}



/* FIXME: Deprecated */
int
mu_parse_config_files (struct mu_cfg_param *param, void *target)
{
  int flags = 0;
  int rc;
  
  if (mu_cfg_parser_verbose)
    flags |= MU_PARSE_CONFIG_VERBOSE;
  if (mu_cfg_parser_verbose > 1)
    flags |= MU_PARSE_CONFIG_DUMP;
  
  if (mu_load_site_rcfile)
    {
      rc = mu_parse_config (MU_CONFIG_FILE, mu_program_name, param,
			    flags | MU_PARSE_CONFIG_GLOBAL, target);
      if (rc == ENOMEM)
	{
	  mu_error ("%s", mu_strerror (rc));
	  return rc;
	}
    }

  if (mu_load_user_rcfile && mu_program_name)
    {
      size_t size = 3 + strlen (mu_program_name) + 1;
      char *file_name = malloc (size);
      if (file_name)
	{
	  strcpy (file_name, "~/.");
	  strcat (file_name, mu_program_name);

	  rc = mu_parse_config (file_name, mu_program_name, param, flags,
				target);
	  if (rc == ENOMEM)
	    {
	      mu_error ("%s", mu_strerror (rc));
	      return rc;
	    }

	  free (file_name);
	}
    }

  if (mu_load_rcfile)
    {
      rc = mu_parse_config (mu_load_rcfile, mu_program_name, param,
			    flags, target);
      if (rc)
	{
	  mu_error (_("errors parsing file %s: %s"), mu_load_rcfile,
		    mu_strerror (rc));
	  return rc;
	}
    }
	
  return 0;
}
