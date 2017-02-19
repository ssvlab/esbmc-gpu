/* This file is part of GNU Mailutils
   Copyright (C) 2007, 2008, 2010 Free Software Foundation, Inc.

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
#include "mailutils/libcfg.h"
#include <mailutils/mu_auth.h>
#include <string.h>

/* FIXME: mu_auth.c should be reviewed */


/* ************************************************************************* */
/* Resource-style configuration                                              */
/* ************************************************************************* */
static int
cb_authentication (mu_debug_t err, void *data, mu_config_value_t *val)
{
  if (val->type == MU_CFG_STRING)
    {
      if (strcmp (val->v.string, "clear") == 0)
	mu_authentication_clear_list ();
      else
	/*FIXME: use err for error reporting*/
	mu_authentication_add_module_list (val->v.string);
    }
  else if (val->type == MU_CFG_LIST)
    {
      int i;
      for (i = 0; i < val->v.arg.c; i++)
	{
	  if (mu_cfg_assert_value_type (&val->v.arg.v[i], MU_CFG_STRING, err))
	    return 1;
	  if (strcmp (val->v.arg.v[i].v.string, "clear") == 0)
	    mu_authentication_clear_list ();
	  else
	    mu_authentication_add_module (val->v.arg.v[i].v.string);
	}
    }
  else
    {
      mu_cfg_format_error (err, MU_DEBUG_ERROR, _("expected string value"));
      return 1;
    }
  return 0;
}

static int
cb_authorization (mu_debug_t err, void *data, mu_config_value_t *val)
{
  if (val->type == MU_CFG_STRING)
    {
      if (strcmp (val->v.string, "clear") == 0)
	mu_authorization_clear_list ();
      else
	/*FIXME: use err for error reporting*/
	mu_authorization_add_module_list (val->v.string);
    }
  else if (val->type == MU_CFG_LIST)
    {
      int i;
      for (i = 0; i < val->v.arg.c; i++)
	{
	  if (mu_cfg_assert_value_type (&val->v.arg.v[i], MU_CFG_STRING, err))
	    return 1;
	  if (strcmp (val->v.arg.v[i].v.string, "clear") == 0)
	    mu_authorization_clear_list ();
	  else
	    mu_authorization_add_module (val->v.arg.v[i].v.string);
	}
    }
  else
    {
      mu_cfg_format_error (err, MU_DEBUG_ERROR, _("expected string value"));
      return 1;
    }
  return 0;
}

static struct mu_cfg_param mu_auth_param[] = {
  { "authentication", mu_cfg_callback, NULL, 0, cb_authentication,
    /* FIXME: The description is incomplete. MU-list is also allowed as
       argument */
    N_("Set a list of modules for authentication. Modlist is a "
       "colon-separated list of module names or a word `clear' to "
       "clear the previously set up values."),
    N_("modlist") },
  { "authorization", mu_cfg_callback, NULL, 0, cb_authorization,
    N_("Set a list of modules for authorization. Modlist is a "
       "colon-separated list of module names or a word `clear' to "
       "clear the previously set up values."),
    N_("modlist") },
  { NULL }
};

int
mu_auth_section_parser
   (enum mu_cfg_section_stage stage, const mu_cfg_node_t *node,
    const char *section_label, void **section_data, void *call_data,
    mu_cfg_tree_t *tree)
{
  switch (stage)
    {
    case mu_cfg_section_start:
      break;

    case mu_cfg_section_end:
      mu_auth_finish_setup ();
    }
  return 0;
}

struct mu_cfg_capa mu_auth_cfg_capa = {
  "auth",  mu_auth_param, mu_auth_section_parser
};
