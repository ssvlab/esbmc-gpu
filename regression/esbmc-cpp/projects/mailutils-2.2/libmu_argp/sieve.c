/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2009, 2010 Free
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
#include "cmdline.h"
#include "mailutils/sieve.h"

enum {
  OPT_CLEAR_INCLUDE_PATH = 256,
  OPT_CLEAR_LIBRARY_PATH
};  

static struct argp_option sieve_argp_option[] = {
  { "includedir", 'I', N_("DIR"), 0,
    N_("append DIR to the list of directories searched for include files"), 0 },
  { "libdir", 'L', N_("DIR"), 0,
    N_("append DIR to the list of directories searched for library files"), 0 },
  { "clear-include-path", OPT_CLEAR_INCLUDE_PATH, NULL, 0,
    N_("clear Sieve include path"), 0 },
  { "clear-library-path", OPT_CLEAR_LIBRARY_PATH, NULL, 0,
    N_("clear Sieve library path"), 0 },
  { "clearpath", 0, NULL, OPTION_ALIAS, NULL },
  { NULL,      0, NULL, 0, NULL, 0 }
};

static error_t
sieve_argp_parser (int key, char *arg, struct argp_state *state)
{
  static mu_list_t lst;
  
  switch (key)
    {
    case 'I':
      mu_argp_node_list_new (lst, "include-path", arg);
      break;

    case 'L':
      mu_argp_node_list_new (lst, "library-path", arg);
      break;

    case OPT_CLEAR_INCLUDE_PATH:
      mu_argp_node_list_new (lst, "clear-include-path", "yes");
      break;

    case OPT_CLEAR_LIBRARY_PATH:
      mu_argp_node_list_new (lst, "clear-library-path", "yes");
      break;
      
    case ARGP_KEY_INIT:
      mu_argp_node_list_init (&lst);
      break;
      
    case ARGP_KEY_FINI:
      mu_argp_node_list_finish (lst, "sieve", NULL);
      break;
			   
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp sieve_argp = {
  sieve_argp_option,
  sieve_argp_parser,
};

static struct argp_child sieve_argp_child = {
  &sieve_argp,
  0,
  N_("Sieve options"),
  0
};

struct mu_cmdline_capa mu_sieve_cmdline = {
  "sieve", &sieve_argp_child
};

