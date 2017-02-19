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
#include "cmdline.h"
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <mailutils/syslog.h>
#include <mailutils/mailbox.h>


/* ************************************************************************* */
/* Common                                                                    */
/* ************************************************************************* */

enum {
  OPT_SHOW_OPTIONS=256,          
  OPT_NO_USER_RCFILE,
  OPT_NO_SITE_RCFILE,
  OPT_RCFILE,
  OPT_RCFILE_LINT,
  OPT_RCFILE_VERBOSE,
  OPT_LOG_FACILITY,          
  OPT_LICENSE,
  OPT_DEBUG_LEVEL,
  OPT_LINE_INFO,
  OPT_HELP_CONFIG,
  OPT_SET
};

static struct argp_option mu_common_argp_options[] = 
{
  { NULL, 0, NULL, 0, N_("Common options"), 0},
  { "show-config-options", OPT_SHOW_OPTIONS, NULL, 0,
    N_("show compilation options"), 0 },
  { "config-help", OPT_HELP_CONFIG, NULL, 0,
    N_("show configuration file summary"), 0 },
  { "no-user-config", OPT_NO_USER_RCFILE, NULL, 0,
    N_("do not load user configuration file"), 0 },
  { "no-user-rcfile", 0, NULL, OPTION_ALIAS, NULL },
  { "no-site-config", OPT_NO_SITE_RCFILE, NULL, 0,
    N_("do not load site configuration file"), 0 },
  { "no-site-rcfile", 0, NULL, OPTION_ALIAS, NULL },
  { "config-file", OPT_RCFILE, N_("FILE"), 0,
    N_("load this configuration file"), 0, },
  { "rcfile", 0, NULL, OPTION_ALIAS, NULL },
  { "config-verbose", OPT_RCFILE_VERBOSE, NULL, 0,
    N_("verbosely log parsing of the configuration files"), 0 },
  { "rcfile-verbose", 0, NULL, OPTION_ALIAS, NULL },
  { "config-lint", OPT_RCFILE_LINT, NULL, 0,
    N_("check configuration file syntax and exit"), 0 },
  { "rcfile-lint", 0, NULL, OPTION_ALIAS, NULL },
  { "set", OPT_SET, N_("PARAM=VALUE"), 0,
    N_("set configuration parameter"), 0 },
  { NULL, 0, NULL, 0, NULL, 0 }
};

static void
set_config_param (const char *path, struct argp_state *state)
{
  mu_cfg_node_t *node;
  int rc = mu_cfg_create_subtree (path, &node);
  if (rc)
    argp_error (state, "cannot create node: %s", mu_strerror (rc));
  mu_cfg_tree_add_node (mu_argp_tree, node);
}

static error_t
mu_common_argp_parser (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case OPT_SHOW_OPTIONS:
      mu_print_options ();
      exit (0);

    case OPT_NO_USER_RCFILE:
      mu_load_user_rcfile = 0;
      break;
      
    case OPT_NO_SITE_RCFILE:
      mu_load_site_rcfile = 0;
      break;
      
    case OPT_RCFILE:
      mu_load_rcfile = arg;
      break;

    case OPT_RCFILE_LINT:
      mu_cfg_parser_verbose++;
      mu_rcfile_lint = 1;
      break;
      
    case OPT_RCFILE_VERBOSE:
      mu_cfg_parser_verbose++;
      break;

    case OPT_HELP_CONFIG:
      mu_help_config_mode = 1;
      break;

    case OPT_SET:
      set_config_param (arg, state);
      break;
      
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

struct argp mu_common_argp = {
  mu_common_argp_options,
  mu_common_argp_parser,
};

struct argp_child mu_common_argp_child = {
  &mu_common_argp,
  0,
  NULL,
  0,
};

struct mu_cmdline_capa mu_common_cmdline = {
  "common", &mu_common_argp_child
};


/* ************************************************************************* */
/* Logging                                                                   */
/* ************************************************************************* */

static struct argp_option mu_logging_argp_option[] = {
  {"log-facility", OPT_LOG_FACILITY, N_("FACILITY"), 0,
   N_("output logs to syslog FACILITY"), 0},
  { NULL,      0, NULL, 0, NULL, 0 }
};

static error_t
mu_logging_argp_parser (int key, char *arg, struct argp_state *state)
{
  static mu_list_t lst;
  
  switch (key)
    {
      /* log */
    case OPT_LOG_FACILITY:
      mu_argp_node_list_new (lst, "facility", arg);
      break;
	  
    case ARGP_KEY_INIT:
      mu_argp_node_list_init (&lst);
      break;
      
    case ARGP_KEY_FINI:
      mu_argp_node_list_finish (lst, "logging", NULL);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

struct argp mu_logging_argp = {
  mu_logging_argp_option,
  mu_logging_argp_parser,
};

struct argp_child mu_logging_argp_child = {
  &mu_logging_argp,
  0,
  NULL,
  0
};

struct mu_cmdline_capa mu_logging_cmdline = {
  "logging", &mu_logging_argp_child
};


/* ************************************************************************* */
/* License                                                                   */
/* ************************************************************************* */

/* Option to print the license. */
static struct argp_option mu_license_argp_option[] = {
  { "license", OPT_LICENSE, NULL, 0, N_("print license and exit"), -2 },
  { NULL,      0, NULL, 0, NULL, 0 }
};

static error_t
mu_license_argp_parser (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case OPT_LICENSE:
      printf (_("License for %s:\n\n"), argp_program_version);
      printf ("%s", mu_license_text);
      exit (0);

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

struct argp mu_license_argp = {
  mu_license_argp_option,
  mu_license_argp_parser,
};

struct argp_child mu_license_argp_child = {
  &mu_license_argp,
  0,
  NULL,
  0
};

struct mu_cmdline_capa mu_license_cmdline = {
  "license", &mu_license_argp_child 
};


/* ************************************************************************* */
/* Mailer                                                                    */
/* ************************************************************************* */

/* Options used by programs that send mail. */
static struct argp_option mu_mailer_argp_option[] = {
  {"mailer", 'M', N_("MAILER"), 0,
   N_("use specified URL as the default mailer"), 0},
  { NULL,      0, NULL, 0, NULL, 0 }
};

static error_t
mu_mailer_argp_parser (int key, char *arg, struct argp_state *state)
{
  static mu_list_t lst;

  switch (key)
    {
      /* mailer */
    case 'M':
      mu_argp_node_list_new (lst, "url", arg);
      break;

    case ARGP_KEY_INIT:
      mu_argp_node_list_init (&lst);
      break;

    case ARGP_KEY_FINI:
      mu_argp_node_list_finish (lst, "mailer", NULL);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

struct argp mu_mailer_argp = {
  mu_mailer_argp_option,
  mu_mailer_argp_parser,
};

struct argp_child mu_mailer_argp_child = {
  &mu_mailer_argp,
  0,
  NULL,
  0
};

struct mu_cmdline_capa mu_mailer_cmdline = {
  "mailer", &mu_mailer_argp_child
};


static struct argp_option mu_debug_argp_options[] = 
{
  { "debug-level", OPT_DEBUG_LEVEL, N_("LEVEL"), 0,
    N_("set Mailutils debugging level"), 0 },
  { "debug-line-info", OPT_LINE_INFO, NULL, 0,
    N_("show source info with debugging messages"), 0 },
  { NULL }
};

static error_t
mu_debug_argp_parser (int key, char *arg, struct argp_state *state)
{
  static mu_list_t lst;

  switch (key)
    {
    case OPT_DEBUG_LEVEL:
      mu_global_debug_from_string (arg, "command line");
      /*mu_argp_node_list_new (lst, "level", arg);*/
      break;

    case OPT_LINE_INFO:
      mu_argp_node_list_new (lst, "line-info", "yes");
      break;
      
    case ARGP_KEY_INIT:
      mu_argp_node_list_init (&lst);
      break;
      
    case ARGP_KEY_FINI:
      mu_argp_node_list_finish (lst, "debug", NULL);
      break;
      
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

struct argp mu_debug_argp = {
  mu_debug_argp_options,
  mu_debug_argp_parser,
};

struct argp_child mu_debug_argp_child = {
  &mu_debug_argp,
  0,
  N_("Global debugging settings"),
  0
};

struct mu_cmdline_capa mu_debug_cmdline = {
  "debug", &mu_debug_argp_child
};
