/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2008, 2009, 2010
   Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#ifdef __EXT_QNX
# undef __EXT_QNX
#endif
#include <unistd.h>

#include <mailutils/errno.h>
#include <mailutils/locker.h>
#include <mailutils/nls.h>

#include "mailutils/libargp.h"

const char *program_version = "GNU dotlock (" PACKAGE_STRING ")";
static char doc[] =
N_("GNU dotlock -- lock mail spool files.")
"\v"
N_("Returns 0 on success, 3 if locking the file fails because\
 it's already locked, and 1 if some other kind of error occurred.");

static char args_doc[] = N_("FILE");

static struct argp_option options[] = {
  {"unlock", 'u', NULL, 0,
   N_("unlock"), 0},

  {"force", 'f', N_("MINUTES"), OPTION_ARG_OPTIONAL,
   N_("forcibly break an existing lock older than a certain time"), 0},

  {"retry", 'r', N_("RETRIES"), OPTION_ARG_OPTIONAL,
   N_("retry the lock a few times"), 0},

  {"debug", 'd', NULL, 0,
   N_("print details of failure reasons to stderr"), 0},

  {NULL, 0, NULL, 0, NULL, 0}
};

static error_t parse_opt (int key, char *arg, struct argp_state *state);

static struct argp argp = {
  options,
  parse_opt,
  args_doc,
  doc,
};

static const char *file;
static int unlock;
static int flags;
static int retries;
static time_t force;
static int debug;

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  static mu_list_t lst;

  switch (key)
    {
    case 'd':
      mu_argp_node_list_new (lst, "debug", "yes");
      break;

    case 'u':
      unlock = 1;
      break;

    case 'r':
      if (arg)
	mu_argp_node_list_new (lst, "retry", arg);
      break;

    case 'f':
      mu_argp_node_list_new (lst, "force", arg ? arg : "0");
      break;

    case ARGP_KEY_ARG:
      if (file)
	argp_error (state, _("only one FILE can be specified"));
      file = arg;
      break;

    case ARGP_KEY_NO_ARGS:
      if (!mu_help_config_mode)
	argp_error (state, _("FILE must be specified"));
      return ARGP_ERR_UNKNOWN;
      
    case ARGP_KEY_INIT:
      mu_argp_node_list_init (&lst);
      break;
      
    case ARGP_KEY_FINI:
      mu_argp_node_list_finish (lst, NULL, NULL);
      break;
      
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}


struct mu_cfg_param dotlock_cfg_param[] = {
  { "force", mu_cfg_time, &force, 0, NULL,
    N_("Forcibly break an existing lock older than the specified time.") },
  { "retry", mu_cfg_int, &retries, 0, NULL,
    N_("Number of times to retry acquiring the lock.") },
  { "debug", mu_cfg_bool, &debug, 0, NULL,
    N_("Print details of failure reasons to stderr.") },
  { NULL }
};



const char *dotlock_capa[] = {
  "license",
  "common",
  "debug",
  NULL
};

int
main (int argc, char *argv[])
{
  mu_locker_t locker = 0;
  int err = 0;
  pid_t usergid = getgid ();
  pid_t mailgid = getegid ();

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  /* Drop permissions during argument parsing. */

  if (setegid (usergid) < 0)
    return MU_DL_EX_ERROR;

  argp_err_exit_status = MU_DL_EX_ERROR;
  
  mu_argp_init (program_version, NULL);
  if (mu_app_init (&argp, dotlock_capa, dotlock_cfg_param, 
		   argc, argv, 0, NULL, NULL))
    exit (1);

  if (force)
    {
      force *= 60;
      flags |= MU_LOCKER_TIME;
    }

  if (retries != 0)
    flags |= MU_LOCKER_RETRY;
  
  if ((err = mu_locker_create (&locker, file, flags)))
    {
      if (debug)
	mu_diag_funcall (MU_DIAG_ERROR, "mu_locker_create", NULL, err);
      return MU_DL_EX_ERROR;
    }

  if (force != 0)
    mu_locker_set_expire_time (locker, force);

  if (retries != 0)
    mu_locker_set_retries (locker, retries);

  if (setegid (mailgid) < 0)
    return MU_DL_EX_ERROR;

  if (unlock)
    err = mu_locker_remove_lock (locker);
  else
    err = mu_locker_lock (locker);

  setegid(usergid);

  mu_locker_destroy (&locker);

  if (debug && err)
    mu_error (unlock ? _("unlocking the file %s failed: %s") :
	      _("locking the file %s failed: %s"),
	      file, mu_strerror (err));

  switch (err)
    {
    case 0:
      err = MU_DL_EX_OK;
      break;
    case EPERM:
      err = MU_DL_EX_PERM;
      break;
    case MU_ERR_LOCK_NOT_HELD:
      err = MU_DL_EX_NEXIST;
      break;
    case MU_ERR_LOCK_CONFLICT:
      err = MU_DL_EX_EXIST;
      break;
    default:
      err = MU_DL_EX_ERROR;
      break;
    }

  return err;
}

