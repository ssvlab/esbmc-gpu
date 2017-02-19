/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2006, 2007, 2010 Free Software Foundation, Inc.

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
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <ctype.h>
#include <string.h>
#include <mailutils/mailutils.h>
#include "mailutils/libargp.h"

const char *program_version = "muauth (" PACKAGE_STRING ")";
static char doc[] =
"muauth -- test mailutils authentication and authorization schemes";
static char args_doc[] = "key";

static const char *capa[] = {
  "auth",
  "license",
  "common",
  "debug",
  NULL
};

static struct argp_option options[] = {
  { "password", 'p', "STRING", 0, "user password", 0 },
  { "uid", 'u', NULL, 0, "test getpwuid functions", 0 },
  { "name", 'n', NULL, 0, "test getpwnam functions", 0 },
  { NULL },
};

enum mu_auth_key_type key_type = mu_auth_key_name;
char *password;

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case 'p':
      password = arg;
      break;

    case 'u':
      key_type = mu_auth_key_uid;
      break;

    case 'n':
      key_type = mu_auth_key_name;
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp argp = {
  options,
  parse_opt,
  args_doc,
  doc,
  NULL,
  NULL, NULL
};
           
int
main (int argc, char * argv [])
{
  int rc, index;
  struct mu_auth_data *auth;
  void *key;
  uid_t uid;
  
  MU_AUTH_REGISTER_ALL_MODULES ();
  mu_argp_init (program_version, NULL);
  if (mu_app_init (&argp, capa, NULL, argc, argv, 0, &index, NULL))
    exit (1);

  if (index == argc)
    {
      mu_error ("not enough arguments, try `%s --help' for more info",
		argv[0]);
      return 1;
    }

  if (key_type == mu_auth_key_uid)
    {
      uid = strtoul (argv[index], NULL, 0);
      key = &uid;
    }
  else
    key = argv[index];
  
  rc = mu_get_auth (&auth, key_type, key);
  printf ("mu_get_auth => %d, %s\n", rc, mu_strerror (rc));
  if (rc == 0)
    {
      printf ("user name:  %s\n", auth->name);
      printf ("password:   %s\n", auth->passwd);
      printf ("uid:        %lu\n", (unsigned long) auth->uid);
      printf ("gid:        %lu\n", (unsigned long) auth->gid);
      printf ("gecos:      %s\n", auth->gecos);
      printf ("home:       %s\n", auth->dir);
      printf ("shell:      %s\n", auth->shell);
      printf ("mailbox:    %s\n", auth->mailbox);
      printf ("change_uid: %d\n", auth->change_uid);
	
      rc = mu_authenticate (auth, password);
      printf ("mu_authenticate => %d, %s\n", rc, mu_strerror (rc));
      mu_auth_data_free (auth);
    }
  return rc != 0;
}
      
  
