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

#include "pop3d.h"
#include <mailutils/argcv.h>
#include <xalloc.h>
#include "mailutils/libargp.h"

int db_list (char *input_name, char *output_name);
int db_make (char *input_name, char *output_name);

#define ACT_CREATE  0
#define ACT_ADD     1
#define ACT_DELETE  2
#define ACT_LIST    3
#define ACT_CHPASS  4

static int permissions = 0600;

struct action_data {
  int action;
  char *input_name;
  char *output_name;
  char *username;
  char *passwd;
};

void check_action(int action);
int action_create (struct action_data *ap);
int action_add (struct action_data *ap);
int action_delete (struct action_data *ap);
int action_list (struct action_data *ap);
int action_chpass (struct action_data *ap);

int (*ftab[]) (struct action_data *) = {
  action_create,
  action_add,
  action_delete,
  action_list,
  action_chpass
};

const char *program_version = "popauth (" PACKAGE_STRING ")";
static char doc[] = N_("GNU popauth -- manage pop3 authentication database");
static error_t popauth_parse_opt  (int key, char *arg,
				   struct argp_state *astate);

void popauth_version (FILE *stream, struct argp_state *state);
void (*argp_program_version_hook) (FILE *stream, struct argp_state *state) =
                                   popauth_version;

static struct argp_option options[] = 
{
  { NULL, 0, NULL, 0, N_("Actions are:"), 1 },
  { "add", 'a', 0, 0, N_("add user"), 1 },
  { "modify", 'm', 0, 0, N_("modify user's record (change password)"), 1 },
  { "delete", 'd', 0, 0, N_("delete user's record"), 1 },
  { "list", 'l', 0, 0, N_("list the contents of DBM file"), 1 },
  { "create", 'c', 0, 0, N_("create the DBM from a plaintext file"), 1 },

  { NULL, 0, NULL, 0,
    N_("Default action is:\n"
    "  For the file owner: --list\n"
    "  For a user: --modify --username <username>\n"), 2 },
  
  { NULL, 0, NULL, 0, N_("Options are:"), 3 },
  { "file", 'f', N_("FILE"), 0, N_("read input from FILE (default stdin)"), 3 },
  { "output", 'o', N_("FILE"), 0, N_("direct output to file"), 3 },
  { "password", 'p', N_("STRING"), 0, N_("specify user's password"), 3 },
  { "user", 'u', N_("USERNAME"), 0, N_("specify user name"), 3 },
  { "permissions", 'P', N_("PERM"), 0, N_("force given permissions on the database"), 3 },
  { NULL, }
};

static struct argp argp = {
  options,
  popauth_parse_opt,
  NULL,
  doc,
  NULL,
  NULL, NULL
};

static const char *popauth_argp_capa[] = {
  "common",
  "license",
  NULL
};

static void
set_db_perms (struct argp_state *astate, char *opt, int *pperm)
{
  int perm = 0;
   
  if (mu_isdigit(opt[0]))
    {
      char *p;
      perm = strtoul (opt, &p, 8);
      if (*p)
	{
	  argp_error (astate, _("invalid octal number: %s"), opt);
	  exit (EX_USAGE);
	}
    }
  *pperm = perm;
}

static error_t
popauth_parse_opt (int key, char *arg, struct argp_state *astate)
{
  struct action_data *ap = astate->input;
  switch (key)
    {
    case ARGP_KEY_INIT:
      memset (ap, 0, sizeof(*ap));
      ap->action = -1;
      break;
      
    case 'a':
      check_action (ap->action);
      ap->action = ACT_ADD;
      break;

    case 'c':
      check_action (ap->action);
      ap->action = ACT_CREATE;
      break;
      
    case 'l':
      check_action (ap->action);
      ap->action = ACT_LIST;
      break;
	
    case 'd':
      check_action (ap->action);
      ap->action = ACT_DELETE;
      break;
	  
    case 'p':
      ap->passwd = arg;
      break;
      
    case 'm':
      check_action (ap->action);
      ap->action = ACT_CHPASS;
      break;
	
    case 'f':
      ap->input_name = arg;
      break;
	  
    case 'o':
      ap->output_name = arg;
      break;
	
    case 'u':
      ap->username = arg;
      break;
	
    case 'P':
      set_db_perms (astate, arg, &permissions);
      break;
      
    case ARGP_KEY_FINI:
      if (ap->action == -1)
	{
	  /* Deduce the default action */
	  if (getuid () == 0)
	    ap->action = ACT_LIST;
	  else
	    ap->action = ACT_CHPASS;
	}
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

int
main(int argc, char **argv)
{
  struct action_data adata;

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mu_argp_init (program_version, NULL);
  if (mu_app_init (&argp, popauth_argp_capa, NULL,
		   argc, argv, 0, NULL, &adata))
    exit (EX_USAGE);

  return (*ftab[adata.action]) (&adata);
}

void
check_action (int action)
{
  if (action != -1)
    {
      mu_error (_("You may not specify more than one `-aldp' option"));
      exit (EX_USAGE);
    }
}

int
check_user_perm (int action, struct action_data *ap)
{
  struct stat sb;
  struct passwd *pw;
  uid_t uid;
  
  if (!ap->input_name)
    ap->input_name = APOP_PASSFILE;

  if (mu_dbm_stat (ap->input_name, &sb))
    {
      if (ap->action == ACT_ADD)
	{
	  DBM_FILE db;
	  if (mu_dbm_open (ap->input_name, &db, MU_STREAM_CREAT, permissions))
	    {
	      mu_diag_funcall (MU_DIAG_ERROR, "mu_dbm_open",
			       ap->input_name, errno);
	      exit (EX_SOFTWARE);
	    }
	  mu_dbm_close (db);
	  mu_dbm_stat (ap->input_name, &sb);
	}
      else
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "stat", ap->input_name, errno);
	  exit (EX_OSERR);
	}
    }

  uid = getuid ();
  if (uid == 0 || sb.st_uid == uid)
    return 0;

  if (ap->username)
    {
      mu_error (_("Only the file owner can use --username"));
      exit (EX_USAGE);
    }

  if (action != ACT_CHPASS)
    {
      mu_error (_("Operation not allowed"));
      exit (EX_USAGE);
    }
  pw = getpwuid (uid);
  if (!pw)
    exit (EX_OSERR);
  ap->username = pw->pw_name;
  return 1;
}

int
action_list (struct action_data *ap)
{
  FILE *fp;
  DBM_FILE db;
  DBM_DATUM key;
  DBM_DATUM contents;
  
  check_user_perm (ACT_LIST, ap);
  if (mu_dbm_open (ap->input_name, &db, MU_STREAM_READ, 0))
    {
      mu_error (_("cannot open file %s: %s"),
		ap->input_name, mu_strerror (errno));
      return 1;
    }
  
  if (ap->output_name)
    {
      fp = fopen (ap->output_name, "w");
      if (!fp)
	{
	  mu_error (_("cannot create file %s: %s"), ap->output_name, mu_strerror (errno));
	  return 1;
	}
    }
  else
    fp = stdout;

  if (ap->username)
    {
      memset (&key, 0, sizeof key);
      memset (&contents, 0, sizeof contents);
      MU_DATUM_PTR (key) = ap->username;
      MU_DATUM_SIZE (key) = strlen (ap->username);
      if (mu_dbm_fetch (db, key, &contents))
	{
	  mu_error (_("no such user: %s"), ap->username);
	}
      else
	{
	  fprintf (fp, "%.*s: %.*s\n",
		   (int) MU_DATUM_SIZE (key),
		   (char*) MU_DATUM_PTR (key),
		   (int) MU_DATUM_SIZE (contents),
		   (char*) MU_DATUM_PTR (contents));
	  mu_dbm_datum_free (&contents);
	}
    }
  else
    {
      for (key = mu_dbm_firstkey (db); MU_DATUM_PTR(key);
	   key = mu_dbm_nextkey (db, key))
	{
	  memset (&contents, 0, sizeof contents);
	  mu_dbm_fetch (db, key, &contents);
	  fprintf (fp, "%.*s: %.*s\n",
		   (int) MU_DATUM_SIZE (key),
		   (char*) MU_DATUM_PTR (key),
		   (int) MU_DATUM_SIZE (contents),
		   (char*) MU_DATUM_PTR (contents));
	  mu_dbm_datum_free (&contents);
	}
    }
  
  mu_dbm_close (db);
  fclose (fp);
  return 0;
}

int
action_create (struct action_data *ap)
{
  FILE *fp;
  DBM_FILE db;
  DBM_DATUM key;
  DBM_DATUM contents;
  char buf[256];
  int line = 0;

  /* Make sure we have proper privileges if popauth is setuid */
  setuid (getuid ());
  
  if (ap->input_name)
    {
      fp = fopen (ap->input_name, "r");
      if (!fp)
	{
	  mu_error (_("cannot open file %s: %s"),
		    ap->input_name, mu_strerror (errno));
	  return 1;
	}
    }
  else
    {
      ap->input_name = "";
      fp = stdin;
    }
  
  if (!ap->output_name)
    ap->output_name = APOP_PASSFILE;
  if (mu_dbm_open (ap->output_name, &db, MU_STREAM_CREAT, permissions))
    {
      mu_error (_("cannot create database %s: %s"), ap->output_name, mu_strerror (errno));
      return 1;
    }

  line = 0;
  while (fgets (buf, sizeof buf - 1, fp))
    {
      int len;
      int argc;
      char **argv;

      len = strlen (buf);
      if (buf[len-1] == '\n')
	buf[--len] = 0;
      
      line++;
      if (mu_argcv_get (buf, ":", NULL, &argc, &argv))
	{
	  mu_argcv_free (argc, argv);
	  continue;
	}

      if (argc == 0 || argv[0][0] == '#')
	{
	  mu_argcv_free (argc, argv);
	  continue;
	}
      
      if (argc != 3 || argv[1][0] != ':' || argv[1][1] != 0)
	{
	  mu_error (_("%s:%d: malformed line"), ap->input_name, line);
	  mu_argcv_free (argc, argv);
	  continue;
	}

      memset (&key, 0, sizeof key);
      memset (&contents, 0, sizeof contents);
      MU_DATUM_PTR (key) = argv[0];
      MU_DATUM_SIZE (key) = strlen (argv[0]);
      MU_DATUM_PTR (contents) = argv[2];
      MU_DATUM_SIZE (contents) = strlen (argv[2]);

      if (mu_dbm_insert (db, key, contents, 1))
	mu_error (_("%s:%d: cannot store datum"), ap->input_name, line);

      mu_argcv_free (argc, argv);
    }
  mu_dbm_close (db);
  fclose (fp);
  return 0;
}

int
open_io (int action, struct action_data *ap, DBM_FILE *db, int *not_owner)
{
  int rc = check_user_perm (action, ap);
  if (not_owner)
    *not_owner = rc;
  if (mu_dbm_open (ap->input_name, db, MU_STREAM_RDWR, permissions))
    {
      mu_error (_("cannot open file %s: %s"),
		ap->input_name, mu_strerror (errno));
      return 1;
    }
  return 0;
}

void
fill_pass (struct action_data *ap)
{
  if (!ap->passwd)
    {
      char *p;

      while (1) {
	if (ap->passwd)
	  free (ap->passwd);
	p = getpass (_("Password:"));
	if (!p)
	  exit (EX_DATAERR);
	ap->passwd = strdup (p);
	/* TRANSLATORS: Please try to format this string so that it has
	   the same length as the translation of 'Password:' above */
	p = getpass (_("Confirm :"));
	if (strcmp (ap->passwd, p) == 0)
	  break;
	mu_error (_("Passwords differ. Please retry."));
      } 
    }
}

int
action_add (struct action_data *ap)
{
  DBM_FILE db;
  DBM_DATUM key;
  DBM_DATUM contents;
  int rc;
  
  if (!ap->username)
    {
      mu_error (_("missing username to add"));
      return 1;
    }

  if (open_io (ACT_ADD, ap, &db, NULL))
    return 1;

  fill_pass (ap);
  
  memset (&key, 0, sizeof key);
  memset (&contents, 0, sizeof contents);
  MU_DATUM_PTR (key) = ap->username;
  MU_DATUM_SIZE (key) = strlen (ap->username);
  MU_DATUM_PTR (contents) = ap->passwd;
  MU_DATUM_SIZE (contents) = strlen (ap->passwd);

  rc = mu_dbm_insert (db, key, contents, 1);
  if (rc)
    mu_error (_("cannot store datum"));

  mu_dbm_close (db);
  return rc;
}

int
action_delete (struct action_data *ap)
{
  DBM_FILE db;
  DBM_DATUM key;
  int rc;
  
  if (!ap->username)
    {
      mu_error (_("missing username to delete"));
      return 1;
    }

  if (open_io (ACT_DELETE, ap, &db, NULL))
    return 1;
  
  MU_DATUM_PTR (key) = ap->username;
  MU_DATUM_SIZE (key) = strlen (ap->username);

  rc = mu_dbm_delete (db, key);
  if (rc)
    mu_error (_("cannot remove record for %s"), ap->username);

  mu_dbm_close (db);
  return rc;
}

int
action_chpass (struct action_data *ap)
{
  DBM_FILE db;
  DBM_DATUM key;
  DBM_DATUM contents;
  int rc;
  int not_owner;
  
  if (open_io (ACT_CHPASS, ap, &db, &not_owner))
    return 1;

  if (!ap->username)
    {
      mu_error (_("missing username"));
      return 1;
    }

  memset (&key, 0, sizeof key);
  memset (&contents, 0, sizeof contents);

  MU_DATUM_PTR (key) = ap->username;
  MU_DATUM_SIZE (key) = strlen (ap->username);
  if (mu_dbm_fetch (db, key, &contents))
    {
      mu_error (_("no such user: %s"), ap->username);
      return 1;
    }

  if (not_owner)
    {
      char *oldpass, *p;
      
      oldpass = xmalloc (MU_DATUM_SIZE (contents) + 1);
      memcpy (oldpass, MU_DATUM_PTR (contents), MU_DATUM_SIZE (contents));
      oldpass[MU_DATUM_SIZE (contents)] = 0;
      p = getpass (_("Old Password:"));
      if (!p)
	return 1;
      if (strcmp (oldpass, p))
	{
	  mu_error (_("Sorry"));
	  return 1;
	}
    }

  fill_pass (ap);
  
  mu_dbm_datum_free (&contents);
  MU_DATUM_PTR (contents) = ap->passwd;
  MU_DATUM_SIZE (contents) = strlen (ap->passwd);
  rc = mu_dbm_insert (db, key, contents, 1);
  if (rc)
    mu_error (_("cannot replace datum"));

  mu_dbm_close (db);
  return rc;
}

void
popauth_version (FILE *stream, struct argp_state *state)
{
#if defined(WITH_GDBM)
# define FORMAT "GDBM"
#elif defined(WITH_BDB)
# define FORMAT "Berkeley DB"
#elif defined(WITH_NDBM)
# define FORMAT "NDBM"
#elif defined(WITH_OLD_DBM)
# define FORMAT "Old DBM"
#elif defined(WITH_TOKYOCABINET)
# define FORMAT "Tokyo Cabinet"
#endif
  printf ("%s\n", argp_program_version);
  printf (_("Database format: %s\n"), FORMAT);
  printf (_("Database location: %s\n"), APOP_PASSFILE);
  exit (EX_OK);
}
