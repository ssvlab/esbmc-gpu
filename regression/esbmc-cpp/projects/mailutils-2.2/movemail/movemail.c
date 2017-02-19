/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2005, 2006, 2007, 2008, 2009, 2010 Free Software
   Foundation, Inc.

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

#if defined(HAVE_CONFIG_H)
# include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <pwd.h>
#include <grp.h>
#include <unistd.h>
#include <mailutils/mailutils.h>
#include <mailutils/tls.h>
#include <mu_asprintf.h>
#include "mailutils/libargp.h"
#include <muaux.h>

const char *program_version = "movemail (" PACKAGE_STRING ")";
static char doc[] = N_("GNU movemail -- move messages across mailboxes.");
static char args_doc[] = N_("inbox-url destfile [POP-password]");

enum {
  EMACS_OPTION=256,
  IGNORE_ERRORS_OPTION,
  PROGRAM_ID_OPTION
};

static struct argp_option options[] = {
  { "preserve", 'p', NULL, 0, N_("preserve the source mailbox") },
  { "keep-messages", 0, NULL, OPTION_ALIAS, NULL },
  { "reverse",  'r', NULL, 0, N_("reverse the sorting order") },
  { "emacs", EMACS_OPTION, NULL, 0,
    N_("output information used by Emacs rmail interface") },
  { "uidl", 'u', NULL, 0,
    N_("use UIDLs to avoid downloading the same message twice") },
  { "verbose", 'v', NULL, 0,
    N_("increase verbosity level") },
  { "owner", 'P', N_("MODELIST"), 0,
    N_("control mailbox ownership") },
  { "ignore-errors", IGNORE_ERRORS_OPTION, NULL, 0,
    N_("try to continue after errors") },
  { "program-id", PROGRAM_ID_OPTION, N_("FMT"), 0,
    N_("set program identifier for diagnostics (default: program name)") },
  { NULL,      0, NULL, 0, NULL, 0 }
};

static int reverse_order;
static int preserve_mail; 
static int emacs_mode;
static int uidl_option;
static int verbose_option;
static int ignore_errors;
static char *program_id_option;

enum set_ownership_mode
  {
    copy_owner_id,
    copy_owner_name,
    set_owner_id,
    set_owner_name
  };
#define SET_OWNERSHIP_MAX 4

struct user_id
{
  uid_t uid;
  gid_t gid;
};

struct set_ownership_method
{
  enum set_ownership_mode mode;
  union
  {
    char *name;
    struct user_id id;
  } owner;
};

static struct set_ownership_method so_methods[SET_OWNERSHIP_MAX];
static int so_method_num;

struct set_ownership_method *
get_next_so_method ()
{
  if (so_method_num == MU_ARRAY_SIZE (so_methods))
    {
      mu_error (_("ownership method table overflow"));
      exit (1);
    }
  return so_methods + so_method_num++;
}

mu_kwd_t method_kwd[] = {
  { "copy-id", copy_owner_id }, 
  { "copy-name", copy_owner_name },
  { "set-name", set_owner_name },
  { "user", set_owner_name },
  { "set-id", set_owner_id },
  { NULL }
};

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  static mu_list_t lst;

  switch (key)
    {
    case 'r':
      mu_argp_node_list_new (lst, "reverse", "yes");
      break;

    case 'p':
      mu_argp_node_list_new (lst, "preserve", "yes");
      break;

    case 'P':
      mu_argp_node_list_new (lst, "mailbox-ownership", arg);
      break;

    case 'u':
      mu_argp_node_list_new (lst, "uidl", "yes");
      break;

    case 'v':
      verbose_option++;
      break;
      
    case EMACS_OPTION:
      mu_argp_node_list_new (lst, "emacs", "yes");
      break;

    case IGNORE_ERRORS_OPTION:
      mu_argp_node_list_new (lst, "ignore-errors", "yes");
      break;

    case PROGRAM_ID_OPTION:
      mu_argp_node_list_new (lst, "program-id", arg);
      break;
      
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

static struct argp argp = {
  options,
  parse_opt,
  args_doc,
  doc,
  NULL,
  NULL, NULL
};


static int
_cb_mailbox_ownership (mu_debug_t debug, const char *str)
{
  if (strcmp (str, "clear") == 0)
    so_method_num = 0;
  else
    {
      int code;
      char *p;
      size_t len = strcspn (str, "=");
      struct set_ownership_method *meth;
	  
      if (mu_kwd_xlat_name_len (method_kwd, str, len, &code))
	{
	  mu_cfg_format_error (debug, MU_DEBUG_ERROR, 
			       _("invalid ownership method: %s"),
			       str);
	  return 1;
	}
      
      meth = get_next_so_method ();
      meth->mode = code;
      switch (meth->mode)
	{
	case copy_owner_id:
	case copy_owner_name:
	  break;
	  
	case set_owner_id:
	  if (!str[len])
	    {
	      mu_cfg_format_error (debug, MU_DEBUG_ERROR, 
				   _("ownership method %s requires value"),
				   str);
	      return 1;
	    }
	  str += len + 1;
	  meth->owner.id.uid = strtoul (str, &p, 0);
	  if (*p)
	    {
	      if (*p == ':')
		{
		  str = p + 1;
		  meth->owner.id.gid = strtoul (str, &p, 0);
		  if (*p)
		    {
		      mu_cfg_format_error (debug, MU_DEBUG_ERROR, 
					   _("expected gid number, but found %s"),
					   str);
		      return 1;
		    }
		}
	      else
		{
		  mu_cfg_format_error (debug, MU_DEBUG_ERROR, 
				       _("expected uid number, but found %s"),
				       str);
		  return 1;
		}
	    }
	  else
	    meth->owner.id.gid = (gid_t) -1;
	  break;
	  
	case set_owner_name:
	  if (!str[len])
	    {
	      mu_cfg_format_error (debug, MU_DEBUG_ERROR, 
				   _("ownership method %s requires value"),
				   str);
	      return 1;
	    }
	  meth->owner.name = mu_strdup (str + len + 1);
	}
    }
  return 0;
}

static int
cb_mailbox_ownership (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  int i;
  
  if (val->type == MU_CFG_STRING)
    {
      const char *str = val->v.string;
      if (!strchr (str, ','))
	return _cb_mailbox_ownership (debug, str);
      else
	{
	  int argc;
	  char **argv;

	  if (mu_argcv_get_np (str, strlen (str), ",", NULL, 0,
			       &argc, &argv, NULL))
	    {
	      mu_cfg_format_error (debug, MU_DEBUG_ERROR, 
				   _("cannot parse %s"),
				   str);
	      return 1;
	    }

	  for (i = 0; i < argc; i++)
	    if (_cb_mailbox_ownership (debug, argv[i]))
	      return 1;

	  mu_argcv_free (argc, argv);
	  return 0;
	}
    }
		
  if (mu_cfg_assert_value_type (val, MU_CFG_LIST, debug))
    return 1;

  for (i = 0; i < val->v.arg.c; i++)
    {
      if (mu_cfg_assert_value_type (&val->v.arg.v[i], MU_CFG_STRING, debug))
	return 1;
      if (_cb_mailbox_ownership (debug, val->v.arg.v[i].v.string))
	return 1;
    }
  return 0;
}

struct mu_cfg_param movemail_cfg_param[] = {
  { "preserve", mu_cfg_bool, &preserve_mail, 0, NULL,
    N_("Do not remove messages from the source mailbox.") },
  { "reverse",  mu_cfg_bool, &reverse_order, 0, NULL,
    N_("Reverse message sorting order.") },
  { "emacs", mu_cfg_bool, &emacs_mode, 0, NULL,
    N_("Output information used by Emacs rmail interface.") },
  { "uidl", mu_cfg_bool, &uidl_option, 0, NULL,
    N_("Use UIDLs to avoid downloading the same message twice.") },
  { "verbose", mu_cfg_int, &verbose_option, 0, NULL,
    N_("Set verbosity level.") },
  { "ignore-errors", mu_cfg_bool, &ignore_errors, 0, NULL,
    N_("Continue after an error.") },
  { "program-id", mu_cfg_string, &program_id_option, 0, NULL,
    N_("Set program identifier string (default: program name)") },
  { "mailbox-ownership", mu_cfg_callback, NULL, 0,
    cb_mailbox_ownership,
    N_("Define a list of methods for setting mailbox ownership. Valid "
       "methods are:\n"
       " copy-id          get owner UID and GID from the source mailbox\n"
       " copy-name        get owner name from the source mailbox URL\n"
       " set-id=UID[:GID] set supplied UID and GID\n"
       " set-name=USER    make destination mailbox owned by USER"),
    N_("methods: list") },
  { NULL }
};


static const char *movemail_capa[] = {
  "common",
  "debug",
  "license",
  "locking",
  "mailbox",
  "auth",
  NULL 
};

void
die (mu_mailbox_t mbox, const char *msg, int status)
{
  mu_url_t url = NULL;
  
  mu_mailbox_get_url (mbox, &url);
  if (emacs_mode)
    mu_error (_("%s:mailbox `%s': %s: %s"),
	      mu_errname (status),
	      mu_url_to_string (url),
	      msg,
	      mu_strerror (status));
  else
    mu_error (_("mailbox `%s': %s: %s"),
	      mu_url_to_string (url), msg, mu_strerror (status));
  exit (1);
}

void
lock_mailbox (mu_mailbox_t mbox)
{
  mu_locker_t lock;
  int status;
  
  status = mu_mailbox_get_locker (mbox, &lock);
  if (status)
    die (mbox, _("cannot retrieve locker"), status);
      
  if (!lock)
    /* Remote mailboxes have no lockers */
    return;

  status = mu_locker_lock (lock);

  if (status)
    die (mbox, _("cannot lock"), status);
}


void
attach_passwd_ticket (mu_mailbox_t mbx, char *passwd)
{
  mu_folder_t folder = NULL;
  mu_authority_t auth = NULL;
  mu_secret_t secret;
  mu_ticket_t t;
  int rc;

  rc = mu_secret_create (&secret, passwd, strlen (passwd));
  if (rc)
    {
      mu_error ("mu_secret_create: %s", mu_strerror (rc));
      exit (1);
    }
  
  mu_ticket_create (&t, NULL);
  mu_ticket_set_secret (t, secret);

  if ((rc = mu_mailbox_get_folder (mbx, &folder)))
    die (mbx, _("mu_mailbox_get_folder failed"), rc);

  if ((rc = mu_folder_get_authority (folder, &auth)))
    die (mbx, _("mu_folder_get_authority failed"), rc);

  if (auth && (rc = mu_authority_set_ticket (auth, t)))
    die (mbx, _("mu_authority_set_ticket failed"), rc);
}


/* Create and open a mailbox associated with the given URL,
   flags and (optionally) password */
void
open_mailbox (mu_mailbox_t *mbx, char *name, int flags, char *passwd)
{
  int status = mu_mailbox_create_default (mbx, name);

  if (status)
    {
      if (name)
	mu_error (_("could not create mailbox `%s': %s"),
		  name,
		  mu_strerror (status));
      else
	mu_error (_("could not create default mailbox: %s"),
		  mu_strerror (status));
      exit (1);
    }

  if (passwd)
    attach_passwd_ticket (*mbx, passwd);
  status = mu_mailbox_open (*mbx, flags);
  if (status)
    die (*mbx, _("cannot open"), status);
  lock_mailbox (*mbx);
}

int
move_message (mu_mailbox_t src, mu_mailbox_t dst, size_t msgno)
{
  int rc;
  mu_message_t msg;

  if ((rc = mu_mailbox_get_message (src, msgno, &msg)) != 0)
    {
      mu_error (_("cannot read message %lu: %s"),
		(unsigned long) msgno, mu_strerror (rc));
      return rc;
    }
  if ((rc = mu_mailbox_append_message (dst, msg)) != 0)
    {
      mu_error (_("cannot append message %lu: %s"),
		(unsigned long) msgno, mu_strerror (rc));
      return rc;
    }
  if (!preserve_mail)
    {
      mu_attribute_t attr;
      mu_message_get_attribute (msg, &attr);
      mu_attribute_set_deleted (attr);
    }
  return rc;
}

/* Open source mailbox using compatibility syntax. Source_name is
   of the form:

     po:USERNAME[:POP-SERVER]

   if POP-SERVER part is omitted, the MAILHOST environment variable
   will be consulted. */
void
compatibility_mode (mu_mailbox_t *mbx, char *source_name, char *password,
		    int flags)
{
  char *tmp;
  char *user_name = strtok (source_name+3, ":");
  char *host = strtok (NULL, ":");
  if (!host)
    host = getenv ("MAILHOST");
  if (!host)
    {
      mu_error (_("hostname of the POP3 server is unknown"));
      exit (1);
    }
  asprintf (&tmp, "pop://%s@%s", user_name, host);
  open_mailbox (mbx, tmp, flags, password);
  free (tmp);
}

static mu_mailbox_t source, dest;

static void
close_mailboxes (void)
{
  mu_mailbox_close (dest);
  mu_mailbox_close (source);
}  

static int
get_mbox_owner_id (mu_mailbox_t mbox, mu_url_t url, struct user_id *id)
{
  const char *s;
  int rc = mu_url_sget_scheme  (url, &s);
  if (rc)
    die (mbox, _("cannot get scheme"), rc);
  if ((strcmp (s, "/") == 0
       || strcmp (s, "mbox") == 0
       || strcmp (s, "mh") == 0
       || strcmp (s, "maildir") == 0))
    {
      struct stat st;
      
      rc = mu_url_sget_path  (url, &s);
      if (rc)
	die (mbox, _("cannot get path"), rc);
      if (stat (s, &st))
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "stat", s, errno);
	  exit (1);
	}
      id->uid = st.st_uid;
      id->gid = st.st_gid;
      return 0;
    }
  else if (verbose_option)
    mu_diag_output (MU_DIAG_WARNING,
		    _("ignoring copy-name: not a local mailbox"));
  return 1;
}

static int
get_user_id (const char *name, struct user_id *id)
{
  struct mu_auth_data *auth = mu_get_auth_by_name (name);
  
  if (!auth)
    {
      if (verbose_option)
	mu_diag_output (MU_DIAG_WARNING, _("no such user: %s"), name);
      return 1;
    }

  id->uid = auth->uid;
  id->gid = auth->gid;
  mu_auth_data_free (auth);
  return 0;
}  

static int
get_mbox_owner_name (mu_mailbox_t mbox, mu_url_t url, struct user_id *id)
{
  const char *s;
  int rc = mu_url_sget_user (url, &s);
  if (rc)
    /* FIXME */
    die (mbox, _("cannot get mailbox owner name"), rc);

  return get_user_id (s, id);
}

static int
guess_mbox_owner (mu_mailbox_t mbox, struct user_id *id)
{
  mu_url_t url = NULL;
  int rc;
  struct set_ownership_method *meth;
  
  rc = mu_mailbox_get_url (mbox, &url);
  if (rc)
    {
      mu_diag_funcall (MU_DIAG_ERROR, "mu_mailbox_get_url", NULL, rc);
      exit (1);
    }

  rc = 1;
  for (meth = so_methods; rc == 1 && meth < so_methods + so_method_num; meth++)
    {
      switch (meth->mode)
	{
	case copy_owner_id:
	  rc = get_mbox_owner_id (mbox, url, id);
	  break;
	  
	case copy_owner_name:
	  rc = get_mbox_owner_name (mbox, url, id);
	  break;
	  
	case set_owner_id:
	  id->uid = meth->owner.id.uid;
	  rc = 0;
	  if (meth->owner.id.gid == (gid_t)-1)
	    {
	      struct passwd *pw = getpwuid (id->uid);
	      if (pw)
		id->gid = pw->pw_gid;
	      else
		{
		  if (verbose_option)
		    mu_diag_output (MU_DIAG_WARNING,
				    _("no user with uid %lu found"),
				    (unsigned long) id->uid);
		  rc = 1;
		}
	    }
	  break;
	  
	case set_owner_name:
	  rc = get_user_id (meth->owner.name, id);
	  break;
	}
    }
  
  return rc;
}

static void
switch_owner (mu_mailbox_t mbox)
{
  struct user_id user_id;

  if (so_method_num == 0)
    return;

  if (getuid ())
    {
      if (verbose_option)
	mu_diag_output (MU_DIAG_WARNING,
			_("ignoring mailbox-ownership statement"));
      return;
    }
  
  if (guess_mbox_owner (mbox, &user_id) == 0)
    {
      if (mu_switch_to_privs (user_id.uid, user_id.gid, NULL))
	exit (1);
    }
  else
    {
      mu_error (_("no suitable method for setting mailbox ownership"));
      exit (1);
    }
}

static int
_compare_uidls (const void *item, const void *value)
{
  const struct mu_uidl *a = item;
  const struct mu_uidl *b = value;

  return strcmp (a->uidl, b->uidl);
}

#define __cat2__(a,b) a ## b
#define DCL_VTX(what)						\
static int							\
 __cat2__(_vtx_,what) (const char *name, void *data, char **p)	\
{								\
  mu_url_t url = data;			                        \
  int rc = __cat2__(mu_url_aget_,what) (url, p);                \
  if (rc == MU_ERR_NOENT)	                                \
    {                                                           \
      *p = strdup ("");						\
      return 0;                                                 \
    }                                                           \
  return rc;                                                    \
}

DCL_VTX (host)
DCL_VTX (user)
DCL_VTX (path)

static void
set_program_id (const char *source_name, const char *dest_name)
{
  int rc;
  mu_vartab_t vtab;
  char *id;
  mu_url_t url;
      
  mu_vartab_create (&vtab);
  mu_vartab_define (vtab, "progname", mu_program_name, 1);
  mu_vartab_define (vtab, "source", source_name, 1);
  rc = mu_mailbox_get_url (source, &url);
  if (rc)
    mu_diag_output (MU_DIAG_INFO,
		    _("cannot obtain source mailbox URL: %s"),
		    mu_strerror (rc));
  else
    {
      mu_vartab_define_exp (vtab, "source:user", _vtx_user, NULL, url);
      mu_vartab_define_exp (vtab, "source:host", _vtx_host, NULL, url);
      mu_vartab_define_exp (vtab, "source:path", _vtx_path, NULL, url);
    }
      
  mu_vartab_define (vtab, "dest", dest_name, 1);
  rc = mu_mailbox_get_url (dest, &url);
  if (rc)
    mu_diag_output (MU_DIAG_INFO,
		    _("cannot obtain destination mailbox URL: %s"),
		    mu_strerror (rc));
  else
    {
      mu_vartab_define_exp (vtab, "dest:user", _vtx_user, NULL, url);
      mu_vartab_define_exp (vtab, "dest:host", _vtx_host, NULL, url);
      mu_vartab_define_exp (vtab, "dest:path", _vtx_path, NULL, url);
    }
      
  rc = mu_vartab_expand (vtab, program_id_option, &id);
  mu_vartab_destroy (&vtab);
  /*      asprintf (&id, "%s: %s", mu_program_name, s);
	  free (s);*/
  /* FIXME: Don't use mu_set_program_name here, because it
     plays wise with its argument. We need a mu_set_diag_prefix
     function. */
  mu_program_name = id;
}

int
main (int argc, char **argv)
{
  int index;
  size_t i, total;
  int rc = 0;
  int errs = 0;
  char *source_name, *dest_name;
  int flags;
  mu_list_t src_uidl_list = NULL;
  size_t msg_count = 0;
  
  /* Native Language Support */
  MU_APP_INIT_NLS ();
  MU_AUTH_REGISTER_ALL_MODULES ();
  
  /* Register the desired mailbox formats.  */
  mu_register_all_mbox_formats ();

  /* argument parsing */
  
#ifdef WITH_TLS
  mu_gocs_register ("tls", mu_tls_module_init);
#endif
  mu_argp_init (program_version, NULL);
  if (mu_app_init (&argp, movemail_capa, movemail_cfg_param, 
		   argc, argv, 0, &index, NULL))
    exit (1);

  argc -= index;
  argv += index;

  if (argc < 2 || argc > 3)
    {
      mu_error (_("wrong number of arguments"));
      return 1;
    }

  atexit (close_mailboxes);
  
  source_name = argv[0];
  dest_name = argv[1];

  flags = preserve_mail ? MU_STREAM_READ : MU_STREAM_RDWR;
  
  if (strncmp (source_name, "po:", 3) == 0)
    compatibility_mode (&source, source_name, argv[2], flags);
  else
    open_mailbox (&source, source_name, flags, argv[2]);

  switch_owner (source);
  
  open_mailbox (&dest, dest_name, MU_STREAM_RDWR | MU_STREAM_CREAT, NULL);

  if (program_id_option)
    set_program_id (source_name, dest_name);

  rc = mu_mailbox_messages_count (source, &total);
  if (rc)
    {
      mu_error(_("cannot count messages: %s"), mu_strerror (rc));
      exit (1);
    }
  
  if (verbose_option)
    mu_diag_output (MU_DIAG_INFO,
		    _("number of messages in source mailbox: %lu"),
		    (unsigned long) total);

  if (uidl_option)
    {
      mu_iterator_t itr;
      mu_list_t dst_uidl_list = NULL;

      rc = mu_mailbox_get_uidls (source, &src_uidl_list);
      if (rc)
	die (source, _("cannot get UIDLs"), rc);
      rc = mu_mailbox_get_uidls (dest, &dst_uidl_list);
      if (rc)
	die (dest, _("cannot get UIDLs"), rc);

      mu_list_set_comparator (dst_uidl_list, _compare_uidls);
      
      mu_list_get_iterator (src_uidl_list, &itr);
      for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
	   mu_iterator_next (itr))
	{
	  struct mu_uidl *uidl;
	      
	  mu_iterator_current (itr, (void **)&uidl);
	  if (mu_list_locate (dst_uidl_list, uidl, NULL) == 0)
	    mu_iterator_ctl (itr, mu_itrctl_delete, NULL);
	}
      mu_iterator_destroy (&itr);
      mu_list_destroy (&dst_uidl_list);
      mu_list_set_comparator (src_uidl_list, NULL);
    }

  /* FIXME: Implementing a mailbox iterator would allow to merge the three
     branches of this conditional. */
  if (src_uidl_list)
    {
      mu_iterator_t itr;

      rc = mu_list_get_iterator (src_uidl_list, &itr);
      if (rc)
	{
	  mu_error(_("cannot get iterator: %s"), mu_strerror (rc));
	  exit (1);
	}
      rc = mu_iterator_ctl (itr, mu_itrctl_set_direction, &reverse_order);
      if (rc)
	{
	  mu_error(_("cannot set iteration direction: %s"), mu_strerror (rc));
	  exit (1);
	}
      for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
	   mu_iterator_next (itr))
	{
	  struct mu_uidl *uidl;
	      
	  mu_iterator_current (itr, (void **)&uidl);
	  rc = move_message (source, dest, uidl->msgno);
	  if (rc == 0)
	    msg_count++;
	  else if (!ignore_errors)
	    break;
	  else
	    errs = 1;
	}
      mu_iterator_destroy (&itr);
    }
  else if (reverse_order)
    {
      for (i = total; i > 0; i--)
	{
	  rc = move_message (source, dest, i);
	  if (rc == 0)
	    msg_count++;
	  else if (!ignore_errors)
	    break;
	  else
	    errs = 1;
	}
    }
  else
    {
      for (i = 1; i <= total; i++)
	{
	  rc = move_message (source, dest, i);
	  if (rc == 0)
	    msg_count++;
	  else if (!ignore_errors)
	    break;
	  else
	    errs = 1;
	}
    }
  
  if (verbose_option)
    mu_diag_output (MU_DIAG_INFO,
		    _("number of processed messages: %lu"),
		    (unsigned long) msg_count);
  
  if (rc)
    return !!rc;
  
  mu_mailbox_sync (dest);
  rc = mu_mailbox_close (dest);
  mu_mailbox_destroy (&dest);
  if (rc)
    mu_error (_("cannot close destination mailbox: %s"), mu_strerror (rc));
  else
    mu_mailbox_flush (source, 1);

  mu_mailbox_close (source);
  mu_mailbox_destroy (&source);

  return !(rc == 0 && errs == 0);
}
