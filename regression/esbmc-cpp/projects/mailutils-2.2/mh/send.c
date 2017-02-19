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

/* MH send command */

#include <mh.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>
#include <pwd.h>

const char *program_version = "send (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH send")"\v"
N_("Options marked with `*' are not yet implemented.\n\
Use -help to obtain the list of traditional MH options.");
static char args_doc[] = N_("file [file...]");

/* GNU options */
static struct argp_option options[] = {
  {"alias",         ARG_ALIAS,         N_("FILE"), 0,
   N_("specify additional alias file") },
  {"draft",         ARG_DRAFT,         NULL, 0,
   N_("use prepared draft") },
  {"draftfolder",   ARG_DRAFTFOLDER,   N_("FOLDER"), 0,
   N_("specify the folder for message drafts") },
  {"draftmessage",  ARG_DRAFTMESSAGE,  NULL, 0,
   N_("treat the arguments as a list of messages from the draftfolder") },
  {"nodraftfolder", ARG_NODRAFTFOLDER, NULL, 0,
   N_("undo the effect of the last --draftfolder option") },
  {"filter",        ARG_FILTER,        N_("FILE"), 0,
  N_("* use filter FILE to preprocess the body of the message") },
  {"nofilter",      ARG_NOFILTER,      NULL, 0,
   N_("* undo the effect of the last --filter option") },
  {"format",        ARG_FORMAT,        N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("* reformat To: and Cc: addresses") },
  {"noformat",      ARG_NOFORMAT,      NULL, OPTION_HIDDEN, "" },
  {"forward",       ARG_FORWARD,       N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("* in case of failure forward the draft along with the failure notice to the sender") },
  {"noforward",     ARG_NOFORWARD,     NULL, OPTION_HIDDEN, "" },
  {"mime",          ARG_MIME,          N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("* use MIME encapsulation") },
  {"nomime",        ARG_NOMIME,        NULL, OPTION_HIDDEN, "" },
  {"msgid",         ARG_MSGID,         N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("add Message-ID: field") },
  {"nomsgid",       ARG_NOMSGID,       NULL, OPTION_HIDDEN, ""},
  {"push",          ARG_PUSH,          N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("run in the backround") },
  {"nopush",        ARG_NOPUSH,        NULL, OPTION_HIDDEN, "" },
  {"preserve",      ARG_PRESERVE,      N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("keep draft files") },
  {"keep",          0, NULL, OPTION_ALIAS, NULL},
  {"split",         ARG_SPLIT,         N_("SECONDS"), 0,
   N_("split the draft into several partial messages and send them with SECONDS interval") },
  {"chunksize",     ARG_CHUNKSIZE,     N_("NUMBER"), 0,
   N_("set the size of chunk for --split (in bytes)") },
  {"verbose",       ARG_VERBOSE,       N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("print the transcript of interactions with the transport system") },
  {"noverbose",     ARG_NOVERBOSE,     NULL, OPTION_HIDDEN, "" },
  {"watch",         ARG_WATCH,         N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("monitor the delivery of mail") },
  {"nowatch",       ARG_NOWATCH,       NULL, OPTION_HIDDEN, "" },
  {"width",         ARG_WIDTH,         N_("NUMBER"), 0,
   N_("* make header fields no longer than NUMBER columns") },
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  { 0 }
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"alias",         1, 0, "aliasfile" },
  {"draft",         5, 0, NULL },
  {"draftfolder",   6, 0, "folder" },
  {"draftmessage",  6, 0, "message"},
  {"nodraftfolder", 3, 0, NULL },
  {"filter",        2, 0, "filterfile"},
  {"nofilter",      3, 0, NULL },
  {"format",        4, MH_OPT_BOOL, NULL},
  {"forward",       4, MH_OPT_BOOL, NULL},
  {"mime",          2, MH_OPT_BOOL, NULL},
  {"msgid",         2, MH_OPT_BOOL, NULL},
  {"push",          2, MH_OPT_BOOL, NULL},
  {"preserve",      2, MH_OPT_BOOL, NULL},
  {"keep",          1, MH_OPT_BOOL, NULL},
  {"split",         1, 0, "seconds"},
  {"verbose",       1, MH_OPT_BOOL, NULL},
  {"watch",         2, MH_OPT_BOOL, NULL},
  {"width",         2, 0, NULL },
  { 0 }
};

static int use_draft;            /* Use the prepared draft */
static const char *draft_folder; /* Use this draft folder */
static int reformat_recipients;  /* --format option */
static int forward_notice;       /* Forward the failure notice to the sender,
				    --forward flag */
static int mime_encaps;          /* Use MIME encapsulation */
static int append_msgid;         /* Append Message-ID: header */
static int background;           /* Operate in the background */

static int split_message;            /* Split the message */
static unsigned long split_interval; /* Interval in seconds between sending two
					successive partial messages */
static size_t split_size = 76*632;   /* Size of split parts */
static int verbose;              /* Produce verbose diagnostics */
static int watch;                /* Watch the delivery process */
static unsigned width = 76;      /* Maximum width of header fields */

static int keep_files;           /* Keep draft files */

#define DEFAULT_X_MAILER "MH (" PACKAGE_STRING ")"

#define WATCH(c) do {\
  if (watch)\
    watch_printf c;\
} while (0)

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  char *p;
  
  switch (key)
    {
    case ARG_ALIAS:
      mh_alias_read (arg, 1);
      break;

    case ARG_CHUNKSIZE:
      split_size = strtoul (arg, &p, 10);
      if (*p)
	{
	  argp_error (state, "%s: %s", arg, _("invalid number"));
	  exit (1);
	}
      break;
	
    case ARG_DRAFT:
      use_draft = 1;
      break;
	
    case ARG_DRAFTFOLDER:
      draft_folder = arg;
      break;
      
    case ARG_NODRAFTFOLDER:
      draft_folder = NULL;
      break;
      
    case ARG_DRAFTMESSAGE:
      if (!draft_folder)
	draft_folder = mh_global_profile_get ("Draft-Folder",
					      mu_folder_directory ());
      break;
      
    case ARG_FILTER:
      mh_opt_notimpl ("-filter");
      break;

    case ARG_NOFILTER:
      mh_opt_notimpl ("-nofilter");
      break;
 
    case ARG_FORMAT:
      mh_opt_notimpl_warning ("-format"); 
      reformat_recipients = is_true (arg);
      break;
      
    case ARG_NOFORMAT:
      mh_opt_notimpl_warning ("-noformat"); 
      reformat_recipients = 0;
      break;
      
    case ARG_FORWARD:
      mh_opt_notimpl_warning ("-forward");
      forward_notice = is_true (arg);
      break;
      
    case ARG_NOFORWARD:
      mh_opt_notimpl_warning ("-noforward");
      forward_notice = 0;
      break;
      
    case ARG_MIME:
      mh_opt_notimpl_warning ("-mime");
      mime_encaps = is_true (arg);
      break;
      
    case ARG_NOMIME:
      mh_opt_notimpl_warning ("-nomime");
      mime_encaps = 0;
      break;
      
    case ARG_MSGID:
      append_msgid = is_true (arg);
      break;
      
    case ARG_NOMSGID:
      append_msgid = 0;
      break;

    case ARG_PRESERVE:
      keep_files = is_true (arg);
      break;
      
    case ARG_PUSH:
      background = is_true (arg);
      break;
      
    case ARG_NOPUSH:
      background = 0;
      break;
      
    case ARG_SPLIT:
      split_message = 1;
      split_interval = strtoul (arg, &p, 10);
      if (*p)
	{
	  argp_error (state, "%s: %s", arg, _("invalid number"));
	  exit (1);
	}
      break;
      
    case ARG_VERBOSE:
      verbose = is_true (arg);
      break;
      
    case ARG_NOVERBOSE:
      verbose = 0;
      break;
      
    case ARG_WATCH:
      watch = is_true (arg);
      break;
      
    case ARG_NOWATCH:
      watch = 0;
      break;
      
    case ARG_WIDTH:
      mh_opt_notimpl_warning ("-width");
      width = strtoul (arg, &p, 10);
      if (*p)
	{
	  argp_error (state, _("invalid number"));
	  exit (1);
	}
      break;
      
    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static void
watch_printf (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  vfprintf (stderr, fmt, ap);
  fprintf (stderr, "\n");
  va_end (ap);
}

struct list_elt           /* Element of the send list */
{
  const char *file_name;  /* Duplicated in msg stream, but there's no way
			     to get it from there */
  mu_message_t msg;       /* Corresponding message */
};

static mu_list_t mesg_list;
static mh_context_t *mts_profile;

int
check_file (char *name)
{
  struct list_elt *elt;
  mu_message_t msg;
  char *file_name = mh_expand_name (draft_folder, name, 0);
  
  msg = mh_file_to_message (NULL, file_name);
  if (!msg)
    {
      free (file_name);
      return 1;
    }
  if (!mesg_list && mu_list_create (&mesg_list))
    {
      free (file_name);
      mu_error (_("cannot create message list"));
      return 1;
    }
  elt = xmalloc (sizeof *elt);
  elt->file_name = file_name;
  elt->msg = msg;
  return mu_list_append (mesg_list, elt);
}

void
read_mts_profile ()
{
  const char *p;
  char *hostname = NULL;
  int rc;
  mh_context_t *local_profile;

  p = mh_expand_name (MHLIBDIR, "mtstailor", 0);
  mts_profile = mh_context_create (p, 1);
  mh_context_read (mts_profile);

  p = mu_tilde_expansion ("~/.mtstailor", "/", NULL);
  local_profile = mh_context_create (p, 1);
  if (mh_context_read (local_profile) == 0)
    mh_context_merge (mts_profile, local_profile);
  mh_context_destroy (&local_profile);

  if ((p = mh_context_get_value (mts_profile, "localname", NULL)))
    {
      hostname = xstrdup (p);
      mu_set_user_email_domain (p);
    }
  else if ((rc = mu_get_host_name (&hostname)))
    mu_error (_("cannot get system host name: %s"), mu_strerror (rc));

  if ((p = mh_context_get_value (mts_profile, "localdomain", NULL)))
    {
      char *newdomain;

      if (!hostname)
	exit (1);
      
      newdomain = xmalloc (strlen (hostname) + 1 + strlen (p) + 1);
      strcpy (newdomain, hostname);
      strcat (newdomain, ".");
      strcat (newdomain, p);
      rc = mu_set_user_email_domain (newdomain);
      free (newdomain);
      if (rc)
	{
	  mu_error (_("cannot set user mail domain: %s"), mu_strerror (rc));
	  exit (1);
	}
    }

  if ((p = mh_context_get_value (mts_profile, "username", NULL)))
    {
      size_t len;
      const char *domain;
      char *newemail;
      int rc;
      
      rc = mu_get_user_email_domain (&domain);
      if (rc)
	{
	  mu_error (_("cannot get user email: %s"), mu_strerror (rc));
	  exit (1);
	}
      len = strlen (p) + 1 + strlen (domain) + 1;
      newemail = xmalloc (len);
      strcpy (newemail, p);
      strcat (newemail, "@");
      strcat (newemail, domain);

      rc = mu_set_user_email (newemail);
      if (rc)
	{
	  mu_error (_("cannot set user email (%s): %s"),
		    newemail, mu_strerror (rc));
	  exit (1);
	}
      
      free (newemail);
    }
}


mu_mailer_t
open_mailer ()
{
  const char *url = mh_context_get_value (mts_profile,
					  "url",
					  "sendmail:/usr/sbin/sendmail");
  mu_mailer_t mailer;
  int status;
    
  WATCH ((_("Creating mailer %s"), url));
  status = mu_mailer_create (&mailer, url);
  if (status)
    {
      mu_error (_("cannot create mailer `%s'"), url);
      return NULL;
    }

  if (verbose)
    {
      mu_debug_t debug = NULL;
      mu_mailer_get_debug (mailer, &debug);
      mu_debug_set_level (debug, MU_DEBUG_LEVEL_UPTO (MU_DEBUG_PROT));
    }

  WATCH ((_("Opening mailer %s"), url));
  status = mu_mailer_open (mailer, MU_STREAM_RDWR);
  if (status)
    {
      mu_error (_("cannot open mailer `%s'"), url);
      return NULL;
    }
  return mailer;
}

static void
create_message_id (mu_header_t hdr)
{
  char *p = mh_create_message_id (0);
  mu_header_set_value (hdr, MU_HEADER_MESSAGE_ID, p, 1);
  free (p);
}

static const char *
get_sender_personal ()
{
  const char *s = mh_global_profile_get ("signature", getenv ("SIGNATURE"));
  if (!s)
    {
      struct passwd *pw = getpwuid (getuid ());
      if (pw && pw->pw_gecos[0])
	{
	  char *p = strchr (pw->pw_gecos, ',');
	  if (p)
	    *p = 0;
	  s = pw->pw_gecos;
	}
    }
  return s;
}

static void
set_address_header (mu_header_t hdr, char *name, mu_address_t addr)
{
  size_t s = mu_address_format_string (addr, NULL, 0);
  char *value = xmalloc (s + 1);
  mu_address_format_string (addr, value, s);
  mu_header_set_value (hdr, name, value, 1);
  free (value);
}

void
expand_aliases (mu_message_t msg)
{
  mu_header_t hdr;
  mu_address_t addr_to = NULL,
               addr_cc = NULL,
               addr_bcc = NULL;

  mh_expand_aliases (msg, &addr_to, &addr_cc, &addr_bcc);

  mu_message_get_header (msg, &hdr);
  if (addr_to)
    {
      set_address_header (hdr, MU_HEADER_TO, addr_to);
      mu_address_destroy (&addr_to);
    }

  if (addr_cc)
    {
      set_address_header (hdr, MU_HEADER_CC, addr_cc);
      mu_address_destroy (&addr_cc);
    }

  if (addr_bcc)
    {
      set_address_header (hdr, MU_HEADER_BCC, addr_bcc);
      mu_address_destroy (&addr_bcc);
    }
}

void
fix_fcc (mu_message_t msg)
{
  mu_header_t hdr;
  char *fcc;
  
  mu_message_get_header (msg, &hdr);
  if (mu_header_aget_value (hdr, MU_HEADER_FCC, &fcc) == 0)
    {
      int i, argc;
      char **argv;
      int need_fixup = 0;
      size_t fixup_len = 0;
      
      mu_argcv_get (fcc, ",", NULL, &argc, &argv);
      for (i = 0; i < argc; i += 2)
	{
	  if (strchr ("+%~/=", argv[i][0]) == NULL)
	    {
	      need_fixup++;
	      fixup_len ++;
	    }
	  fixup_len += strlen (argv[i]);
	}

      if (need_fixup)
	{
	  char *p;

	  /* the new fcc string contains: folder names - fixup_len characters
	     long, (argc - 2)/2 comma-space pairs and a terminating
	     nul */
	  fcc = realloc (fcc, fixup_len + argc - 2 + 1);
	  for (i = 0, p = fcc; i < argc; i++)
	    {
	      if (i % 2 == 0)
		{
		  if (strchr ("+%~/=", argv[i][0]) == NULL)
		    *p++ = '+';
		  strcpy (p, argv[i]);
		  p += strlen (argv[i]);
		}
	      else
		{
		  *p++ = ',';
		  *p++ = ' ';
		}
	    }
	  *p = 0;
	}

      mu_argcv_free (argc, argv);

      if (need_fixup)
	{
	  mu_header_set_value (hdr, MU_HEADER_FCC, fcc, 1);
	  WATCH ((_("Fixed fcc: %s"), fcc));
	}
      free (fcc);
    }	  
}

/* Convert MH-style DCC headers to normal BCC.
   FIXME: Normally we should iterate through the headers to catch
   multiple Dcc occurrences (the same holds true for Fcc as well),
   however at the time of this writing we have mu_header_get_field_value,
   but we miss mu_header_set_field_value. */
void
fix_dcc (mu_message_t msg)
{
  mu_header_t hdr;
  char *dcc;
  
  mu_message_get_header (msg, &hdr);
  if (mu_header_aget_value (hdr, MU_HEADER_DCC, &dcc) == 0)
    {
      char *bcc = NULL;
      
      mu_header_set_value (hdr, MU_HEADER_DCC, NULL, 1);
      mu_header_aget_value (hdr, MU_HEADER_BCC, &bcc);
      if (bcc)
	{
	  char *newbcc = realloc (bcc, strlen (bcc) + 1 + strlen (dcc) + 1);
	  if (!newbcc)
	    {
	      mu_error (_("not enough memory"));
	      free (dcc);
	      free (bcc);
	      return;
	    }
	  bcc = newbcc;
	  strcat (bcc, ",");
	  strcat (bcc, dcc);
	  free (dcc);
	}
      else
	bcc = dcc;

      WATCH ((_("Fixed bcc: %s"), bcc));
      mu_header_set_value (hdr, MU_HEADER_BCC, bcc, 1);
      free (bcc);
    }
}

void
backup_file (const char *file_name)
{
  char *new_name = xmalloc (strlen (file_name) + 2);
  char *p = strrchr (file_name, '/');
  if (p)
    {
      size_t len = p - file_name + 1;
      memcpy (new_name, file_name, len);
      new_name[len++] = ',';
      strcpy (new_name + len, p + 1);
    }
  else
    {
      new_name[0] = ',';
      strcpy (new_name + 1, file_name);
    }
  WATCH ((_("Renaming %s to %s"), file_name, new_name));

  if (unlink (new_name) && errno != ENOENT)
    mu_diag_funcall (MU_DIAG_ERROR, "unlink", new_name, errno);
  else if (rename (file_name, new_name))
    mu_error (_("cannot rename `%s' to `%s': %s"),
	      file_name, new_name, mu_strerror (errno));
  free (new_name);
}

int
_action_send (void *item, void *data)
{
  struct list_elt *elt = item;
  mu_message_t msg = elt->msg;
  int rc;
  mu_mailer_t mailer;
  mu_header_t hdr;
  size_t n;

  WATCH ((_("Getting message %s"), elt->file_name));

  if (mu_message_get_header (msg, &hdr) == 0)
    {
      char date[80];
      time_t t = time (NULL);
      struct tm *tm = localtime (&t);
      
      mu_strftime (date, sizeof date, "%a, %d %b %Y %H:%M:%S %z", tm);
      mu_header_set_value (hdr, MU_HEADER_DATE, date, 1);

      if (mu_header_get_value (hdr, MU_HEADER_FROM, NULL, 0, &n))
	{
	  char *from;
	  char *email = mu_get_user_email (NULL);
	  const char *pers = get_sender_personal ();
	  if (pers)
	    {
	      asprintf (&from, "\"%s\" <%s>", pers, email);
	      free (email);
	    }
	  else
	    from = email;

	  mu_header_set_value (hdr, MU_HEADER_FROM, from, 1);
	  free (from);
	}
	  
      if (append_msgid
	  && mu_header_get_value (hdr, MU_HEADER_MESSAGE_ID, NULL, 0, &n))
	create_message_id (hdr);

      if (mu_header_get_value (hdr, MU_HEADER_X_MAILER, NULL, 0, &n))
	{
	  const char *p = mh_context_get_value (mts_profile,
						"x-mailer", "yes");

	  if (!strcmp (p, "yes"))
	    mu_header_set_value (hdr, MU_HEADER_X_MAILER,
				 DEFAULT_X_MAILER, 0);
	  else if (strcmp (p, "no"))
	    mu_header_set_value (hdr, MU_HEADER_X_MAILER, p, 0);
	}
    }
  
  expand_aliases (msg);
  fix_fcc (msg);
  fix_dcc (msg);
  
  mailer = open_mailer ();
  if (!mailer)
    return 1;

  WATCH ((_("Sending message %s"), elt->file_name));
  if (split_message)
    {
      struct timeval delay;
      delay.tv_sec = split_interval;
      delay.tv_usec = 0;
      rc = mu_mailer_send_fragments (mailer, msg,
				     split_size, &delay,
				     NULL, NULL);
    }
  else
    rc = mu_mailer_send_message (mailer, msg, NULL, NULL);
  if (rc)
    {
      mu_error(_("cannot send message: %s"), mu_strerror (rc));
      return 1;
    }

  WATCH ((_("Destroying the mailer")));
  mu_mailer_close (mailer);
  mu_mailer_destroy (&mailer);

  if (!keep_files)
    backup_file (elt->file_name);
  
  return 0;
}

static int
do_send (int argc, char **argv)
{
  int i, rc;
  char *p;
  
  /* Verify all arguments */
  for (i = 0; i < argc; i++)
    if (check_file (argv[i]))
      return 1;

  /* Process the mtstailor file and detach from the console if
     required */
  read_mts_profile ();
  
  if (background && daemon (0, 0) < 0)
    {
      mu_error (_("cannot switch to background: %s"), mu_strerror (errno));
      return 1;
    }

  /* Prepend url specifier to the folder dir. We won't need this
     when the default format becomes configurable */
  asprintf (&p, "mh:%s", mu_folder_directory ());
  mu_set_folder_directory (p);
  free (p);
  
  /* Finally, do the work */
  rc = mu_list_do (mesg_list, _action_send, NULL);
  return rc;
}
	  
int
main (int argc, char **argv)
{
  int index;
  
  MU_APP_INIT_NLS ();
  
  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  mh_read_aliases ();
  
  argc -= index;
  argv += index;

  if (argc == 0)
    {
      struct stat st;
      static char *xargv[2];
      xargv[0] = mh_draft_name ();

      if (stat (xargv[0], &st))
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "stat", xargv[0], errno);
	  return 1;
	}

      if (!use_draft && !mh_usedraft (xargv[0]))
	exit (0);
      xargv[1] = NULL;
      argv = xargv;
      argc = 1;
    }

  return do_send (argc, argv);  
}
