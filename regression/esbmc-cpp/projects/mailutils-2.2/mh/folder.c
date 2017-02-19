/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

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

/* MH folder command */

#include <mh.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include <dirent.h>

#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
#include <obstack.h>

const char *program_version = "folder (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH folder")"\v"
N_("Use -help to obtain the list of traditional MH options.");
static char args_doc[] = N_("[action] [msg]");

static struct argp_option options[] = {
  {N_("Actions are:"), 0, 0, OPTION_DOC, NULL, 0 },
  {"print", ARG_PRINT, NULL, 0,
   N_("list the folders (default)"), 1 },
  {"list",  ARG_LIST,  NULL, 0,
   N_("list the contents of the folder stack"), 1},
  {"pack",  ARG_PACK,  N_("NUMBER"), OPTION_ARG_OPTIONAL,
   N_("remove holes in message numbering, begin numbering from NUMBER (default: first message number)"), 1},
  {"push",  ARG_PUSH,  N_("FOLDER"), OPTION_ARG_OPTIONAL,
    N_("push the folder on the folder stack. If FOLDER is specified, it is pushed. "
       "Otherwise, if a folder is given in the command line (via + or --folder), "
       "it is pushed on stack. Otherwise, the current folder and the top of the folder "
       "stack are exchanged"), 1},
  {"pop",   ARG_POP,    NULL, 0,
   N_("pop the folder off the folder stack"), 1},
  
  {N_("Options are:"), 0, 0, OPTION_DOC, NULL, 2 },
  
  {"folder", ARG_FOLDER, N_("FOLDER"), 0,
   N_("specify folder to operate upon"), 3},
  {"all",    ARG_ALL,    NULL, 0,
   N_("list all folders"), 3},
  {"create", ARG_CREATE, N_("BOOL"), OPTION_ARG_OPTIONAL, 
    N_("create non-existing folders"), 3},
  {"nocreate", ARG_NOCREATE, NULL, OPTION_HIDDEN, ""},
  {"fast",   ARG_FAST, N_("BOOL"), OPTION_ARG_OPTIONAL, 
    N_("list only the folder names"), 3},
  {"nofast", ARG_NOFAST, NULL, OPTION_HIDDEN, ""},
  {"header", ARG_HEADER, N_("BOOL"), OPTION_ARG_OPTIONAL, 
    N_("print the header line"), 3},
  {"noheader", ARG_NOHEADER, NULL, OPTION_HIDDEN, ""},
  {"recurse",ARG_RECURSIVE, N_("BOOL"), OPTION_ARG_OPTIONAL,
    N_("scan folders recursively"), 3},
  {"norecurse", ARG_NORECURSIVE, NULL, OPTION_HIDDEN, ""},
  {"total",  ARG_TOTAL, N_("BOOL"), OPTION_ARG_OPTIONAL, 
    N_("output the total statistics"), 3},
  {"nototal", ARG_NOTOTAL, NULL, OPTION_HIDDEN, ""},
  {"verbose", ARG_VERBOSE, NULL, 0,
   N_("verbosely list actions taken"), 3},
  {"dry-run", ARG_DRY_RUN, NULL, 0,
   N_("do nothing, print what would be done (with --pack)"), 3},
   
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},

  {NULL},
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"print",   2, 0, NULL },
  {"list",    1, 0, NULL },
  {"push",    2, 0, NULL },
  {"pop",     2, 0, NULL },
  {"all",     1, 0, NULL },
  {"pack",    2, 0, NULL },
  {"create",  1, MH_OPT_BOOL, NULL},
  {"fast",    1, MH_OPT_BOOL, NULL},
  {"header",  1, MH_OPT_BOOL, NULL},
  {"recurse", 1, MH_OPT_BOOL, NULL},
  {"total",   1, MH_OPT_BOOL, NULL},
  {NULL},
};

typedef int (*folder_action) ();

static int action_print ();
static int action_list ();
static int action_push ();
static int action_pop ();
static int action_pack ();
static folder_action action = action_print;

int show_all = 0; /* List all folders. Raised by --all switch */
int create_flag = -1; /* Create non-existent folders (--create).
		         -1: Prompt before creating
		          0: Do not create
		          1: Always create without prompting */
int fast_mode = 0; /* Fast operation mode. (--fast) */
  /* The following two vars are three-state, -1 meaning the default for
     current mode */
int print_header = -1; /* Display the header line (--header) */
int print_total = -1;  /* Display total stats */
int verbose = 0;   /* Verbosely list actions taken */
size_t pack_start; /* Number to be assigned to the first message in packed
		      folder. 0 means do not change first message number. */
int dry_run;       /* Dry run mode */ 
const char *push_folder; /* Folder name to push on stack */

const char *mh_seq_name; /* Name of the mh sequence file (defaults to
			    .mh_sequences) */
int has_folder;    /* Folder has been explicitely given */
size_t max_depth = 1;  /* Maximum recursion depth (0 means infinity) */ 

#define OPTION_IS_SET(opt) ((opt) == -1 ? show_all : opt)

static int
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_DRY_RUN:
      dry_run++;
      break;
	
    case ARG_PACK:
      action = action_pack;
      if (arg)
	{
	  char *p;
	  pack_start = strtoul (arg, &p, 10);
	  if (*p)
	    argp_error (state, _("invalid number"));
	}
      break;
      
    case ARG_PRINT:
      action = action_print;
      break;
      
    case ARG_LIST:
      action = action_list;
      break;

    case ARG_PUSH:
      action = action_push;
      if (arg)
	{
	  push_folder = mh_current_folder ();
	  mh_set_current_folder (arg);
	}
      break;
      
    case ARG_POP:
      action = action_pop;
      break;
      
    case ARG_ALL:
      show_all = 1;
      break;

    case ARG_CREATE:
      create_flag = is_true (arg);
      break;

    case ARG_NOCREATE:
      create_flag = 0;
      
    case ARG_FAST:
      fast_mode = is_true (arg);
      break;

    case ARG_NOFAST:
      fast_mode = 0;
      break;
      
    case ARG_HEADER:
      print_header = is_true (arg);
      break;

    case ARG_NOHEADER:
      print_header = 0;
      break;
      
    case ARG_RECURSIVE:
      max_depth = is_true (arg) ? 0 : 1;
      break;

    case ARG_NORECURSIVE:
      max_depth = 0;
      break;
      
    case ARG_TOTAL:
      print_total = is_true (arg);
      break;

    case ARG_NOTOTAL:
      print_total = 0;
      break;
      
    case ARG_FOLDER:
      has_folder = 1;
      push_folder = mh_current_folder ();
      mh_set_current_folder (arg);
      break;
      
    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    case ARG_VERBOSE:
      verbose++;
      break;
      
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}


/* ************************************************************* */
/* Printing */

struct folder_info
{
  char *name;              /* Folder name */
  size_t message_count;    /* Number of messages in this folder */
  size_t min;              /* First used sequence number (=uid) */
  size_t max;              /* Last used sequence number */
  size_t cur;              /* UID of the current message */
  size_t others;           /* Number of non-message files */ 
};

struct obstack folder_info_stack; /* Memory storage for folder infp */
struct folder_info *folder_info;  /* Finalized array of information
				     structures */
size_t folder_info_count;         /* Number of the entries in the array */

size_t message_count;             /* Total number of messages */

int name_prefix_len;              /* Length of the mu_path_folder_dir */

void
install_folder_info (const char *name, struct folder_info *info)
{
  info->name = strdup (name) + name_prefix_len;
  obstack_grow (&folder_info_stack, info, sizeof (*info));
  folder_info_count++;
  message_count += info->message_count;
}

static int
folder_info_comp (const void *a, const void *b)
{
  return strcmp (((struct folder_info *)a)->name,
		 ((struct folder_info *)b)->name);
}

static void
read_seq_file (struct folder_info *info, const char *prefix, const char *name)
{
  char *pname = NULL;
  mh_context_t *ctx;
  const char *p;
  
  asprintf (&pname, "%s/%s", prefix, name);
  if (!pname)
    abort ();
  ctx = mh_context_create (pname, 1);
  mh_context_read (ctx);
  
  p = mh_context_get_value (ctx, "cur", NULL);
  if (p)
    info->cur = strtoul (p, NULL, 0);
  free (pname);
  free (ctx);
}

static void
_scan (const char *name, size_t depth)
{
  DIR *dir;
  struct dirent *entry;
  struct folder_info info;
  char *p;
  struct stat st;
  size_t uid;

  dir = opendir (name);

  if (!dir && errno == ENOENT)
    {
      if (create_flag)
	{
	  if (mh_check_folder (name, create_flag == -1))
	    {
	      push_folder = NULL;
	      return;
	    }
	  dir = opendir (name);
	}
      else
	exit (1);
    }

  if (!dir)
    {
      mu_error (_("cannot scan folder %s: %s"), name, strerror (errno));
      return;
    }

  if (max_depth == 1)
    {
      if (fast_mode && depth > 0)
	{
	  memset (&info, 0, sizeof (info));
	  info.name = strdup (name);
	  install_folder_info (name, &info);
	  closedir (dir);
	  return;
	}
    }
  
  if (max_depth && depth > max_depth)
    {
      closedir (dir);
      return;
    }
  
  memset (&info, 0, sizeof (info));
  info.name = strdup (name);
  while ((entry = readdir (dir)))
    {
      if (entry->d_name[0] == '.')
	{
	  if (strcmp (entry->d_name, mh_seq_name) == 0)
	    read_seq_file (&info, name, entry->d_name);
	}
      else if (entry->d_name[0] != ',')
	{
	  asprintf (&p, "%s/%s", name, entry->d_name);
	  if (stat (p, &st) < 0)
	    mu_diag_funcall (MU_DIAG_ERROR, "stat", p, errno);
	  else if (S_ISDIR (st.st_mode))
	    {
	      info.others++;
	      _scan (p, depth+1);
	    }
	  else
	    {
	      char *endp;
	      uid = strtoul (entry->d_name, &endp, 10);
	      if (*endp)
		info.others++;
	      else
		{
		  info.message_count++;
		  if (info.min == 0 || uid < info.min)
		    info.min = uid;
		  if (uid > info.max)
		    info.max = uid;
		}
	    }
	}
    }
  
  if (info.cur)
    {
      asprintf (&p, "%s/%s", name, mu_umaxtostr (0, info.cur));
      if (stat (p, &st) < 0 || !S_ISREG (st.st_mode))
	info.cur = 0;
      free (p);
    }
  closedir (dir);
  if (depth > 0)
    install_folder_info (name, &info);
}
    
static void
print_all ()
{
  struct folder_info *info, *end = folder_info + folder_info_count;

  for (info = folder_info; info < end; info++)
    {
      int len = strlen (info->name);
      if (len < 22)
	printf ("%22.22s", info->name);
      else
	printf ("%s", info->name);
      
      if (strcmp (info->name, mh_current_folder ()) == 0)
	printf ("+");
      else
	printf (" ");
      
      if (info->message_count)
	{
	  printf (ngettext(" has %4lu message  (%4lu-%4lu)",
			   " has %4lu messages (%4lu-%4lu)",
			   info->message_count),
		  (unsigned long) info->message_count,
		  (unsigned long) info->min,
		  (unsigned long) info->max);
	  if (info->cur)
	    printf ("; cur=%4lu", (unsigned long) info->cur);
	}
      else
	{
	  printf (_(" has no messages"));
	}
      
      if (info->others)
	{
	  if (!info->cur)
	    printf (";           ");
	  else
	    printf ("; ");
	  printf (_("(others)"));
	}
      printf (".\n");
    }
}

static void
print_fast ()
{
  struct folder_info *info, *end = folder_info + folder_info_count;

  for (info = folder_info; info < end; info++)
    printf ("%s\n", info->name);
}

static int
action_print ()
{
  const char *folder_dir = mu_folder_directory ();
  mh_seq_name = mh_global_profile_get ("mh-sequences", MH_SEQUENCES_FILE);

  name_prefix_len = strlen (folder_dir);
  if (folder_dir[name_prefix_len - 1] == '/')
    name_prefix_len++;
  name_prefix_len++;  /* skip past the slash */

  obstack_init (&folder_info_stack);

  if (show_all)
    {
      _scan (folder_dir, 0);
    }
  else
    {
      char *p = mh_expand_name (NULL, mh_current_folder (), 0);
      _scan (p, 1);
      free (p);
    }
  
  folder_info = obstack_finish (&folder_info_stack);
  qsort (folder_info, folder_info_count, sizeof (folder_info[0]),
	 folder_info_comp);

  if (fast_mode)
    print_fast ();
  else
    {
      if (OPTION_IS_SET (print_header))
	printf (_("Folder                  # of messages     (  range  )  cur msg   (other files)\n"));
		
      print_all ();

      if (OPTION_IS_SET (print_total))
	{
	  printf ("\n%24.24s=", _("TOTAL"));
	  printf (ngettext ("%4lu message  ", "%4lu messages ",
			    message_count),
		  (unsigned long) message_count);
	  printf (ngettext ("in %4lu folder", "in %4lu folders",
			    folder_info_count),
		  (unsigned long) folder_info_count);
	  printf ("\n");
	}
    }
  if (push_folder)
    mh_global_save_state ();

  return 0;
}


/* ************************************************************* */
/* Listing */

static int
action_list ()
{
  const char *stack = mh_global_context_get ("Folder-Stack", NULL);

  printf ("%s", mh_current_folder ());
  if (stack)
    printf (" %s", stack);
  printf ("\n");
  return 0;
}


/* ************************************************************* */
/* Push & pop */

static void
get_stack (int *pc, char ***pv)
{
  int status;
  const char *stack = mh_global_context_get ("Folder-Stack", NULL);
  if (!stack)
    {
      *pc = 0;
      *pv = NULL;
    }
  else if ((status = mu_argcv_get (stack, NULL, "#", pc, pv)) != 0)
    {
      mu_diag_funcall (MU_DIAG_ERROR, "mu_argcv_get", stack, status);
      exit (1);
    }
}

static void
set_stack (int c, char **v)
{
  char *str;
  int status = mu_argcv_string (c, v, &str);
  if (status)
    {
      mu_error ("%s", mu_strerror (status));
      exit (1);
    }
  mu_argcv_free (c, v);
  mh_global_context_set ("Folder-Stack", str);
  free (str);
}

static void
push_val (int *pc, char ***pv, const char *val)
{
  int c = *pc;
  char **v = *pv;

  c++;
  if (c == 1)
    {
      v = xcalloc (c + 1, sizeof (*v));
    }
  else
    {
      v = xrealloc (v, (c + 1) * sizeof (*v));
      memmove (&v[1], &v[0], c * sizeof (*v));
    }
  v[0] = xstrdup (val);

  *pv = v;
  *pc = c;
}

static char *
pop_val (int *pc, char ***pv)
{
  char *val;
  int c;
  char **v;
  
  if (*pc == 0)
    return NULL;
  c = *pc;
  v = *pv;
  val = v[0];
  memmove (&v[0], &v[1], c * sizeof (*v));
  c--;

  *pc = c;
  *pv = v;
  return val;
}
  
static int
action_push ()
{
  int c;
  char **v;

  get_stack (&c, &v);
  
  if (push_folder)
    push_val (&c, &v, push_folder);
  else 
    {
      char *t = v[0];
      v[0] = xstrdup (mh_current_folder ());
      mh_set_current_folder (t);
      free (t);
    }

  set_stack (c, v);

  action_list ();
  mh_global_save_state ();
  return 0;
}

static int
action_pop ()
{
  int c;
  char **v;

  get_stack (&c, &v);

  if (c)
    {
      char *p = pop_val (&c, &v);
      set_stack (c, v);
      mh_set_current_folder (p);
      free (p);
    }

  action_list ();
  mh_global_save_state ();
  return 0;
}


/* ************************************************************* */
/* Packing */

struct pack_tab
{
  size_t orig;
  size_t new;
};

static int
pack_rename (struct pack_tab *tab, int reverse)
{
  int rc;
  const char *s1;
  const char *s2;
  const char *from, *to;
  
  s1 = mu_umaxtostr (0, tab->orig);
  s2 = mu_umaxtostr (1, tab->new);

  if (!reverse)
    {
      from = s1;
      to = s2;
    }
  else
    {
      from = s2;
      to = s1;
    }

  if (verbose)
    fprintf (stderr, _("Renaming %s to %s\n"), from, to);

  if (!dry_run)
    {
      if ((rc = rename (from, to)))
	mu_error (_("cannot rename `%s' to `%s': %s"),
		  from, to, mu_strerror (errno));
    }
  else
    rc = 0;
  
  return rc;
}

/* Reverse ordering of COUNT entries in array TAB */
static void
reverse (struct pack_tab *tab, size_t count)
{
  size_t i, j;

  for (i = 0, j = count-1; i < j; i++, j--)
    {
      size_t tmp;
      tmp = tab[i].orig;
      tab[i].orig = tab[j].orig;
      tab[j].orig = tmp;

      tmp = tab[i].new;
      tab[i].new = tab[j].new;
      tab[j].new = tmp;
    }
} 

static void
roll_back (const char *folder_name, struct pack_tab *pack_tab, size_t i)
{
  size_t start;
  
  if (i == 0)
    return;
  
  start = i - 1;
  mu_error (_("rolling back changes..."));
  while (--i >= 0)
    if (pack_rename (pack_tab + i, 1))
      {
	mu_error (_("CRITICAL ERROR: Folder `%s' left in an inconsistent state, because an error\n"
		    "occurred while trying to roll back the changes.\n"
		    "Message range %s-%s has been renamed to %s-%s."),
		  folder_name,
		  mu_umaxtostr (0, pack_tab[0].orig),
                  mu_umaxtostr (1, pack_tab[start].orig),
		  mu_umaxtostr (2, pack_tab[0].new),
                  mu_umaxtostr (3, pack_tab[start].new));
	mu_error (_("You will have to fix it manually."));
	exit (1);
      }
  mu_error (_("folder `%s' restored successfully"), folder_name);
}

struct fixup_data
{
  const char *folder_dir;
  struct pack_tab *pack_tab;
  size_t count;
};

static int
pack_cmp (const void *a, const void *b)
{
  const struct pack_tab *pa = a;
  const struct pack_tab *pb = b;

  if (pa->orig < pb->orig)
    return -1;
  else if (pa->orig > pb->orig)
    return 1;
  return 0;
}

static size_t
pack_xlate (struct pack_tab *pack_tab, size_t count, size_t n)
{
  struct pack_tab key, *p;

  key.orig = n;
  p = bsearch (&key, pack_tab, count, sizeof pack_tab[0], pack_cmp);
  return p ? p->new : 0;
}

static int
_fixup (const char *name, const char *value, struct fixup_data *fd, int flags)
{
  int i, j, argc;
  char **argv;
  mh_msgset_t msgset;

  if (verbose)
    fprintf (stderr, "Sequence `%s'...\n", name);
  
  if (mu_argcv_get (value, "", NULL, &argc, &argv))
    return 0;

  msgset.list = xcalloc (argc, sizeof msgset.list[0]);
  for (i = j = 0; i < argc; i++)
    {
      size_t n = pack_xlate (fd->pack_tab, fd->count,
			     strtoul (argv[i], NULL, 0));
      if (n)
	msgset.list[j++] = n;
    }
  msgset.count = j;

  mh_seq_add (name, &msgset, flags | SEQ_ZERO);
  free (msgset.list);

  if (verbose)
    {
      const char *p = mh_seq_read (name, flags);
      fprintf (stderr, "Sequence %s: %s\n", name, p);
    }
  
  return 0;
}

static int
fixup_global (const char *name, const char *value, void *data)
{
  return _fixup (name, value, data, 0);
}

static int
fixup_private (const char *name, const char *value, void *data)
{
  struct fixup_data *fd = data;
  int nlen = strlen (name);  
  if (nlen < 4 || memcmp (name, "atr-", 4))
    return 0;
  name += 4;

  nlen = strlen (name) - strlen (fd->folder_dir);
  if (nlen > 0 && strcmp (name + nlen, fd->folder_dir) == 0)
    {
      int rc;
      char *s = xmalloc (nlen);
      memcpy (s, name, nlen - 1);
      s[nlen-1] = 0;
      rc = _fixup (s, value, fd, SEQ_PRIVATE);
      free (s);
    }
  return 0;
}

int
action_pack ()
{
  const char *folder_dir = mh_expand_name (NULL, mh_current_folder (), 0);
  mu_mailbox_t mbox = mh_open_folder (mh_current_folder (), 0);
  struct pack_tab *pack_tab;
  size_t i, count, start;
  int status;
  struct fixup_data fd;
  
  /* Allocate pack table */
  if (mu_mailbox_messages_count (mbox, &count))
    {
      mu_error (_("cannot read input mailbox: %s"), mu_strerror (errno));
      return 1;
    }
  pack_tab = xcalloc (count, sizeof pack_tab[0]); /* Never freed. No use to
						     try to. */

  /* Populate it with message numbers */
  if (verbose)
    fprintf (stderr, _("Getting message numbers.\n"));
    
  for (i = 0; i < count; i++)
    {
      mu_message_t msg;
      status = mu_mailbox_get_message (mbox, i + 1, &msg);
      if (status)
	{
	  mu_error (_("%lu: cannot get message: %s"),
		    (unsigned long) i, mu_strerror (status));
	  return 1;
	}
      mh_message_number (msg, &pack_tab[i].orig);
    }
  if (verbose)
    fprintf (stderr, ngettext ("%s message number collected.\n",
			       "%s message numbers collected.\n",
			       (unsigned long) count),
	     mu_umaxtostr (0, count));
  
  mu_mailbox_close (mbox);
  mu_mailbox_destroy (&mbox);

  /* Compute new message numbers */
  if (pack_start == 0)
    pack_start = pack_tab[0].orig;

  for (i = 0, start = pack_start; i < count; i++)
    pack_tab[i].new = start++;

  if (pack_start > pack_tab[0].orig)
    {
      if (verbose)
	fprintf (stderr, _("Reverting pack table.\n"));
      reverse (pack_tab, i);
    }
  
  /* Change to the folder directory and rename messages */
  status = chdir (folder_dir);
  if (status)
    {
      mu_error (_("cannot change to directory `%s': %s"),
		folder_dir, mu_strerror (status));
      return 1;
    }

  for (i = 0; i < count; i++)
    {
      if (pack_rename (pack_tab + i, 0))
	{
	  roll_back (folder_dir, pack_tab, i);
	  return 1;
	}
    }

  if (verbose)
    fprintf (stderr, _("Finished packing messages.\n"));

  /* Fix-up sequences */
  fd.folder_dir = folder_dir;
  fd.pack_tab = pack_tab;
  fd.count = count;
  if (verbose)
    fprintf (stderr, _("Fixing global sequences\n"));
  mh_global_sequences_iterate (fixup_global, &fd);
  if (verbose)
    fprintf (stderr, _("Fixing private sequences\n"));
  mh_global_context_iterate (fixup_private, &fd);

  if (!dry_run)
    mh_global_save_state ();
  
  return 0;
}

int
main (int argc, char **argv)
{
  int index = 0;
  mh_msgset_t msgset;

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  /* If  folder  is invoked by a name ending with "s" (e.g.,  folders),
     `-all'  is  assumed */
  if (mu_program_name[strlen (mu_program_name) - 1] == 's')
    show_all = 1;
  
  if (has_folder)
    {
      /* If a +folder is given along with the -all switch, folder will, in
	 addition to setting the current folder, list the top-level
	 subfolders for the current folder (with -norecurse) or list all
	 sub-folders under the current folder recursively (with -recurse). */
      if (show_all && max_depth)
	max_depth = 2;
      show_all = 0;
    }
    
  if (argc - index == 1)
    {
      mu_mailbox_t mbox = mh_open_folder (mh_current_folder (), 0);
      mh_msgset_parse (mbox, &msgset, argc - index, argv + index, "cur");
      mh_msgset_current (mbox, &msgset, 0);
      mh_global_save_state ();
      mu_mailbox_close (mbox);
      mu_mailbox_destroy (&mbox);
    }
  else if (argc - index > 1)
    {
      mu_error (_("too many arguments"));
      exit (1);
    }
  
  return (*action) ();
}
