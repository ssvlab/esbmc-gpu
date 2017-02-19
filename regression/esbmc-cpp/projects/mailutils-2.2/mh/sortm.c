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

/* MH sortm command */

#include <mh.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>

const char *program_version = "sortm (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH sortm")"\v"
N_("Use -help to obtain the list of traditional MH options.");
static char args_doc[] = N_("[msgs]");

#define ARG_QUICKSORT 1024
#define ARG_SHELL     1025

/* GNU options */
static struct argp_option options[] = {
  {"folder",        ARG_FOLDER,        N_("FOLDER"), 0,
   N_("specify folder to operate upon")},

  {N_("Setting sort keys:"), 0,  NULL, OPTION_DOC,  NULL, 0},
  {"datefield",     ARG_DATEFIELD,     N_("STRING"), 0,
   N_("sort on the date field (default `Date:')"), 10},
  {"nodatefield",   ARG_NODATEFIELD,   NULL,       0,
   N_("undo the effect of the last --datefield option"), 10},
  {"limit",         ARG_LIMIT,         N_("DAYS"), 0,
   N_("consider two datefields equal if their difference lies within the given nuber of DAYS."), 11},
  {"nolimit",       ARG_NOLIMIT,       NULL,       0,
   N_("undo the effect of the last --limit option"), 11},
  {"textfield",     ARG_TEXTFIELD,     N_("STRING"), 0,
   N_("sort on the text field"), 15},
  {"notextfield",   ARG_NOTEXTFIELD,   NULL,       0,
   N_("undo the effect of the last --textfield option"), 15},
  {"numfield",      ARG_NUMFIELD,      N_("STRING"), 0,
   N_("sort on the numeric field"), 16},
  
  {N_("Actions:"), 0,  NULL, OPTION_DOC,  NULL, 16},
  {"reorder", ARG_REORDER,    0, 0,
   N_("reorder the messages (default)"), 20 },
  {"dry-run", ARG_DRY_RUN,    0, 0,
   N_("do not do anything, only show what would have been done"), 20 },
  {"list",    ARG_LIST,   0, 0,
   N_("list the sorted messages"), 20 },
  {"form",    ARG_FORM, N_("FILE"),   0,
   N_("read format from given file"), 23},
  {"format",  ARG_FORMAT, N_("FORMAT"), 0,
   N_("use this format string"), 23},

  {"verbose",       ARG_VERBOSE,       N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("verbosely list executed actions"), 30 },
  {"noverbose",     ARG_NOVERBOSE,     NULL, OPTION_HIDDEN, "" },

  {N_("Select sort algorithm:"), 0,  NULL, OPTION_DOC,  NULL, 30},

  {"shell",     ARG_SHELL,      0, 0,
   N_("use shell algorithm"), 40 },
  {"quicksort", ARG_QUICKSORT,  0, 0,
   N_("use quicksort algorithm (default)"), 40 },

  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},

  { NULL },
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"datefield",     1, 0, "field" },
  {"nodatefield",   3, 0, 0 },
  {"textfield",     1, 0, "field" },
  {"notextfield",   3, 0, 0 },
  {"limit",         1, 0, "days" },
  {"nolimit",       3, 0, 0 },
  {"verbose",       1, MH_OPT_BOOL, NULL},
  { NULL },
};

static int limit;
static int verbose;
static mu_mailbox_t mbox;
static const char *mbox_path;
static mh_msgset_t msgset;

#define ACTION_REORDER   0
#define ACTION_DRY_RUN   1
#define ACTION_LIST      2  

static int algorithm = ARG_QUICKSORT;
static int action = ACTION_REORDER;
static char *format_str = mh_list_format;
static mh_format_t format;

typedef int (*compfun) (void *, void *);
static void addop (char *field, compfun comp);
static void remop (compfun comp);
static int comp_text (void *a, void *b);
static int comp_date (void *a, void *b);
static int comp_number (void *a, void *b);

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_FOLDER:
      mh_set_current_folder (arg);
      break;
      
    case ARG_DATEFIELD:
      addop (arg, comp_date);
      break;
      
    case ARG_NUMFIELD:
      addop (arg, comp_number);
      break;

    case ARG_NODATEFIELD:
      remop (comp_date);
      break;
      
    case ARG_TEXTFIELD:
      addop (arg, comp_text);
      break;
      
    case ARG_NOTEXTFIELD:
      remop (comp_text);
      break;
      
    case ARG_LIMIT:
      limit = strtoul (arg, NULL, 0);
      break;
      
    case ARG_NOLIMIT:
      limit = -1;
      break;
      
    case ARG_VERBOSE:
      if (!arg || mu_isalpha (arg[0]))
	verbose = is_true (arg);
      else
	verbose = arg[0] - '0';
      break;
      
    case ARG_NOVERBOSE:
      verbose = 0;
      break;
      
    case ARG_FORM:
      mh_read_formfile (arg, &format_str);
      break;
      
    case ARG_FORMAT:
      format_str = arg;
      break;

    case ARG_REORDER:
      action = ACTION_REORDER;
      break;
      
    case ARG_LIST:
      action = ACTION_LIST;
      break;

    case ARG_DRY_RUN:
      action = ACTION_DRY_RUN;
      if (!verbose)
	verbose = 1;
      break;

    case ARG_SHELL:
    case ARG_QUICKSORT:
      algorithm = key;
      break;
      
    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}


/* *********************** Comparison functions **************************** */
struct comp_op {
  char *field;
  compfun comp;
};

static mu_list_t oplist;

static void
addop (char *field, compfun comp)
{
  struct comp_op *op = xmalloc (sizeof (*op));
  
  if (!oplist)
    {
      if (mu_list_create (&oplist))
	{
	  mu_error (_("can't create operation list"));
	  exit (1);
	}
      mu_list_set_destroy_item (oplist, mu_list_free_item);
    }
  op->field = field;
  op->comp = comp;
  mu_list_append (oplist, op);
}

struct rem_data {
  struct comp_op *op;
  compfun comp;
};

static int
rem_action (void *item, void *data)
{
  struct comp_op *op = item;
  struct rem_data *d = data;
  if (d->comp == op->comp)
    d->op = op;
  return 0;
}

static void
remop (compfun comp)
{
  struct rem_data d;
  d.comp = comp;
  d.op = NULL;
  mu_list_do (oplist, rem_action, &d);
  mu_list_remove (oplist, d.op);
}

struct comp_data {
  int r;
  mu_message_t m[2];
};

static int
compare_action (void *item, void *data)
{
  struct comp_op *op = item;
  struct comp_data *dp = data;
  char *a, *ap, *b, *bp;
  mu_header_t h;
  
  if (mu_message_get_header (dp->m[0], &h)
      || mu_header_aget_value (h, op->field, &a))
    return 0;

  if (mu_message_get_header (dp->m[1], &h)
      || mu_header_aget_value (h, op->field, &b))
    {
      free (a);
      return 0;
    }

  ap = a;
  bp = b;
  if (mu_c_strcasecmp (op->field, MU_HEADER_SUBJECT) == 0)
    {
      if (mu_c_strncasecmp (ap, "re:", 3) == 0)
	ap += 3;
      if (mu_c_strncasecmp (b, "re:", 3) == 0)
	bp += 3;
    }
  
  dp->r = op->comp (ap, bp);
  free (a);
  free (b);

  return dp->r; /* go on until the difference is found */
}

static int
compare_messages (mu_message_t a, mu_message_t b, size_t anum, size_t bnum)
{
  struct comp_data d;

  d.r = 0;
  d.m[0] = a;
  d.m[1] = b;
  mu_list_do (oplist, compare_action, &d);
  if (d.r == 0)
    {
      if (anum < bnum)
	d.r = -1;
      else if (anum > bnum)
	d.r = 1;
    }
      
  if (verbose > 1)
    fprintf (stderr, "%d\n", d.r);
  return d.r;
}

static int
comp_text (void *a, void *b)
{
  return mu_c_strcasecmp (a, b);
}

static int
comp_number (void *a, void *b)
{
  long na, nb;

  na = strtol (a, NULL, 0);
  nb = strtol (b, NULL, 0);
  if (na > nb)
    return 1;
  else if (na < nb)
    return -1;
  return 0;
}

/*FIXME: Also used in imap4d*/
static int
_parse_822_date (char *date, time_t * timep)
{
  struct tm tm;
  mu_timezone tz;
  const char *p = date;

  if (mu_parse822_date_time (&p, date + strlen (date), &tm, &tz) == 0)
    {
      *timep = mu_tm2time (&tm, &tz);
      return 0;
    }
  return 1;
}

static int
comp_date (void *a, void *b)
{
  time_t ta, tb;
  
  if (_parse_822_date (a, &ta) || _parse_822_date (b, &tb))
    return 0;

  if (ta < tb)
    {
      if (limit && tb - ta <= limit)
	return 0;
      return -1;
    }
  else if (ta > tb)
    {
      if (limit && ta - tb <= limit)
	return 0;
      return 1;
    }
  return 0;
}


/* *********************** Sorting routines ***************************** */

static int
comp0 (size_t na, size_t nb)
{
  mu_message_t a, b;

  if (mu_mailbox_get_message (mbox, na, &a)
      || mu_mailbox_get_message (mbox, nb, &b))
    return 0;
  if (verbose > 1)
    fprintf (stderr,
	     _("comparing messages %s and %s: "),
	     mu_umaxtostr (0, na),
	     mu_umaxtostr (1, nb));
  return compare_messages (a, b, na, nb);
}

int
comp (const void *a, const void *b)
{
  return comp0 (* (size_t*) a, * (size_t*) b);
}


/* ****************************** Shell sort ****************************** */
#define prevdst(h) ((h)-1)/3

static int
startdst (unsigned count, int *num)
{
  int i, h;

  for (i = h = 1; 9*h + 4 < count; i++, h = 3*h+1)
    ;
  *num = i;
  return h;
}

void
shell_sort ()
{
    int h, s, i, j;
    size_t hold;

    for (h = startdst (msgset.count, &s); s > 0; s--, h = prevdst (h))
      {
	if (verbose > 1)
	  fprintf (stderr, _("distance %d\n"), h);
        for (j = h; j < msgset.count; j++)
	  {
            hold = msgset.list[j];
            for (i = j - h;
		 i >= 0 && comp0 (hold, msgset.list[i]) < 0; i -= h)
	      msgset.list[i + h] = msgset.list[i];
	    msgset.list[i + h] = hold;
	  }
      }
}


/* ****************************** Actions ********************************* */

void
list_message (size_t num)
{
  mu_message_t msg = NULL;
  char *buffer;
  mu_mailbox_get_message (mbox, num, &msg);
  mh_format (&format, msg, num, 76, &buffer);
  printf ("%s\n", buffer);
  free (buffer);
}

void
swap_message (size_t a, size_t b)
{
  char *path_a, *path_b;
  char *tmp;
  
  asprintf (&path_a, "%s/%s", mbox_path, mu_umaxtostr (0, a));
  asprintf (&path_b, "%s/%s", mbox_path, mu_umaxtostr (1, b));
  tmp = mu_tempname (mbox_path);
  rename (path_a, tmp);
  unlink (path_a);
  rename (path_b, path_a);
  unlink (path_b);
  rename (tmp, path_b);
  free (tmp);
}

void
transpose(size_t i, size_t n)
{
  size_t j;

  for (j = i+1; j < msgset.count; j++)
    if (msgset.list[j] == n)
      {
	size_t t = msgset.list[i];
	msgset.list[i] = msgset.list[j];
	msgset.list[j] = t;
	break;
      }
}
  
static int got_signal;

RETSIGTYPE
sighandler (int sig)
{
  got_signal = 1;
}

void
sort ()
{
  size_t *oldlist, i;
  oldlist = xmalloc (msgset.count * sizeof (*oldlist));
  memcpy (oldlist, msgset.list, msgset.count * sizeof (*oldlist));

  switch (algorithm)
    {
    case ARG_QUICKSORT:
      qsort(msgset.list, msgset.count, sizeof(msgset.list[0]),
	    comp);
      break;

    case ARG_SHELL:
      shell_sort();
      break;
    }

  switch (action)
    {
    case ACTION_LIST:
      for (i = 0; i < msgset.count; i++)
	list_message (msgset.list[i]);
      break;

    default:
      /* Install signal handlers */
      signal (SIGINT, sighandler);
      signal (SIGQUIT, sighandler);
      signal (SIGTERM, sighandler);
      
      if (verbose)
	fprintf (stderr, _("Transpositions:\n"));
      for (i = 0, got_signal = 0; !got_signal && i < msgset.count; i++)
	{
	  if (msgset.list[i] != oldlist[i])
	    {
	      size_t old_num, new_num;
	      mu_message_t msg;

	      mu_mailbox_get_message (mbox, oldlist[i], &msg);
	      mh_message_number (msg, &old_num);
	      mu_mailbox_get_message (mbox, msgset.list[i], &msg);
	      mh_message_number (msg, &new_num);
	      transpose (i, oldlist[i]);
	      if (verbose)
		fprintf (stderr, "{%s, %s}\n",
			 mu_umaxtostr (0, old_num),
			 mu_umaxtostr (1, new_num));
	      if (action == ACTION_REORDER)
		swap_message (old_num, new_num);
	    }
	}
    }
}


/* Main */

int
main (int argc, char **argv)
{
  int index;
  mu_url_t url;
  
  MU_APP_INIT_NLS ();
  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option,
		 args_doc, doc, opt_handler, NULL, &index);
  if (!oplist)
    addop ("date", comp_date);

  if (action == ACTION_LIST && mh_format_parse (format_str, &format))
    {
      mu_error (_("Bad format string"));
      exit (1);
    }
  
  mbox = mh_open_folder (mh_current_folder (), 0);
  mu_mailbox_get_url (mbox, &url);
  mbox_path = mu_url_to_string (url);
  if (memcmp (mbox_path, "mh:", 3) == 0)
    mbox_path += 3;
  
  argc -= index;
  argv += index;

  mh_msgset_parse (mbox, &msgset, argc, argv, "all");
  sort (mbox, msgset);
  return 0;
}
