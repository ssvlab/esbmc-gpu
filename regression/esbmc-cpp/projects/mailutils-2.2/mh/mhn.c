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

/* MH mhn command */

#include <mh.h>
#include <signal.h>
#include <mailutils/mime.h>
#define obstack_chunk_alloc xmalloc
#define obstack_chunk_free free
#include <obstack.h>

const char *program_version = "mhn (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH mhn")"\v"
N_("Options marked with `*' are not yet implemented.\n\
Use -help to obtain the list of traditional MH options.");
static char args_doc[] = "[msgs]";

static struct argp_option options[] = {
  {"folder",       ARG_FOLDER,       N_("FOLDER"), 0,
   N_("specify folder to operate upon"), 0},
  {"file",         ARG_FILE,         N_("FILE"),   0,
   N_("specify file to operate upon"), 0},
  
  {N_("MIME editing options"),   0,  NULL, OPTION_DOC,  NULL, 5},
  {"compose",      ARG_COMPOSE,      N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("compose the MIME message (default)"), 6},
  {"nocompose",    ARG_NOCOMPOSE,    NULL, OPTION_HIDDEN, "", 6},
  
  {N_("Listing options"),   0,  NULL, OPTION_DOC,  NULL, 0},
  {"list",         ARG_LIST,         N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("list the table of contents"), 11 },
  {"nolist",       ARG_NOLIST,       NULL, OPTION_HIDDEN, "", 11 },
  {"headers",      ARG_HEADER,       N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("print the banner above the listing"), 12},
  {"noheaders",    ARG_NOHEADERS,    NULL, OPTION_HIDDEN, "", 12 },
  {"realsize",     ARG_REALSIZE,     N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("list the decoded sizes"), 12},
  {"norealsize",   ARG_NOREALSIZE,   NULL, OPTION_HIDDEN, "", 12 },

  {N_("Display options"),   0,  NULL, OPTION_DOC,  NULL, 20},
  {"show",         ARG_SHOW,         N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("display the contents of the messages"), 21},
  {"noshow",       ARG_NOSHOW,       NULL, OPTION_HIDDEN, "", 21 },
  {"serialonly",   ARG_SERIALONLY,   N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("* display messages serially"), 22},
  {"noserialonly", ARG_NOSERIALONLY, NULL, OPTION_HIDDEN, "", 22 },
  {"form",         ARG_FORM,         N_("FILE"), 0,
   N_("read mhl format from FILE"), 22},
  {"pause",        ARG_PAUSE,        N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("pause prior to displaying content"), 22},
  {"nopause",      ARG_NOPAUSE,      NULL, OPTION_HIDDEN, "", 22 },
   
  {N_("Saving options"),   0,  NULL, OPTION_DOC,  NULL, 30},
  {"store",        ARG_STORE,        N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("store the contents of the messages on disk"), 31},
  {"nostore",      ARG_NOSTORE,      NULL, OPTION_HIDDEN, "", 31 },
  {"auto",         ARG_AUTO,         N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("use filenames from the content headers"), 31},
  {"noauto",       ARG_NOAUTO,       NULL, OPTION_HIDDEN, "", 31 },
  {"charset",      ARG_CHARSET,      N_("NAME"), 0,
   N_("use this charset to represent attachment file names"), 31},
  
  {N_("Other options"),    0,  NULL, OPTION_DOC,  NULL, 40},
  {"part",         ARG_PART,         N_("PART"), 0,
   N_("limit the scope of the operation to the given part"), 41},
  {"type",         ARG_TYPE,         N_("CONTENT"), 0,
   N_("operate on message part with given multipart content"), 41 },
  {"verbose",      ARG_VERBOSE,     N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("print additional information"), 41 },
  {"noverbose",    ARG_NOVERBOSE,   NULL, OPTION_HIDDEN, "", 41 },
  {"quiet",        ARG_QUIET, 0, 0,
   N_("be quiet")},
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  {NULL}
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"file",          2, 0, "filename"},
  {"list",          1, MH_OPT_BOOL, NULL},
  {"headers",       1, MH_OPT_BOOL, NULL},
  {"realsize",      1, MH_OPT_BOOL, NULL},
  {"show",          2, MH_OPT_BOOL, NULL},
  {"serialonly",    2, MH_OPT_BOOL, NULL},
  {"form",          2, 0, "formfile"},
  {"pause",         3, MH_OPT_BOOL, NULL},
  {"store",         1, MH_OPT_BOOL, NULL},
  {"auto",          1, MH_OPT_BOOL, NULL},
  {"part",          3, 0, "part"},
  {"type",          1, 0, "type"},
  {"verbose",       1, MH_OPT_BOOL, NULL},
  {NULL}
};

typedef struct _msg_part *msg_part_t;

static msg_part_t msg_part_create (size_t num);
static void msg_part_destroy (msg_part_t p);
static int msg_part_eq (msg_part_t a, msg_part_t b);
static void msg_part_incr (msg_part_t p);
static void msg_part_decr (msg_part_t p);
static void msg_part_set_subpart (msg_part_t p, size_t subpart);
static void msg_part_print (msg_part_t p, int width);
static msg_part_t msg_part_parse (char *str);
static int msg_part_level (msg_part_t p);
static size_t msg_part_subpart (msg_part_t p, int level);

enum mhn_mode {
  mode_compose,
  mode_list,
  mode_show,
  mode_store,
};

static enum mhn_mode mode = mode_compose;

#define OPT_HEADERS    001
#define OPT_REALSIZE   002
#define OPT_PAUSE      004
#define OPT_AUTO       010
#define OPT_SERIALONLY 020
#define OPT_VERBOSE    040
#define OPT_QUIET      100

static int mode_options = OPT_HEADERS;
static char *formfile;
static char *content_type;
static char *content_subtype;
static char *input_file;
static int width = 80;
static char *charset;  /* Charset for output file names.  NULL means
			  no recoding is necessary. */

static mh_msgset_t msgset;
static mu_mailbox_t mbox;
static mu_message_t message;
static msg_part_t req_part;

/* Show flags */
#define MHN_EXCLUSIVE_EXEC  001
#define MHN_STDIN           002
#define MHN_LISTING         004
#define MHN_CONFIRM         010

void
sfree (char **ptr)
{
  if (!*ptr)
    return;
  free (*ptr);
  *ptr = NULL;
}

void
split_content (const char *content, char **type, char **subtype)
{
  char *p = strchr (content, '/');
  if (p)
    {
      int len = p - content;
      *type = xmalloc (len + 1);
      memcpy (*type, content, len);
      (*type)[len] = 0;

      p++;
      *subtype = xmalloc (strlen (p) + 1);
      strcpy (*subtype, p);
    }
  else
    {
      *type = xmalloc (strlen (content) + 1);
      strcpy (*type, content);
      *subtype = NULL;
    }
}

int
_get_hdr_value (mu_header_t hdr, const char *name, char **value)
{
  int status = mu_header_aget_value (hdr, name, value);
  if (status == 0)
    {
      /* Remove the newlines.  */
      char *nl;
      while ((nl = strchr (*value, '\n')) != NULL)
	*nl = ' ';
    }
  return status;
}

int
_get_content_type (mu_header_t hdr, char **value, char **rest)
{
  char *type = NULL;
  _get_hdr_value (hdr, MU_HEADER_CONTENT_TYPE, &type);
  if (type == NULL || *type == '\0')
    {
      if (type)
	free (type);
      type = strdup ("text/plain"); /* Default.  */
      if (rest)
	*rest = NULL;
    }
  else
    {
      char *p = strchr (type, ';');
      if (p)
	*p++ = 0;
      if (rest)
	*rest = p;
    }
  *value = type;
  return 0;
}

static int
_get_content_encoding (mu_header_t hdr, char **value)
{
  char *encoding = NULL;
  _get_hdr_value (hdr, MU_HEADER_CONTENT_TRANSFER_ENCODING, &encoding);
  if (encoding == NULL || *encoding == '\0')
    {
      if (encoding)
	free (encoding);
      encoding = strdup ("7bit"); /* Default.  */
    }
  *value = encoding;
  return 0;
}

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_FOLDER: 
      mh_set_current_folder (arg);
      break;

    case ARG_FILE:
      input_file = arg;
      break;
      
      /* Operation mode */
    case ARG_COMPOSE:
      if (is_true (arg))
	{
	  mode = mode_compose;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOCOMPOSE:
      if (mode == mode_compose)
	mode = mode_compose;
      break;

    case ARG_LIST:
      if (is_true (arg))
	{
	  mode = mode_list;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOLIST:
      if (mode == mode_list)
	mode = mode_compose;
      break;

    case ARG_SHOW:
      if (is_true (arg))
	{
	  mode = mode_show;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOSHOW:
      if (mode == mode_show)
	mode = mode_compose;
      break;

    case ARG_STORE:
      if (is_true (arg))
	{
	  mode = mode_store;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOSTORE:
      if (mode == mode_store)
	mode = mode_compose;
      break;

      /* List options */
    case ARG_HEADER:
      if (is_true (arg))
	{
	  mode_options |= OPT_HEADERS;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOHEADERS:
      mode_options &= ~OPT_HEADERS;
      break;
      
    case ARG_REALSIZE:
      if (is_true (arg))
	{
	  mode_options |= OPT_REALSIZE;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOREALSIZE:
      mode_options &= ~OPT_REALSIZE;
      break;

      /* Display options */

    case ARG_SERIALONLY:
      mh_opt_notimpl_warning ("-[no]serialonly");
      if (is_true (arg))
	{
	  mode_options |= OPT_SERIALONLY;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOSERIALONLY:
      mode_options &= ~OPT_SERIALONLY;
      break;

    case ARG_PAUSE:
      if (is_true (arg))
	{
	  mode_options |= OPT_PAUSE;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOPAUSE:
      mode_options &= ~OPT_PAUSE;
      break;

      /* Store options */
    case ARG_AUTO:
      if (is_true (arg))
	{
	  mode_options |= OPT_AUTO;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOAUTO:
      mode_options &= ~OPT_AUTO;
      break;

    case ARG_FORM:
      formfile = arg;
      break;

      /* Common options */
    case ARG_VERBOSE:
      if (is_true (arg))
	{
	  mode_options |= OPT_VERBOSE;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOVERBOSE:
      mode_options &= ~OPT_VERBOSE;
      break;

    case ARG_TYPE:
      sfree (&content_type);
      sfree (&content_subtype);
      split_content (arg, &content_type, &content_subtype);
      break;
      
    case ARG_PART:
      req_part = msg_part_parse (arg);
      break;

    case ARG_QUIET:
      mode_options |= OPT_QUIET;
      break;
	
    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    case ARG_CHARSET:
      charset = arg;
      break;
      
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}


/* *********************** Message part functions ************************* */

struct _msg_part {
  int level;
  int maxlevel;
  size_t *part;
};

msg_part_t 
msg_part_create (size_t num)
{
  msg_part_t p = xmalloc (sizeof (*p));
  p->maxlevel = 16;
  p->part = xmalloc (sizeof (*p->part) * p->maxlevel);
  p->part[0] = num;
  p->level = 0;
  return p;
}

void
msg_part_destroy (msg_part_t p)
{
  free (p->part);
  free (p);
}

int
msg_part_eq (msg_part_t a, msg_part_t b)
{
  int i, level = a->level < b->level ? a->level : b->level;

  for (i = 1; i <= level; i++)
    if (a->part[i] != b->part[i])
      return 1;
  return 0;
}

void
msg_part_incr (msg_part_t p)
{
  if (p->level == p->maxlevel)
    {
      p->maxlevel += 16;
      p->part = xrealloc (p->part, sizeof (*p->part) * p->maxlevel);
    }
  p->level++;
}

void
msg_part_decr (msg_part_t p)
{
  if (p->level <= 0)
    abort ();
  p->level--;
}

void
msg_part_set_subpart (msg_part_t p, size_t subpart)
{
  p->part[p->level] = subpart;
}

void
msg_part_print (msg_part_t p, int max_width)
{
  int i;
  int width = 0;
  
  for (i = 1; i <= p->level; i++)
    {
      if (i > 1)
	{
	  printf (".");
	  width++;
	}
      width += printf ("%s", mu_umaxtostr (0, p->part[i]));
    }
  for (; width < max_width; width++)
    putchar (' ');
}

char *
msg_part_format (msg_part_t p)
{
  int i;
  int width = 0;
  char *str, *s;
  
  for (i = 1; i <= p->level; i++)
    {
      if (i > 1)
	width++;
      width += strlen (mu_umaxtostr (0, p->part[i]));
    }

  str = s = xmalloc (width + 1);
  for (i = 1; i <= p->level; i++)
    {
      if (i > 1)
	*s++ = '.';
      s += sprintf (s, "%s", mu_umaxtostr (0, p->part[i]));
    }
  *s = 0;
  return str;
}

void
msg_part_format_stk (struct obstack *stk, msg_part_t p)
{
  int i;
  
  for (i = 1; i <= p->level; i++)
    {
      int len;
      const char *buf;
  
      if (i > 1)
	obstack_1grow (stk, '.');

      buf = mu_umaxtostr (0, p->part[i]);
      len = strlen (buf);
      obstack_grow (stk, buf, len);
    }
}

msg_part_t
msg_part_parse (char *str)
{
  msg_part_t p = msg_part_create (0);

  do
    {
      char *endp;
      size_t num = strtoul (str, &endp, 10);
      
      switch (*endp)
	{
	case '.':
	  str = endp + 1;
	  break;
	case 0:
	  str = endp;
	  break;
	default:
	  mu_error (_("malformed part specification (near %s)"), endp);
	  exit (1);
	}
      msg_part_incr (p);
      msg_part_set_subpart (p, num);
    }
  while (*str);
  return p;
}

int
msg_part_level (msg_part_t p)
{
  return p->level;
}

size_t
msg_part_subpart (msg_part_t p, int level)
{
  if (level <= p->level)
    return p->part[level];
  return 0;
}


/* *********************** Context file accessors ************************* */

const char *
_mhn_profile_get (char *prefix, char *type, char *subtype, char *defval)
{
  char *name;
  const char *str;
  
  if (subtype)
    {
      asprintf (&name, "mhn-%s-%s/%s", prefix, type, subtype);
      str = mh_global_profile_get (name, NULL);
      free (name);
      if (!str)
	return _mhn_profile_get (prefix, type, NULL, defval);
    }
  else
    {
      asprintf (&name, "mhn-%s-%s", prefix, type);
      str = mh_global_profile_get (name, defval);
      free (name);
    }
  return str;
}

char *
mhn_compose_command (char *typestr, int *flags, char *file)
{
  const char *p, *str;
  char *type, *subtype, *typeargs;
  struct obstack stk;

  split_content (typestr, &type, &subtype);
  str = _mhn_profile_get ("compose", type, subtype, NULL);
  if (!str) 
    return NULL;
  
  /* Expand macro-notations:
    %a  additional arguments
    %f  filename containing content
    %F  %f, and stdout is not redirected
    %s  subtype */

  obstack_init (&stk);
  p = mu_str_skip_class (str, MU_CTYPE_SPACE);
  
  if (*p == '|')
    p++;

  for ( ; *p; p++)
    {
      if (*p == '%')
	{
	  switch (*++p)
	    {
	    case 'a':
	      /* additional arguments */
	      obstack_grow (&stk, typeargs, strlen (typeargs));
	      break;
	      
	    case 'F':
	      /* %f, and stdout is not redirected */
	      *flags |= MHN_STDIN;
	      /*FALLTHRU*/
	    case 'f':
	      obstack_grow (&stk, file, strlen (file));
	      break;
	      
	    case 's':
	      /* subtype */
	      obstack_grow (&stk, subtype, strlen (subtype));
	      break;
	      
	    default:
	      obstack_1grow (&stk, *p);
	      p++;
	    }
	}
      else
	obstack_1grow (&stk, *p);
    }
  obstack_1grow (&stk, 0);

  free (type);
  free (subtype);

  str = obstack_finish (&stk);
  p = mu_str_skip_class (str, MU_CTYPE_SPACE);
  if (!*p)
    str = NULL;
  else
    str = strdup (p);

  obstack_free (&stk, NULL);
  return (char*) str;
}

char *
mhn_show_command (mu_message_t msg, msg_part_t part, int *flags,
		  char **tempfile)
{
  const char *p, *str;
  char *tmp;
  char *typestr, *type, *subtype, *typeargs;
  struct obstack stk;
  mu_header_t hdr;

  mu_message_get_header (msg, &hdr);
  _get_content_type (hdr, &typestr, &typeargs);
  split_content (typestr, &type, &subtype);
  str = _mhn_profile_get ("show", type, subtype, NULL);
  if (!str) /* FIXME */
    return NULL;
  
  /* Expand macro-notations:
    %a  additional arguments
    %e  exclusive execution
    %f  filename containing content
    %F  %e, %f, and stdin is terminal not content
    %l  display listing prior to displaying content
    %p  %l, and ask for confirmation
    %s  subtype
    %d  content description */

  obstack_init (&stk);
  p = mu_str_skip_class (str, MU_CTYPE_SPACE);
  
  if (*p == '|')
    p++;

  for ( ; *p; p++)
    {
      if (*p == '%')
	{
	  switch (*++p)
	    {
	    case 'a':
	      /* additional arguments */
	      obstack_grow (&stk, typeargs, strlen (typeargs));
	      break;
	      
	    case 'e':
	      /* exclusive execution */
	      *flags |= MHN_EXCLUSIVE_EXEC;
	      break;
	      
	    case 'f':
	      /* filename containing content */
	      if (!*tempfile)
		*tempfile = mu_tempname (NULL);
	      obstack_grow (&stk, *tempfile, strlen (*tempfile));
	      break;
				     
	    case 'F':
	      /* %e, %f, and stdin is terminal not content */
	      *flags |= MHN_STDIN|MHN_EXCLUSIVE_EXEC;
	      if (!*tempfile)
		*tempfile = mu_tempname (NULL);
	      obstack_grow (&stk, *tempfile, strlen (*tempfile));
	      break;
	      
	    case 'l':
	      /* display listing prior to displaying content */
	      *flags |= MHN_LISTING;
	      break;
	      
	    case 'p':
	      /* %l, and ask for confirmation */
	      *flags |= MHN_LISTING|MHN_CONFIRM;
	      break;
	      
	    case 's':
	      /* subtype */
	      obstack_grow (&stk, subtype, strlen (subtype));
	      break;
	      
	    case 'd':
	      /* content description */
	      if (mu_header_aget_value (hdr, MU_HEADER_CONTENT_DESCRIPTION,
				     &tmp) == 0)
		{
		  obstack_grow (&stk, tmp, strlen (tmp));
		  free (tmp);
		}
	      break;
	      
	    default:
	      obstack_1grow (&stk, *p);
	      p++;
	    }
	}
      else
	obstack_1grow (&stk, *p);
    }
  obstack_1grow (&stk, 0);

  free (typestr);
  free (type);
  free (subtype);

  str = obstack_finish (&stk);
  p = mu_str_skip_class (str, MU_CTYPE_SPACE);
  if (!*p)
    str = NULL;
  else
    str = strdup (p);

  obstack_free (&stk, NULL);
  return (char*) str;
}

char *
mhn_store_command (mu_message_t msg, msg_part_t part, char *name)
{
  const char *p, *str;
  char *tmp;
  char *typestr, *type, *subtype, *typeargs;
  struct obstack stk;
  mu_header_t hdr;
  
  mu_message_get_header (msg, &hdr);
  _get_content_type (hdr, &typestr, &typeargs);
  split_content (typestr, &type, &subtype);
  str = _mhn_profile_get ("store", type, subtype, "%m%P.%s");
  
  /* Expand macro-notations:
     %m  message number
     %P  .part
     %p  part
     %s  subtype */

  obstack_init (&stk);
  for (p = str; *p; p++)
    {
      if (*p == '%')
	{
	  switch (*++p)
	    {
	    case 'a':
	      /* additional arguments */
	      obstack_grow (&stk, typeargs, strlen (typeargs));
	      break;

	    case 'm':
	      if (name)
		obstack_grow (&stk, name, strlen (name));
	      else
		{
                  const char *buf = mu_umaxtostr (0, msg_part_subpart (part, 0));
		  obstack_grow (&stk, buf, strlen (buf));
		}
	      break;

	    case 'P':
	      if (msg_part_level (part) >= 1)
		obstack_1grow (&stk, '.');
	      /*FALLTHRU*/
	    case 'p':
	      if (msg_part_level (part) >= 1)
		msg_part_format_stk (&stk, part);
	      break;
	      
	    case 's':
	      /* subtype */
	      obstack_grow (&stk, subtype, strlen (subtype));
	      break;
	      
	    case 'd':
	      /* content description */
	      if (mu_header_aget_value (hdr, MU_HEADER_CONTENT_DESCRIPTION,
				        &tmp) == 0)
		{
		  obstack_grow (&stk, tmp, strlen (tmp));
		  free (tmp);
		}
	      break;
	      
	    default:
	      obstack_1grow (&stk, *p);
	      p++;
	    }
	}
      else
	obstack_1grow (&stk, *p);
    }
  obstack_1grow (&stk, *p);

  free (typestr);
  free (type);
  free (subtype);
  
  str = obstack_finish (&stk);
  p = mu_str_skip_class (str, MU_CTYPE_SPACE);
  if (!*p)
    str = NULL;
  else
    str = strdup (p);
  
  obstack_free (&stk, NULL);
  return (char*) str;
}


/* ************************* Auxiliary functions ************************** */

static void
split_args (char *argstr, int *pargc, char ***pargv)
{
  int argc;
  char **argv;

  if (mu_argcv_get (argstr, ";", NULL, &argc, &argv) == 0)
    {
      int i, j;
      
      for (i = j = 0; i < argc; i++)
	{
	  if (argv[i][0] != ';')
	    argv[j++] = argv[i];
	}
      argv[j] = NULL;
      *pargc = j;
      *pargv = argv;
    }
  else
    {
      *pargc = 0;
      *pargv = NULL;
    }
}

int
_message_is_external_body (mu_message_t msg, char ***env)
{
  int rc;
  mu_header_t hdr;
  char *typestr, *argstr, *type, *subtype;

  if (mu_message_get_header (msg, &hdr))
    return 0;
  _get_content_type (hdr, &typestr, &argstr);
  split_content (typestr, &type, &subtype);
  rc = subtype && strcmp (subtype, "external-body") == 0;
  if (rc && env)
    {
      int c;
      split_args (argstr, &c, env);
    }

  free (typestr);
  free (type);
  free (subtype);
  return rc;
}

char *
_get_env (char **env, char *name)
{
  int nlen = strlen (name);
  for (; *env; env++)
    {
      int len = strlen (*env);
      if (nlen < len
	  && (*env)[len+1] == '='
	  && mu_c_strncasecmp (*env, name, nlen) == 0)
	return *env + len + 1;
    }
  return NULL;
}

void
_free_env (char **env)
{
  char **p;

  for (p = env; *p; p++)
    free (*p);
  free (env);
}

int
get_extbody_params (mu_message_t msg, char **content, char **descr)
{
  int rc = 0;
  mu_body_t body = NULL;
  mu_stream_t stream = NULL;
  char buf[128];
  size_t n;
	
  mu_message_get_body (msg, &body);
  mu_body_get_stream (body, &stream);
  mu_stream_seek (stream, 0, SEEK_SET);

  while (rc == 0
	 && mu_stream_sequential_readline (stream, buf, sizeof buf, &n) == 0
	 && n > 0)
    {
      char *p;
      int len = strlen (buf);

      if (len > 0 && buf[len-1] == '\n')
	buf[len-1] = 0;

      if (descr
	  && mu_c_strncasecmp (buf, MU_HEADER_CONTENT_DESCRIPTION ":",
			       sizeof (MU_HEADER_CONTENT_DESCRIPTION)) == 0)
	{
	  p = mu_str_skip_class (buf + sizeof (MU_HEADER_CONTENT_DESCRIPTION),
				 MU_CTYPE_SPACE);
	  *descr = strdup (p);
	}
      else if (content
	       && mu_c_strncasecmp (buf, MU_HEADER_CONTENT_TYPE ":",
			            sizeof (MU_HEADER_CONTENT_TYPE)) == 0)
	{
	  char *q;
	  p = mu_str_skip_class (buf + sizeof (MU_HEADER_CONTENT_TYPE),
				 MU_CTYPE_SPACE);
	  q = strchr (p, ';');
	  if (q)
	    *q = 0;
	  *content = strdup (p);
	}
    }
  return 0;
}


/* ************************** Message iterators *************************** */


typedef int (*msg_handler_t) (mu_message_t msg, msg_part_t part,
			      char *type, char *encoding,
			      void *data);

int
match_content (char *content)
{
  int rc;
  char *type, *subtype;

  if (!content_type && !content_subtype)
    return 0;
  if (!content)
    return 0;

  split_content (content, &type, &subtype);

  if ((rc = mu_c_strcasecmp (content_type, type)) == 0)
    {
      if (content_subtype)
	rc = mu_c_strcasecmp (content_subtype, subtype);
    }
  else 
    rc = mu_c_strcasecmp (content_type, subtype);

  free (type);
  free (subtype);
  return rc;
}

int
call_handler (mu_message_t msg, msg_part_t part, char *type, char *encoding,
	      msg_handler_t fun, void *data)
{
  if (req_part && msg_part_eq (req_part, part))
    return 0;
  if (match_content (type))
    return 0;
  return fun (msg, part, type, encoding, data);
}

int
handle_message (mu_message_t msg, msg_part_t part, msg_handler_t fun, void *data)
{
  mu_header_t hdr;
  char *type = NULL;
  char *encoding;
  int ismime = 0;
  
  mu_message_get_header (msg, &hdr);
  _get_content_type (hdr, &type, NULL);
  _get_content_encoding (hdr, &encoding);

  fun (msg, part, type, encoding, data);
  sfree (&type);
  sfree (&encoding);
  
  mu_message_is_multipart (msg, &ismime);
  if (ismime)
    {
      size_t i, nparts;

      mu_message_get_num_parts (msg, &nparts);

      msg_part_incr (part);
      for (i = 1; i <= nparts; i++)
	{
	  mu_message_t message = NULL;
	  
	  if (mu_message_get_part (msg, i, &message) == 0)
	    {
	      mu_message_get_header (message, &hdr);
	      _get_content_type (hdr, &type, NULL);
	      _get_content_encoding (hdr, &encoding);

	      msg_part_set_subpart (part, i);

	      if (mu_message_is_multipart (message, &ismime) == 0 && ismime)
		handle_message (message, part, fun, data);
	      else
		call_handler (message, part, type, encoding, fun, data);
	      sfree (&type);
	      sfree (&encoding);
	    }
	}
      msg_part_decr (part);
    }
  return 0;
}

int
mhn_message_size (mu_message_t msg, size_t *psize)
{
  mu_body_t body;
  *psize = 0;
  mu_message_get_body (msg, &body);
  if (mode_options & OPT_REALSIZE)
    {
      mu_stream_t dstr = NULL, bstr = NULL;

      if (mu_body_get_stream (body, &bstr) == 0)
	{
	  mu_header_t hdr;
	  char *encoding;
	  size_t size = 0;
	  int rc;
	  
	  mu_message_get_header (msg, &hdr);
	  _get_content_encoding (hdr, &encoding);

	  rc = mu_filter_create (&dstr, bstr, encoding,
			 	 MU_FILTER_DECODE, 
				 MU_STREAM_READ | MU_STREAM_NO_CLOSE);
	  free (encoding);
	  if (rc == 0)
	    {
	      char buf[512];
	      size_t n;
	      
	      while (mu_stream_sequential_read (dstr, buf, sizeof buf, &n) == 0
		     && n > 0)
		size += n;

	      mu_stream_destroy (&dstr, NULL);
	      *psize = size;
	      return 0;
	    }
	}
    }

  return mu_body_size (body, psize);
}


/* ***************************** List Mode ******************************* */

int
list_handler (mu_message_t msg, msg_part_t part, char *type, char *encoding,
	      void *data)
{
  size_t size;
  mu_header_t hdr;
  
  if (msg_part_level (part) == 0)
    printf ("%4lu      ", (unsigned long) msg_part_subpart (part, 0));
  else
    {
      printf ("     ");
      msg_part_print (part, 4);
      putchar (' ');
    }
  printf ("%-25s", type);
  
  mhn_message_size (msg, &size);
  if (size < 1024)
    printf (" %4lu", (unsigned long) size);
  else if (size < 1024*1024)
    printf ("%4luK", (unsigned long) size / 1024);
  else
    printf ("%4luM", (unsigned long) size / 1024 / 1024);
  
  if (mu_message_get_header (msg, &hdr) == 0)
    {
      char *descr;
      if (mu_header_aget_value (hdr, "Content-Description", &descr) == 0)
	{
	  printf (" %s", descr);
	  free (descr);
	}
    }
  
  printf ("\n");

  if (_message_is_external_body (msg, NULL))
    {
      char *content_type = NULL;
      char *content_descr = NULL;

      get_extbody_params (msg, &content_type, &content_descr);
	
      printf ("          ");
      printf ("%-25s", content_type ? content_type : "");
      if (content_descr)
	printf ("       %s", content_descr);
      printf ("\n");
      
      free (content_type);
      free (content_descr);
    }
  
  return 0;
}

int
list_message (mu_message_t msg, size_t num)
{
  size_t uid;
  msg_part_t part;

  mh_message_number (msg, &uid);
  part = msg_part_create (uid);
  handle_message (msg, part, list_handler, NULL);
  msg_part_destroy (part);
  return 0;
}

void
list_iterator (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  list_message (msg, num);
}

int
mhn_list ()
{
  int rc; 

  if (mode_options & OPT_HEADERS)
    printf (_(" msg part type/subtype              size  description\n"));

  if (message)
    rc = list_message (message, 0);
  else
    rc = mh_iterate (mbox, &msgset, list_iterator, NULL);
  return rc;
}


/* ***************************** Show Mode ******************************* */

static mu_list_t mhl_format;

void
cat_message (mu_stream_t out, mu_stream_t in)
{
  int rc = 0;
  char buf[512];
  size_t n;

  mu_stream_seek (in, 0, SEEK_SET);
  while (rc == 0
	 && mu_stream_sequential_read (in, buf, sizeof buf, &n) == 0
	 && n > 0)
    rc = mu_stream_sequential_write (out, buf, n);
}

int
show_internal (mu_message_t msg, msg_part_t part, char *encoding, mu_stream_t out)
{
  int rc;
  mu_body_t body = NULL;
  mu_stream_t dstr, bstr;

  if ((rc = mu_message_get_body (msg, &body)))
    {
      mu_error (_("%s: cannot get message body: %s"),
		mu_umaxtostr (0, msg_part_subpart (part, 0)),
		mu_strerror (rc));
      return 0;
    }
  mu_body_get_stream (body, &bstr);
  rc = mu_filter_create (&dstr, bstr, encoding,
		         MU_FILTER_DECODE, MU_STREAM_READ | MU_STREAM_NO_CLOSE);
  if (rc == 0)
    bstr = dstr;
  cat_message (out, bstr);
  if (dstr)
    mu_stream_destroy (&dstr, mu_stream_get_owner (dstr));
  return 0;
}
  
int
mhn_exec (mu_stream_t *str, const char *cmd, int flags)
{
  int rc = mu_prog_stream_create (str, cmd, MU_STREAM_WRITE);
  if (rc)
    {
      mu_error (_("cannot create proc stream (command %s): %s"),
		cmd, mu_strerror (rc));
    }
  else
    {
      rc = mu_stream_open (*str);
      if (rc)
	mu_error (_("cannot open proc stream (command %s): %s"),
		  cmd, mu_strerror (rc));
    }
  return rc;
}

int
exec_internal (mu_message_t msg, msg_part_t part, char *encoding,
	       const char *cmd, int flags)
{
  int rc;
  mu_stream_t tmp;

  if ((rc = mhn_exec (&tmp, cmd, flags)))
    return rc;
  show_internal (msg, part, encoding, tmp);      
  mu_stream_destroy (&tmp, mu_stream_get_owner (tmp));
  return rc;
}

/* FIXME: stdin is always opened when invoking subprocess, no matter
   what the MHN_STDIN bit is */
   
int 
mhn_run_command (mu_message_t msg, msg_part_t part,
		 char *encoding,
		 char *cmd, int flags,
		 char *tempfile)
{
  int rc = 0;
  
  if (tempfile)
    {
      /* pass content via a tempfile */
      int status;
      int argc;
      char **argv;
      mu_stream_t tmp;
      
      if (mu_argcv_get (cmd, "", "#", &argc, &argv))
	{
	  mu_error (_("cannot parse command line `%s'"), cmd);
	  return ENOSYS;
	}

      rc = mu_file_stream_create (&tmp, tempfile, MU_STREAM_RDWR);
      if (rc)
	{
	  mu_error (_("cannot create temporary stream (file %s): %s"),
		    tempfile, mu_strerror (rc));
	  mu_argcv_free (argc, argv);
	  return rc;
	}
      rc = mu_stream_open (tmp);
      if (rc)
	{
	  mu_error (_("cannot open temporary stream (file %s): %s"),
		    tempfile, mu_strerror (rc));
	  mu_stream_destroy (&tmp, mu_stream_get_owner (tmp));
	  mu_argcv_free (argc, argv);
	  return rc;
	}
      show_internal (msg, part, encoding, tmp);      
      mu_stream_destroy (&tmp, mu_stream_get_owner (tmp));
      rc = mu_spawnvp (argv[0], argv, &status);
      if (status)
	rc = status;
      mu_argcv_free (argc, argv);
    }
  else
    rc = exec_internal (msg, part, encoding, cmd, flags);

  return rc;
}

int
show_handler (mu_message_t msg, msg_part_t part, char *type, char *encoding,
	      void *data)
{
  mu_stream_t out = data;
  char *cmd;
  int flags = 0;
  int fd = 1;
  char *tempfile = NULL;
  int ismime;
  mu_transport_t trans;
 
  if (mu_message_is_multipart (msg, &ismime) == 0 && ismime)
    return 0;

  mu_stream_get_transport (out, &trans);
  fd = (int) trans; /* FIXME */

  if (mode_options & OPT_PAUSE)
    flags |= MHN_CONFIRM;
  cmd = mhn_show_command (msg, part, &flags, &tempfile);
  if (!cmd)
    flags |= MHN_LISTING;
  if (flags & MHN_LISTING)
    {
      char *str;
      const char *p;
      size_t size = 0;

      str = (char*) _("part ");
      mu_stream_sequential_write (out, str, strlen (str));
      str = msg_part_format (part);
      mu_stream_sequential_write (out, str, strlen (str));
      free (str);
      mu_stream_sequential_write (out, " ", 1);
      mu_stream_sequential_write (out, type, strlen (type));
      mhn_message_size (msg, &size);
      p = mu_umaxtostr (0, size);
      mu_stream_sequential_write (out, p, strlen (p));
      mu_stream_sequential_write (out, "\n", 1);
      mu_stream_flush (out);
    }

  if (flags & MHN_CONFIRM)
    {
      if (isatty (fd) && isatty (0))
	{
          char buf[64];
	  printf (_("Press <return> to show content..."));
	  if (!fgets (buf, sizeof buf, stdin) || buf[0] != '\n')
	    return 0;
	}
    } 
      
  if (!cmd)
    {
      const char *pager = mh_global_profile_get ("moreproc", getenv ("PAGER"));
      if (pager)
	exec_internal (msg, part, encoding, pager, 0);
      else
	show_internal (msg, part, encoding, out);
    }
  else
    {
      mhn_run_command (msg, part, encoding, cmd, flags, tempfile);
      free (cmd);
      unlink (tempfile);
      free (tempfile);
    }

  return 0;
}

int
show_message (mu_message_t msg, size_t num, void *data)
{
  msg_part_t part = msg_part_create (num);

  if (mhl_format)
    mhl_format_run (mhl_format, width, 0, MHL_DISABLE_BODY, msg,
		    (mu_stream_t) data);

  handle_message (msg, part, show_handler, data);
  msg_part_destroy (part);
  return 0;
}

void
show_iterator (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  msg_part_t part;
  
  mh_message_number (msg, &num);
  part = msg_part_create (num);
  show_message (msg, num, data);
  msg_part_destroy (part);
}

int
mhn_show ()
{
  int rc;
  mu_stream_t ostr;
  
  rc = mu_stdio_stream_create (&ostr, stdout, 0);
  if (rc)
    {
      mu_error (_("cannot create output stream: %s"), mu_strerror (rc));
      exit (1);
    }
  rc = mu_stream_open (ostr);
  if (rc)
    {
      mu_error (_("cannot open output stream: %s"), mu_strerror (rc));
      exit (1);
    }
  
  if (formfile)
    {
      char *s = mh_expand_name (MHLIBDIR, "mhl.headers", 0);
      if (access (s, R_OK) == 0)
	formfile = "mhl.headers";
      free (s);
    }

  if (formfile)
    {
      char *s = mh_expand_name (MHLIBDIR, formfile, 0);
      mhl_format = mhl_format_compile (s);
      if (!mhl_format)
	exit (1);
      free (s);
    }
    
  if (message)
    rc = show_message (message, 0, ostr);
  else
    rc = mh_iterate (mbox, &msgset, show_iterator, ostr);
  return rc;
}

/* ***************************** Store Mode ****************************** */

char *
normalize_path (const char *cwd, char *path)
{
  int len;
  char *p;
  char *pcwd = NULL;
  
  if (!path)
    return path;

  if (path[0] == '/')
    return NULL;

  if (!cwd)
    cwd = pcwd = mu_getcwd ();

  len = strlen (cwd) + strlen (path) + 2;
  p = xmalloc (len);
  sprintf (p, "%s/%s", cwd, path);
  path = p;

  /* delete trailing delimiter if any */
  if (len && path[len-1] == '/')
    path[len-1] = 0;

  /* Eliminate any /../ */
  for (p = strchr (path, '.'); p; p = strchr (p, '.'))
    {
      if (p > path && p[-1] == '/')
	{
	  if (p[1] == '.' && (p[2] == 0 || p[2] == '/'))
	    /* found */
	    {
	      char *q, *s;

	      /* Find previous delimiter */
	      for (q = p-2; *q != '/' && q >= path; q--)
		;

	      if (q < path)
		break;
	      /* Copy stuff */
	      s = p + 2;
	      p = q;
	      while ((*q++ = *s++))
		;
	      continue;
	    }
	}

      p++;
    }

  if (path[0] == 0)
    {
      path[0] = '/';
      path[1] = 0;
    }

  len = strlen (cwd);
  if (strlen (path) < len || memcmp (path, cwd, len))
    sfree (&path);
  free (pcwd);
  return path;
}

int
store_handler (mu_message_t msg, msg_part_t part, char *type, char *encoding,
	       void *data)
{
  char *prefix = data;
  char *name = NULL;
  char *tmp;
  int ismime;
  int rc;
  mu_stream_t out;
  const char *dir = mh_global_profile_get ("mhn-storage", NULL);
  
  if (mu_message_is_multipart (msg, &ismime) == 0 && ismime)
    return 0;
  
  if (mode_options & OPT_AUTO)
    {
      char *val;
      int rc = mu_message_aget_decoded_attachment_name (msg, charset,
							&val, NULL);
      if (rc == 0)
	{
	  name = normalize_path (dir, val);
	  free (val);
	}
      else if (rc != MU_ERR_NOENT)
	{
	  char *pstr = msg_part_format (part);
	  mu_diag_output (MU_DIAG_WARNING,
			  _("%s: cannot decode attachment name: %s"),
			  pstr, mu_strerror (rc));
	  free (pstr);
	}
    }
  
  if (!name)
    {
      char *fname = mhn_store_command (msg, part, prefix);
      if (dir)
	asprintf (&name, "%s/%s", dir, fname);
      else
	name = fname;
    }
  
  tmp = msg_part_format (part);
  if (prefix)
    printf (_("storing message %s part %s as file %s\n"),
	    prefix,
	    tmp,
	    name);
  else
    printf (_("storing message %s part %s as file %s\n"),
	    mu_umaxtostr (0, msg_part_subpart (part, 0)),
	    tmp,
	    name);
  free (tmp);

  if (!(mode_options & OPT_QUIET) && access (name, R_OK) == 0)
    {
      char *p;
      int rc;
	
      asprintf (&p, _("File %s already exists. Rewrite"), name);
      rc = mh_getyn (p);
      free (p);
      if (!rc)
	{
	  free (name);
	  return 0;
	}
      unlink (name);
    }
  
  rc = mu_file_stream_create (&out, name, MU_STREAM_WRITE|MU_STREAM_CREAT);
  if (rc)
    {
      mu_error (_("cannot create output stream (file %s): %s"),
		name, mu_strerror (rc));
      free (name);
      return rc;
    }
  rc = mu_stream_open (out);
  if (rc)
    {
      mu_error (_("cannot open output stream (file %s): %s"),
		name, mu_strerror (rc));
      free (name);
      mu_stream_destroy (&out, mu_stream_get_owner (out));
      return rc;
    }
  show_internal (msg, part, encoding, out);      

  mu_stream_destroy (&out, mu_stream_get_owner (out));
  free (name);
  
  return 0;
}

void
store_message (mu_message_t msg, void *data)
{
  size_t uid;
  msg_part_t part;

  mh_message_number (msg, &uid);
  part = msg_part_create (uid);
  handle_message (msg, part, store_handler, data);
  msg_part_destroy (part);
}

void
store_iterator (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  store_message (msg, data);
}

int
mhn_store ()
{
  int rc = 0;
  
  if (message)
    {
      char *p = strrchr (input_file, '/');
      if (p)
	p++;
      else
	p = input_file;
      store_message (message, p);
    }
  else
    rc = mh_iterate (mbox, &msgset, store_iterator, NULL);
  return rc;
}


/* ***************************** Compose Mode **************************** */

int
stream_getline (mu_stream_t str, char **buf, size_t *bufsize, size_t *pnum)
{
  int rc;
  size_t numread, n;

  if (!*buf)
    {
      *bufsize = 128;
      *buf = xmalloc (*bufsize);
    }
  numread = 0;
  while ((rc = mu_stream_sequential_readline (str, *buf + numread,
					   *bufsize - numread, &n)) == 0
	 && n > 0)
    {
      numread += n;
      if ((*buf)[numread - 1] != '\n')
	{
	  if (numread + 1 == *bufsize)
	    {
	      *bufsize += 128;
	      *buf = xrealloc (*buf, *bufsize);
	    }
	  continue;
	}
      break;
    }
  if (pnum)
    *pnum = numread;
  return rc;
}

struct compose_env {
  mu_stream_t input;
  mu_mime_t mime;
  size_t line;
  int subpart;
};

size_t
mhn_error_loc (struct compose_env *env)
{
  mu_header_t hdr = NULL;
  size_t n = 0;
  mu_message_get_header (message, &hdr);
  mu_header_lines (hdr, &n);
  return n + 1 + env->line;
}

static int
parse_brace (char **pval, char **cmd, int c, struct compose_env *env)
{
  char *val;
  int len;
  char *rest = *cmd;
  char *sp = strchr (rest, c);
  
  if (!sp)
    {
      mu_error (_("%s:%lu: missing %c"),
		input_file,
		(unsigned long) mhn_error_loc (env),
		c);
      return 1;
    }
  len = sp - rest;
  val = xmalloc (len + 1);
  memcpy (val, rest, len);
  val[len] = 0;
  *cmd = sp + 1;
  *pval = val;
  return 0;
}

#define isdelim(c) (mu_isspace (c) || strchr (";<[(", c))
#define skipws(ptr) (ptr) = mu_str_skip_class (ptr, MU_CTYPE_SPACE)

int
parse_content_type (struct compose_env *env,
		    struct obstack *stk, char **prest, char **id, char **descr)
{
  int status = 0, stop = 0;
  char *rest = *prest;
  char *sp;
  char *comment = NULL;

  while (stop == 0 && status == 0 && *rest)
    {
      skipws (rest);
      switch (*rest++)
	{
	case '(':
	  if (comment)
	    {
	      mu_error (_("%s:%lu: comment redefined"),
			input_file,
			(unsigned long) mhn_error_loc (env));
	      status = 1;
	      break;
	    }
	  status = parse_brace (&comment, &rest, ')', env);
	  break;

	case '[':
	  if (!descr)
	    {
	      mu_error (_("%s:%lu: syntax error"),
			input_file,
			(unsigned long) mhn_error_loc (env));
	      status = 1;
	      break;
	    }
	    
	  if (*descr)
	    {
	      mu_error (_("%s:%lu: description redefined"),
			input_file,
			(unsigned long) mhn_error_loc (env));
	      status = 1;
	      break;
	    }
	  status = parse_brace (descr, &rest, ']', env);
	  break;
	  
	case '<':
	  if (*id)
	    {
	      mu_error (_("%s:%lu: content id redefined"),
			input_file,
			(unsigned long) mhn_error_loc (env));
	      status = 1;
	      break;
	    }
	  status = parse_brace (id, &rest, '>', env);
	  break;

	case ';':
	  obstack_1grow (stk, ';');
	  obstack_1grow (stk, ' ');
	  skipws (rest);
	  sp = rest;
	  for (; *rest && !mu_isspace (*rest) && *rest != '='; rest++)
	    obstack_1grow (stk, *rest);
	  skipws (rest);
	  if (*rest != '=')
	    {
	      mu_error (_("%s:%lu: syntax error"),
			input_file,
			(unsigned long) mhn_error_loc (env));
	      status = 1;
	      break;
	    }
	  rest++;
	  obstack_1grow (stk, '=');
	  skipws (rest);
	  for (; *rest; rest++)
	    {
	      if (isdelim (*rest))
		break;
	      obstack_1grow (stk, *rest);
	    }
	  break;
	    
	default:
	  rest--;
	  stop = 1;
	  break;
	}
    }

  if (comment)
    {
      obstack_grow (stk, " (", 2);
      obstack_grow (stk, comment, strlen (comment));
      obstack_1grow (stk, ')');
      free (comment);
    }
  *prest = rest;
  return status;
}

/* cmd is:  type "/" subtype
            0*(";" attribute "=" value)
            [ "(" comment ")" ]
            [ "<" id ">" ]
            [ "[" description "]" ]
*/
int
parse_type_command (char **pcmd, struct compose_env *env, mu_header_t hdr)
{
  int status = 0;
  char *sp, c;
  char *type = NULL;
  char *subtype = NULL;
  char *descr = NULL, *id = NULL;
  struct obstack stk;
  char *rest = *pcmd;
  
  skipws (rest);
  for (sp = rest; *sp && !isdelim (*sp); sp++)
    ;
  c = *sp;
  *sp = 0;
  split_content (rest, &type, &subtype);
  *sp = c;
  rest = sp;
  
  if (!subtype)
    {
      mu_error (_("%s:%lu: missing subtype"),
		input_file,
		(unsigned long) mhn_error_loc (env));
      return 1;
    }

  obstack_init (&stk);
  obstack_grow (&stk, type, strlen (type));
  obstack_1grow (&stk, '/');
  obstack_grow (&stk, subtype, strlen (subtype));
  status = parse_content_type (env, &stk, &rest, &id, &descr);
  obstack_1grow (&stk, 0);
  
  mu_header_set_value (hdr, MU_HEADER_CONTENT_TYPE, obstack_finish (&stk), 1);
  obstack_free (&stk, NULL);

  if (!id)
    id = mh_create_message_id (env->subpart);

  mu_header_set_value (hdr, MU_HEADER_CONTENT_ID, id, 1);
  free (id);

  if (descr)
    {
      mu_header_set_value (hdr, MU_HEADER_CONTENT_DESCRIPTION, descr, 1);
      free (descr);
    }
      
  *pcmd = rest;
  return status;
}

void
copy_header (mu_message_t msg, mu_header_t out)
{
  size_t i, count;
  mu_header_t hdr;

  mu_message_get_header (msg, &hdr);
  mu_header_get_field_count (hdr, &count);
  for (i = 1; i <= count; i++)
    {
      char *name, *value;

      if (mu_header_aget_field_name (hdr, i, &name))
        continue;

      if (mu_header_aget_field_value (hdr, i, &value))
	{
          free (name);
          continue;
        }

      mu_header_set_value (out, name, value, 0);
      free (name);
      free (value);
    }
}

void
copy_header_to_stream (mu_message_t msg, mu_stream_t stream)
{
  mu_header_t hdr;
  mu_stream_t in;
  char *buf = NULL;
  size_t bufsize = 0, n = 0;
  
  mu_message_get_header (msg, &hdr);
  mu_header_get_stream (hdr, &in);
  mu_stream_seek (in, 0, SEEK_SET);
  while (stream_getline (in, &buf, &bufsize, &n) == 0 && n > 0)
    {
      if (n == 1 && buf[0] == '\n')
	break;
      mu_stream_sequential_write (stream, buf, n);
    }
  free (buf);
}

void
finish_msg (struct compose_env *env, mu_message_t *msg)
{
  mu_header_t hdr;

  if (!msg || !*msg)
    return;
  mu_message_get_header (*msg, &hdr);
  if (mu_header_get_value (hdr, MU_HEADER_CONTENT_TYPE, NULL, 0, NULL))
    mu_header_set_value (hdr, MU_HEADER_CONTENT_TYPE, "text/plain", 1);
  if (mu_header_get_value (hdr, MU_HEADER_CONTENT_ID, NULL, 0, NULL))
    {
      char *p = mh_create_message_id (env->subpart);
      mu_header_set_value (hdr, MU_HEADER_CONTENT_ID, p, 1);
      free (p);
    }
  mu_mime_add_part (env->mime, *msg);
  mu_message_unref (*msg);
  *msg = NULL;
}

void
finish_text_msg (struct compose_env *env, mu_message_t *msg, int ascii)
{
  if (!ascii)
    {
      int rc;
      mu_message_t newmsg;
      mu_header_t hdr;
      mu_body_t body;
      mu_stream_t input, output, fstr;
      
      mu_message_create (&newmsg, NULL);
      mu_message_get_header (newmsg, &hdr);
      mu_header_get_stream (hdr, &output);
      copy_header (*msg, hdr);
      mu_header_set_value (hdr, MU_HEADER_CONTENT_TRANSFER_ENCODING,
			   "quoted-printable", 0);
      
      mu_message_get_body (newmsg, &body);
      mu_body_get_stream (body, &output);
      mu_stream_seek (output, 0, SEEK_SET);

      mu_message_get_body (*msg, &body);
      mu_body_get_stream (body, &input);
      rc = mu_filter_create (&fstr, input, "quoted-printable",
			     MU_FILTER_ENCODE, 
			     MU_STREAM_READ | MU_STREAM_NO_CLOSE);
      if (rc == 0)
	{
	  cat_message (output, fstr);
	  mu_stream_destroy (&fstr, NULL);
	  mu_message_unref (*msg);
	  *msg = newmsg;
	}
      else
	mu_message_destroy (&newmsg, NULL);
    }
  finish_msg (env, msg);
}

#define EXTCONTENT "message/external-body"

int
edit_extern (char *cmd, struct compose_env *env, mu_message_t *msg, int level)
{
  int rc;
  char *rest;
  char *id = NULL;
  mu_header_t hdr, hdr2;
  mu_body_t body;
  mu_stream_t in, out = NULL;
  struct obstack stk;

  if (!*msg)
    mu_message_create (msg, NULL);
  
  if ((rc = mu_header_create (&hdr2, NULL, 0, NULL)) != 0)
    {
      mu_error (_("cannot create header: %s"),
		mu_strerror (rc));
      return 1;
    }

  rest = cmd;
  rc = parse_type_command (&rest, env, hdr2);

  mu_message_get_header (*msg, &hdr);

  obstack_init (&stk);
  obstack_grow (&stk, EXTCONTENT, sizeof (EXTCONTENT) - 1);
  *--rest = ';'; /* FIXME */
  rc = parse_content_type (env, &stk, &rest, &id, NULL);
  obstack_1grow (&stk, 0);
  mu_header_set_value (hdr, MU_HEADER_CONTENT_TYPE, obstack_finish (&stk), 1);
  obstack_free (&stk, NULL);
  if (rc)
    return 1;

  mu_message_get_body (*msg, &body);
  mu_body_get_stream (body, &out);
  mu_stream_seek (out, 0, SEEK_SET);

  if (!id)
    id = mh_create_message_id (env->subpart);
  mu_header_set_value (hdr2, MU_HEADER_CONTENT_ID, id, 1);
  free (id);

  mu_header_get_stream (hdr2, &in);
  mu_stream_seek (in, 0, SEEK_SET);
  cat_message (out, in);
  mu_stream_close (out);
  mu_header_destroy (&hdr2, mu_header_get_owner (hdr2));

  finish_msg (env, msg);
  return 0;
}

int
edit_forw (char *cmd, struct compose_env *env, mu_message_t *pmsg, int level)
{
  char *sp, *id = NULL, *descr = NULL;
  int stop = 0, status = 0;
  int i, argc;
  char **argv;
  mu_header_t hdr;
  mu_mime_t mime;
  mu_message_t msg;
  char *val, *newval;
  
  skipws (cmd);
  while (stop == 0 && status == 0 && *cmd)
    {
      switch (*cmd++)
	{
	case '[':
	  if (descr)
	    {
	      mu_error (_("%s:%lu: description redefined"),
			input_file,
			(unsigned long) mhn_error_loc (env));
	      status = 1;
	      break;
	    }
	  status = parse_brace (&descr, &cmd, ']', env);
	  break;
	  
	case '<':
	  if (id)
	    {
	      mu_error (_("%s:%lu: content id redefined"),
			input_file,
			(unsigned long) mhn_error_loc (env));
	      status = 1;
	      break;
	    }
	  status = parse_brace (&id, &cmd, '>', env);
	  break;

	default:
	  cmd--;
	  stop = 1;
	  break;
	}
      skipws (cmd);
    }
  if (status)
    return status;

  if (mu_argcv_get (cmd, "\n", NULL, &argc, &argv))
    {
      mu_error (_("%s:%lu: syntax error"),
		input_file,
		(unsigned long) mhn_error_loc (env));
      return 1;
    }

  mu_mime_create (&mime, NULL, 0);
  
  mbox = mh_open_folder (argv[0], 0);
  for (i = 1; i < argc; i++)
    {
      mu_message_t input_msg;
      if (mh_get_message (mbox, i, &input_msg) == 0)
	{
	  mu_error (_("%s:%lu: no such message: %lu"),
		    input_file,
		    (unsigned long) mhn_error_loc (env),
		    (unsigned long)i);
	  return 1;
	}

      if ((status = mu_message_create_copy (&msg, input_msg)))
	break;
      mu_mime_add_part (mime, msg);
    }
  mu_argcv_free (argc, argv);

  if (*pmsg)
    {
      mu_message_unref (*pmsg);
      *pmsg = NULL;
    }
  
  mu_mime_get_message (mime, &msg);
  mu_message_get_header (msg, &hdr);
  mu_header_aget_value (hdr, MU_HEADER_CONTENT_TYPE, &val);
  sp = strchr (val, ';');
  if (!sp)
    abort ();
  asprintf (&newval, "multipart/digest%s", sp);
  mu_header_set_value (hdr, MU_HEADER_CONTENT_TYPE, newval, 1);
  free (val);
  free (newval);

  if (!id)
    id = mh_create_message_id (env->subpart);

  mu_header_set_value (hdr, MU_HEADER_CONTENT_ID, id, 1);
  free (id);

  if (descr)
    {
      mu_header_set_value (hdr, MU_HEADER_CONTENT_DESCRIPTION, descr, 1);
      free (descr);
    }
  
  finish_msg (env, &msg);
  return status;
}

int
edit_modify (char *cmd, struct compose_env *env, mu_message_t *msg)
{
  mu_header_t hdr;
  if (!*msg)
    mu_message_create (msg, NULL);
  mu_message_get_header (*msg, &hdr);
  return parse_type_command (&cmd, env, hdr);
}

int
edit_mime (char *cmd, struct compose_env *env, mu_message_t *msg, int level)
{
  int rc;
  mu_header_t hdr;
  mu_body_t body;
  mu_stream_t in, out = NULL, fstr;
  char *encoding;
  char *p, *typestr;
  char *shell_cmd;
  int flags;
  
  if (!*msg)
    mu_message_create (msg, NULL);
  mu_message_get_header (*msg, &hdr);
  rc = parse_type_command (&cmd, env, hdr);
  if (rc)
    return 1;

  mu_rtrim_class (cmd, MU_CTYPE_SPACE);

  _get_content_type (hdr, &typestr, NULL);
  shell_cmd = mhn_compose_command (typestr, &flags, cmd);
  free (typestr);

  /* Open the input stream, whatever it is */
  if (shell_cmd)
    {
      if (mhn_exec (&in, cmd, flags))
	return 1;
    }
  else if (p == cmd)
    {
      mu_error (_("%s:%lu: missing filename"),
		input_file,
		(unsigned long) mhn_error_loc (env));
      finish_msg (env, msg);
      return 1;
    }
  else
    {
      rc = mu_file_stream_create (&in, cmd, MU_STREAM_READ);
      if (rc)
	{
	  mu_error (_("cannot create input stream (file %s): %s"),
		    cmd, mu_strerror (rc));
	  return rc;
	}
      rc = mu_stream_open (in);
      if (rc)
	{
	  mu_error (_("cannot open input stream (file %s): %s"),
		    cmd, mu_strerror (rc));
	  mu_stream_destroy (&in, mu_stream_get_owner (in));
	  return rc;
	}
    }
  
  /* Create filter */

  if (_get_hdr_value (hdr, MU_HEADER_CONTENT_TRANSFER_ENCODING, &encoding))
    {
      char *typestr, *type, *subtype;
      
      _get_content_type (hdr, &typestr, NULL);
      split_content (typestr, &type, &subtype);
      if (mu_c_strcasecmp (type, "message") == 0)
	encoding = strdup ("7bit");
      else if (mu_c_strcasecmp (type, "text") == 0)
	encoding = strdup ("quoted-printable");
      else
	encoding = strdup ("base64");
      mu_header_set_value (hdr, MU_HEADER_CONTENT_TRANSFER_ENCODING, encoding, 1);
      free (typestr);
      free (type);
      free (subtype);
    }

  rc = mu_filter_create (&fstr, in, encoding, MU_FILTER_ENCODE, MU_STREAM_READ);
  if (rc)
    {
      fstr = in;
      in = NULL;
    }
  free (encoding);

  rc = mu_stream_open (fstr);
  if (rc)
    {
      mu_error (_("cannot open filter stream: %s"),
		mu_strerror (rc));
      return rc;
    }
  
  mu_message_get_body (*msg, &body);
  mu_body_get_stream (body, &out);
  cat_message (out, fstr);

  mu_stream_close (out);
  
  mu_stream_destroy (&fstr, mu_stream_get_owner (fstr));
  finish_msg (env, msg);
  return rc;
}

static int
has_nonascii (char *buf, size_t n)
{
  size_t i;
  for (i = 0; i < n; i++)
    if (!mu_isascii (buf[i]))
      return 1;
  return 0;
}

int
mhn_edit (struct compose_env *env, int level)
{
  int status = 0;
  char *buf = NULL;
  size_t bufsize = 0, n;
  mu_body_t body;
  mu_stream_t output;
  mu_message_t msg = NULL;
  size_t line_count = 0;
  int ascii_buf;
  
  while (status == 0
	 && stream_getline (env->input, &buf, &bufsize, &n) == 0 && n > 0)
    {
      env->line++;
      if (!msg)
	{
	  mu_header_t hdr;
	  /* Create new message */
	  mu_message_create (&msg, NULL);
	  mu_message_get_header (msg, &hdr);
	  mu_message_get_body (msg, &body);
	  mu_body_get_stream (body, &output);
	  mu_stream_seek (output, 0, SEEK_SET);
	  line_count = 0;
	  ascii_buf = 1; /* Suppose it is ascii */
	  env->subpart++;
	}
      
      if (buf[0] == '#')
	{
	  if (buf[1] == '#')
	    mu_stream_sequential_write (output, buf+1, n-1);
	  else
	    {
	      char *b2 = NULL;
	      size_t bs = 0, n2;
	      char *tok, *sp, c;

	      /* Collect the whole line */
	      while (n > 2 && buf[n-2] == '\\')
		{
		  int rc = stream_getline (env->input, &b2, &bs, &n2);
		  env->line++;
		  if (rc == 0 && n2 > 0)
		    {
		      if (n + n2 > bufsize)
			{
			  bufsize += 128;
			  buf = xrealloc (buf, bufsize);
			}
		      memcpy (buf + n - 2, b2, n2);
		      n += n2 - 2;
		    }
		}
	      free (b2);

	      mu_stream_close (output);
	      if (line_count)
		/* Close and append the previous part */
		finish_text_msg (env, &msg, ascii_buf);
	      
	      /* Execute the directive */
	      tok = buf;
	      sp = mu_str_skip_class_comp (buf, MU_CTYPE_SPACE);
	      c = *sp;
	      *sp = 0;

	      if (tok[1] == 0)
		/* starts a new message */;
	      else if (tok[1] == '<')
		{
		  *sp = c;
		  status = edit_modify (tok+2, env, &msg);
		}
	      else if (tok[1] == '@')
		{
		  *sp = c;
		  status = edit_extern (tok + 2, env, &msg, level);
		}
	      else if (strcmp (tok, "#forw") == 0)
		{
		  *sp = c;
		  status = edit_forw (sp, env, &msg, level);
		}
	      else if (strcmp (tok, "#begin") == 0)
		{
		  struct compose_env new_env;
		  mu_message_t new_msg;
		  
		  new_env.input = env->input;
		  new_env.line = env->line;
		  new_env.subpart = env->subpart;
		  mu_mime_create (&new_env.mime, NULL, 0);
		  status = mhn_edit (&new_env, level + 1);
		  env->line = new_env.line;
		  env->subpart = new_env.subpart;
		  if (status == 0)
		    {
		      mu_mime_get_message (new_env.mime, &new_msg);
		      mu_mime_add_part (env->mime, new_msg);
		      mu_message_unref (new_msg);
		    }
		}
	      else if (strcmp (tok, "#end") == 0)
		{
		  if (level == 0)
		    {
		      mu_error (_("%s:%lu: unmatched #end"),
				input_file,
				(unsigned long) mhn_error_loc (env));
		      status = 1;
		    }
		  break;
		}
	      else
		{
		  *sp = c;
		  status = edit_mime (tok + 1, env, &msg, level);
		}
	    }
	}
      else if (line_count > 0 || buf[0] != '\n')
	{
	  if (ascii_buf && has_nonascii (buf, n))
	    ascii_buf = 0;
	  mu_stream_sequential_write (output, buf, n);
	  line_count++;
	}
    }
  free (buf);

  if (msg)
    {
      mu_stream_close (output);
      if (line_count)
	finish_text_msg (env, &msg, ascii_buf);
      else
	mu_message_unref (msg);
    }
  
  return status;
}

int
parse_header_directive (char *val, char **encoding, char **charset, char **subject)
{
  char *p;

  /* Provide default values */
  *encoding = NULL;
  *charset = NULL;
  *subject = val;

  if (*val != '#')
    return 1;
  val++;
  
  switch (*val)
    {
    case '#':
      *subject = val;
      return 1;

    case '<':
      for (p = val; *p; p++)
	if (*p == '>')
	  {
	    int i, argc;
	    char **argv;
	    
	    *subject = p + 1;
	    *p = 0;
	    split_args (val + 1, &argc, &argv);
	    for (i = 0; i < argc; i++)
	      {
		if (strlen (argv[i]) > 8
		    && memcmp (argv[i], "charset=", 8) == 0)
		  {
		    free (*charset);
		    *charset = strdup (argv[i] + 8);
		  }
		else if (strlen (argv[i]) > 9
			 && memcmp (argv[i], "encoding=", 9) == 0)
		  {
		    free (*encoding);
		    *encoding = strdup (argv[i] + 9);
		  }
	      }
	    mu_argcv_free (argc, argv);
	    return 0;
	  }
      break;
    }
  return 1;
}


void
mhn_header (mu_message_t msg, mu_message_t omsg)
{
  mu_header_t hdr = NULL;
  char *val;
  
  mu_message_get_header (msg, &hdr);
  if (mu_header_aget_value (hdr, MU_HEADER_SUBJECT, &val) == 0)
    {
      char *subject, *encoding, *charset;
      
      if (parse_header_directive (val, &encoding, &charset, &subject))
	{
	  if (has_nonascii (val, strlen (val)))
	    {
	      int ismime = 0;
	      mu_message_t part = NULL;
	      
	      mu_message_is_multipart (omsg, &ismime);
	      if (ismime)
		mu_message_get_part (omsg, 1, &part);
	      else
		part = omsg;
	      if (part);
		{
		  mu_header_t parthdr = NULL;
		  char *typestr, *typeargs;
		  
		  mu_message_get_header (part, &parthdr);
		  _get_content_type (parthdr, &typestr, &typeargs);
		  _get_content_encoding (parthdr, &encoding);
		  if (typeargs)
		    {
		      int i, argc;
		      char **argv;
		      
		      split_args (typeargs, &argc, &argv);
		      for (i = 0; i < argc; i++)
			if (strlen (argv[i]) > 8
			    && memcmp (argv[i], "charset=", 8) == 0)
			  {
			    charset = strdup (argv[i]+8);
			    break;
			  }
		      mu_argcv_free (argc, argv);
		    }
		  free (typestr);
		}
	    }
	}

      if (charset)
	{
	  char *p;
	  int rc;
	  
	  if (!encoding || strcmp (encoding, "7bit") == 0)
	    {
	      free (encoding);
	      encoding = strdup ("base64");
	    }
 	  rc = mu_rfc2047_encode (charset, encoding, subject, &p);
	  if (rc)
	    mu_error (_("cannot encode subject using %s, %s: %s"),
		      charset, encoding, mu_strerror (rc));
	  else
	    {
	      mu_header_set_value (hdr, MU_HEADER_SUBJECT, p, 1);
	      free (p);
	    }
	}
      free (charset);
      free (encoding);
    }
}

int
mhn_compose ()
{
  int rc;
  mu_mime_t mime = NULL;
  mu_body_t body;
  mu_stream_t stream, in;
  struct compose_env env;
  mu_message_t msg;
  char *name, *backup, *p;
  
  mu_mime_create (&mime, NULL, 0);

  mu_message_get_body (message, &body);
  mu_body_get_stream (body, &stream);
  mu_stream_seek (stream, 0, SEEK_SET);

  env.mime = mime;
  env.input = stream;
  env.subpart = 0;
  env.line = 0;
  rc = mhn_edit (&env, 0);
  if (rc)
    return rc;

  mu_mime_get_message (mime, &msg);
  
  p = strrchr (input_file, '/');
  /* Prepare file names */
  if (p)
    {
      *p = 0;
      name = mu_tempname (input_file);
      asprintf (&backup, "%s/,%s.orig", input_file, p + 1);
      *p = '/';
    }
  else
    {
      name = mu_tempname (NULL);
      asprintf (&backup, ",%s.orig", input_file);
    }
  
  /* Create working draft file */
  unlink (name);
  rc = mu_file_stream_create (&stream, name, MU_STREAM_RDWR|MU_STREAM_CREAT);
  if (rc)
    {
      mu_error (_("cannot create output stream (file %s): %s"),
		name, mu_strerror (rc));
      free (name);
      return rc;
    }
  rc = mu_stream_open (stream);
  if (rc)
    {
      mu_error (_("cannot open output stream (file %s): %s"),
		name, mu_strerror (rc));
      free (name);
      mu_stream_destroy (&stream, mu_stream_get_owner (stream));
      return rc;
    }

  mhn_header (message, msg);
  copy_header_to_stream (message, stream);
  mu_message_get_stream (msg, &in);
  cat_message (stream, in);
  mu_stream_destroy (&stream, mu_stream_get_owner (stream));

  /* Preserve the backup copy and replace the draft */
  unlink (backup);
  rename (input_file, backup);
  rename (name, input_file);

  free (name);
  
  return 0;
}


/* *************************** Main Entry Point ************************** */
     
int
main (int argc, char **argv)
{
  int rc;
  int index;
  
  MU_APP_INIT_NLS ();
  
  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  argc -= index;
  argv += index;

  signal (SIGPIPE, SIG_IGN);
  
  if (input_file)
    {
      if (argc)
	{
	  mu_error (_("extra arguments"));
	  return 1;
	}
      message = mh_file_to_message (mu_folder_directory (), input_file);
      if (!message)
	return 1;
    }
  else if (mode == mode_compose)
    {
      if (argc > 1)
	{
	  mu_error (_("extra arguments"));
	  return 1;
	}
      input_file = argc == 1 ? argv[0] : "draft";
      message = mh_file_to_message (mu_folder_directory (), input_file);
      if (!message)
	return 1;
    }
  else
    {
      mbox = mh_open_folder (mh_current_folder (), 0);
      mh_msgset_parse (mbox, &msgset, argc, argv, "cur");
    }
  
  switch (mode)
    {
    case mode_compose:
      /* Prepare filename for diagnostic purposes */
      if (input_file[0] != '/')
	asprintf (&input_file, "%s/%s", mu_folder_directory (), input_file);
      rc = mhn_compose ();
      break;
      
    case mode_list:
      rc = mhn_list ();
      break;
      
    case mode_show:
      rc = mhn_show ();
      break;
      
    case mode_store:
      rc = mhn_store ();
      break;
      
    default:
      abort ();
    }
  return rc ? 1 : 0;
}
  
