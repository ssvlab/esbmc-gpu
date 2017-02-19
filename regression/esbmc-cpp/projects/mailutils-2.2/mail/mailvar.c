/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils.  If not, see <http://www.gnu.org/licenses/>. */

#include "mail.h"

#define MAILVAR_ALIAS  0x0001
#define MAILVAR_RDONLY 0x0002
#define MAILVAR_HIDDEN 0x0004

#define MAILVAR_TYPEMASK(type) (1<<(8+(type)))

struct mailvar_symbol
{
  struct mailvar_variable var;
  int flags;
  char *descr;
  void (*handler) (struct mailvar_variable *);
};

mu_list_t mailvar_list = NULL;

static void set_decode_fallback (struct mailvar_variable *);
static void set_replyregex (struct mailvar_variable *);
static void set_screen (struct mailvar_variable *);
static void set_debug (struct mailvar_variable *);
  
struct mailvar_symbol mailvar_tab[] =
  {
    /* FIXME: */
    { { "allnet", }, MAILVAR_HIDDEN },
    
    /* For compatibility with other mailx implementations.
       Never used, always true. */
    { { "append", },
      MAILVAR_TYPEMASK (mailvar_type_boolean) | MAILVAR_RDONLY,
      N_("messages saved in mbox are appended to the end rather than prepended") },
    { { "appenddeadletter", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("append the contents of canceled letter to dead.letter file") },
    { { "askbcc", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("prompt user for bcc before composing the message") },
    { { "askcc", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("prompt user for cc before composing the message") },
    { { "ask", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("prompt user for subject before composing the message") },
    { { "asksub", }, MAILVAR_ALIAS, NULL },
    { { "autoinc", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("automatically incorporate newly arrived messages")},
    { { "autoprint", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("delete command behaves like dp") },
    { { "byname" },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("record outgoing messages in a file named after the first recipient; "
	 "overrides the `record' variable") },
    { { "bang", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("replace every occurrence of ! in arguments to the shell command"
	 " with the last executed command") },
    { { "charset", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("output character set for decoded header fields") },
    { { "cmd", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("default shell command for pipe") },
    { { "columns", },
      MAILVAR_TYPEMASK (mailvar_type_number),
      N_("number of columns on terminal screen") },
    { { "crt", },
      MAILVAR_TYPEMASK (mailvar_type_number) |
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("if numeric, sets the minimum number of output lines needed "
	 "to engage paging; if boolean, use the height of the terminal "
	 "screen to compute the threshold") },
    { { "datefield", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("get date from the `Date:' header, instead of the envelope") },
    { { "debug", },
      MAILVAR_TYPEMASK (mailvar_type_string) |
        MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("set Mailutils debug level"),
      set_debug },
    { { "decode-fallback", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("how to represent characters that cannot be rendered using the "
	 "current character set"),
      set_decode_fallback },
    { { "dot", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("input message is terminated with a dot alone on a line") },
    { { "editheaders", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("allow to edit message headers while composing") },
    { { "emptystart", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("start interactive mode if the mailbox is empty") },
    { { "escape", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("character denoting escapes") },
    { { "flipr", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("swap the meaning of reply and Reply commands") },
    { { "folder", },
      MAILVAR_TYPEMASK (mailvar_type_string), 
      N_("folder directory name") },
    { { "fromfield", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("get sender address from the `From:' header, instead of "
	 "the envelope") },
    { { "gnu-last-command", },
      MAILVAR_TYPEMASK (mailvar_type_string) | MAILVAR_RDONLY,
      N_("last executed command line") },
    { { "header", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("run the `headers' command after entering interactive mode") },
    { { "headline", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("format string to use for the header summary"),
      mail_compile_headline },
    { { "hold", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("hold the read or saved messages in the system mailbox") },
    { { "ignore", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("ignore keyboard interrupts when composing messages") },
    { { "ignoreeof", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("typing the EOF character terminates the letter being composed") },
    { { "indentprefix", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("string used by the ~m escape for indenting quoted messages") },
    { { "inplacealiases", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("expand aliases in the address header field "
	 "before entering send mode") },
    /* For compatibility with other mailx implementations.
       Never used, always true. */
    { { "keep", },
      MAILVAR_TYPEMASK (mailvar_type_boolean) | MAILVAR_RDONLY,
      N_("truncate the user's system mailbox when it is empty") },
    { { "keepsave", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("keep saved messages in system mailbox too") },
    { { "mailx", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("enable mailx compatibility mode") },
    { { "metamail", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("interpret the content of message parts; if set to a string "
	 "specifies the name of the externam metamail command") },
    { { "metoo", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("do not remove sender addresses from the recipient list") },
    { { "mimenoask", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("a comma-separated list of MIME types for which "
	 "no confirmation is needed before running metamail interpreter") },
    { { "mode", },
      MAILVAR_TYPEMASK (mailvar_type_string) | MAILVAR_RDONLY,
      N_("the name of current operation mode") },
    { { "nullbody", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("accept messages with an empty body") },
    { { "nullbodymsg", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("display this text when sending a message with empty body") },
    { { "outfolder", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("keep created files in this folder") },
    { { "page", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("pipe command terminates each message with a linefeed") },
    { { "prompt", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("command prompt sequence") },
    { { "quit", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("keyboard interrupts terminate the program") },
    { { "rc", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("read the system-wide configuration file upon startup") },
    { { "readonly", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("mailboxes are opened in readonly mode") },
    { { "record", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("save outgoing messages in this file") },
    { { "recursivealiases", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("recursively expand aliases") },
    { { "regex", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("use regular expressions in message specifications") },
    { { "replyprefix", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("prefix for the subject line of a reply message") },
    { { "replyregex", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("regexp for recognizing subject lines of reply messages"),
      set_replyregex },
    { { "save", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("stored aborted messages in the user's dead.file") },
    { { "screen", },
      MAILVAR_TYPEMASK (mailvar_type_number),
      N_("number of lines on terminal screen"),
      set_screen },
    { { "sendmail", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("URL of the mail transport agent") },
    /* FIXME: Not yet used. */
    { { "sendwait", },
      MAILVAR_TYPEMASK (mailvar_type_boolean) | MAILVAR_HIDDEN, NULL },
    { { "sign", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("signature for use with the ~a command") },
    { { "Sign", },
      MAILVAR_TYPEMASK (mailvar_type_string),
      N_("signature for use with the ~A command") },
    { { "showenvelope", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("`print' command includes the SMTP envelope in its output") },
    { { "showto", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("if the message was sent by the user, print its recipient address "
	 "in the header summary") },
    { { "toplines", },
      MAILVAR_TYPEMASK (mailvar_type_number),
      N_("number of lines to be displayed by `top' or `Top'") },
    { { "variable-pretty-print", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("print variables with short descriptions") },
    { { "varpp", }, MAILVAR_ALIAS },
    { { "variable-strict", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("perform strict checking when setting options") },
    { { "varstrict", }, MAILVAR_ALIAS },
    { { "verbose", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("verbosely trace the process of message delivery") },
    { { "xmailer", },
      MAILVAR_TYPEMASK (mailvar_type_boolean),
      N_("add the `X-Mailer' header to the outgoing messages") },

    /* These will be implemented later */
    { { "onehop", }, MAILVAR_HIDDEN, NULL },

    { { "quiet", }, MAILVAR_TYPEMASK (mailvar_type_boolean) | MAILVAR_HIDDEN,
      "suppresses the printing of the version when first invoked" },
    
    { { NULL }, }
  };

static int mailvar_symbol_count = sizeof (mailvar_tab) / sizeof (mailvar_tab[0]) - 1;

struct mailvar_symbol *
find_mailvar_symbol (const char *var)
{
  struct mailvar_symbol *ep;
  for (ep = mailvar_tab; ep->var.name; ep++)
    if (strcmp (ep->var.name, var) == 0)
      {
	while ((ep->flags & MAILVAR_ALIAS) && ep > mailvar_tab)
	  ep--;
	return ep;
      }
  return NULL;
}

static void
print_descr (FILE *out, const char *s, int n,
	     int doc_col, int rmargin, char *pfx)
{
  if (!s)
    return;
  do
    {
      const char *p;
      const char *space = NULL;

      if (n == 1 && pfx)
	n += fprintf (out, "%s", pfx);
      
      for (; n < doc_col; n++)
	fputc (' ', out);

      for (p = s; *p && p < s + (rmargin - doc_col); p++)
	if (mu_isspace (*p))
	  space = p;
      
      if (!space || p < s + (rmargin - doc_col))
	{
	  fprintf (out, "%s", s);
	  s += strlen (s);
	}
      else
	{
	  for (; s < space; s++)
	    fputc (*s, out);
	  for (; *s && mu_isspace (*s); s++)
	    ;
	}
      fputc ('\n', out);
      n = 1;
    }
  while (*s);
}

/* Functions for dealing with internal mailvar_list variables */
static int
mailvar_variable_comp (const void *a, const void *b)
{
  const struct mailvar_variable *v1 = a;
  const struct mailvar_variable *v2 = b;

  return strcmp (v1->name, v2->name);
}

static int
mailvar_varptr_comp (const void *a, const void *b)
{
  const struct mailvar_variable * const * v1 = a;
  const struct mailvar_variable * const * v2 = b;

  return strcmp ((*v1)->name, (*v2)->name);
}

/* Find mailvar_list entry VAR. If not found and CREATE is not NULL, then
   create the (unset and untyped) variable */
struct mailvar_variable *
mailvar_find_variable (const char *name, int create)
{
  struct mailvar_symbol *sym;
  struct mailvar_variable *var;
  
  sym = find_mailvar_symbol (name);
  if (sym)
    var = &sym->var;
  else
    {
      struct mailvar_variable entry, *p;

      entry.name = (char*)name;
      /* Store it into the list */
      if (mailvar_list == NULL)
	{
	  mu_list_create (&mailvar_list);
	  mu_list_set_comparator (mailvar_list, mailvar_variable_comp);
	}
  
      if (mu_list_locate (mailvar_list, &entry, (void**)&p))
	{
	  if (!create)
	    return 0;
	  else
	    {
	      p = xmalloc (sizeof *p);
	      p->name = xstrdup (name);
	      mu_list_prepend (mailvar_list, p);
	    }
	}
      var = p;
      var->set = 0;
      var->type = mailvar_type_whatever;
      var->value.number = 0;
    }

  return var;
}


/* Retrieve the value of a specified variable of given type.
   The value is stored in the location pointed to by PTR variable.
   VARIABLE and TYPE specify the variable name and type. If the
   variable is not found and WARN is not null, the warning message
   is issued.

   Return value is 0 if the variable is found, 1 otherwise.
   If PTR is not NULL, it must point to

   int           if TYPE is mailvar_type_number or mailvar_type_boolean
   const char *  if TYPE is mailvar_type_string. 

   Passing PTR=NULL may be used to check whether the variable is set
   without retrieving its value. */
   
int
mailvar_get (void *ptr, const char *variable, enum mailvar_type type, int warn)
{
  struct mailvar_variable *var = mailvar_find_variable (variable, 0);

  if (!var->set || var->type != type)
    {
      if (warn)
	util_error (_("No value set for \"%s\""), variable);
      return 1;
    }
  if (ptr)
    switch (type)
      {
      case mailvar_type_string:
	*(char**)ptr = var->value.string;
	break;

      case mailvar_type_number:
	*(int*)ptr = var->value.number;
	break;

      case mailvar_type_boolean:
	*(int*)ptr = var->value.bool;
	break;

      default:
	break;
      }
  
  return 0;
}

/* Initialize mailvar_list entry: clear set indicator and free any memory
   associated with the data */
void
mailvar_variable_reset (struct mailvar_variable *var)
{
  if (!var->set)
    return;
  
  switch (var->type)
    {
    case mailvar_type_string:
      free (var->value.string);
      var->value.string = NULL;
      break;
	      
    default:
      break;
    }
  var->set = 0;
}

/* Set environement
   The  mailvar_set() function adds to the mailvar_list the VARIABLE
   with the given VALUE, if VARIABLE does not already exist.
   If it does exist in the mailvar_list, then its value is changed
   to VALUE if MOPTF_OVERWRITE bit is set in FLAGS, otherwise the
   value is not changed.

   Unless MOPTF_QUIET bit is set in FLAGS, the function performs
   semantic check, using the builtin options table.
 
   If VALUE is null the VARIABLE is unset. */
int
mailvar_set (const char *variable, void *value, enum mailvar_type type,
	     int flags)
{
  struct mailvar_variable *var;
  const struct mailvar_symbol *sym = find_mailvar_symbol (variable);
  int unset = flags & MOPTF_UNSET;
  
  if (!(flags & MOPTF_QUIET)
      && mailvar_get (NULL, "variable-strict", mailvar_type_boolean, 0) == 0)
    {
      if (!sym)
	mu_diag_output (MU_DIAG_WARNING, _("setting unknown variable %s"),
			variable);
      else if (sym->flags & MAILVAR_RDONLY)
	{
	  mu_error (_("Cannot set read-only variable %s"),
		    variable);
	  return 1;
	}
      else if (!(sym->flags & MAILVAR_TYPEMASK (type))
	       && !unset)
	{
	  mu_error (_("Wrong type for %s"), variable);
	  return 1;
	}
    }

  var = mailvar_find_variable (variable, !unset);

  if (!var || (var->set && !(flags & MOPTF_OVERWRITE)))
    return 0;

  mailvar_variable_reset (var);
  if (!unset)
    {
      var->type = type;
      if (value)
	{
	  var->set = 1;
	  switch (type)
	    {
	    case mailvar_type_number:
	      var->value.number = *(int*)value;
	      break;
	  
	    case mailvar_type_string:
	      var->value.string = strdup (value);
	      break;
	  
	    case mailvar_type_boolean:
	      var->value.bool = *(int*)value;
	      break;
		  
	    default:
	      abort();
	    }
	}
    }
    
  /* Special handling for some variables */
  if (sym && sym->flags & MAILVAR_TYPEMASK (type) && sym->handler)
    sym->handler (var);
  
  return 0;
}


static void
set_decode_fallback (struct mailvar_variable *var)
{
  if (mu_set_default_fallback (var->value.string))
    mu_error (_("Incorrect value for decode-fallback"));
}

static void
set_replyregex (struct mailvar_variable *var)
{
  int rc;
  char *err;
  
  if ((rc = mu_unre_set_regex (var->value.string, 0, &err)))
    {
      if (err)
	mu_error ("%s: %s", mu_strerror (rc), err);
      else
	mu_error ("%s", mu_strerror (rc));
    }
}

static void
set_screen (struct mailvar_variable *var)
{
  page_invalidate (1);
}

static void
set_mailbox_debug_level (mu_log_level_t level)
{
  if (mbox)
    {
      mu_debug_t mdbg;
      mu_mailbox_get_debug (mbox, &mdbg);
      mu_debug_set_level (mdbg, level);
    }
}  

#define DEFAULT_DEBUG_LEVEL  \
  (MU_DEBUG_INHERIT | MU_DEBUG_LEVEL_UPTO (MU_DEBUG_TRACE7))

static void
set_debug (struct mailvar_variable *var)
{
  int rc;
  int argc;
  char **argv;
  int i;
  mu_debug_t dbg;

  mu_global_debug_clear_level (NULL);
  set_mailbox_debug_level (0);

  if (var->type == mailvar_type_boolean)
    {
      if (var->set)
	{
	  /* FIXME: What to set here? 
	     mu_global_debug_set_level ("*", DEFAULT_DEBUG_LEVEL); */
	  set_mailbox_debug_level (DEFAULT_DEBUG_LEVEL);
	}
      return;
    }
  
  mu_diag_get_debug (&dbg);
  
  rc = mu_argcv_get (var->value.string, ";", NULL, &argc, &argv);
  if (rc)
    {
      mu_error (_("Cannot parse string: %s"), mu_strerror (rc));
      return;
    }

  for (i = 0; i < argc; i++)
    {
      char *p;
      mu_log_level_t level = MU_DEBUG_INHERIT;
      char *object_name = argv[i];
      
      for (p = object_name; *p && *p != '='; p++)
	;

      if (*p == '=')
	{
	  *p++ = 0;
	  mu_debug_level_from_string (p, &level, dbg);
	}
      else
	level |= MU_DEBUG_LEVEL_UPTO (MU_DEBUG_PROT);
      
      if (strcmp (object_name, "mailbox") == 0)
	set_mailbox_debug_level (level);
      mu_global_debug_set_level (object_name, level);
    }
}


size_t
_mailvar_symbol_count (int set)
{
  if (!set)
    return mailvar_symbol_count;
  else
    {
      struct mailvar_symbol *s;
      size_t count = 0;
      for (s = mailvar_tab; s->var.name; s++)
	if (s->var.set)
	  count++;
      return count;
    }
}

void
_mailvar_symbol_to_array (int set, struct mailvar_variable **vartab)
{
  struct mailvar_symbol *s;
  for (s = mailvar_tab; s->var.name; s++)
    if (!set || s->var.set)
      {
	*vartab = &s->var;
	vartab++;
      }
}

struct mailvar_variable **
mailvar_make_array (int set, size_t *pcount)
{
  struct mailvar_variable **vartab;
  size_t count = 0, symcount;

  symcount = _mailvar_symbol_count (set);
  mu_list_count (mailvar_list, &count);
  vartab = xcalloc (symcount + count + 1, sizeof *vartab);
  mu_list_to_array (mailvar_list, (void**) vartab, count, NULL);
  _mailvar_symbol_to_array (set, vartab + count);
  count += symcount;
  qsort (vartab, count, sizeof *vartab, mailvar_varptr_comp);
  vartab[count] = NULL;
  if (pcount)
    *pcount = count;
  return vartab;
}
  

struct mailvar_iterator
{
  const char *prefix;
  int prefixlen;
  struct mailvar_variable **varptr;
  size_t varcnt;
  size_t pos;
};
  
const char *
mailvar_iterate_next (struct mailvar_iterator *itr)
{
  struct mailvar_variable *vp;
  
  while (itr->pos < itr->varcnt)
    {
      vp = itr->varptr[itr->pos++];
  
      if (strlen (vp->name) >= itr->prefixlen
	  && strncmp (vp->name, itr->prefix, itr->prefixlen) == 0)
	return vp->name;
    }
  return NULL;
}

const char *
mailvar_iterate_first (int set, const char *prefix, struct mailvar_iterator **pitr)
{
  struct mailvar_iterator *itr = xmalloc (sizeof *itr);
  itr->prefix = prefix;
  itr->prefixlen = strlen (prefix);
  itr->varptr = mailvar_make_array (set, &itr->varcnt);
  itr->pos = 0;
  *pitr = itr;
  return mailvar_iterate_next (itr);
}

void
mailvar_iterate_end (struct mailvar_iterator **pitr)
{
  if (pitr && *pitr)
    {
      struct mailvar_iterator *itr = *pitr;
      free (itr->varptr);
      free (itr);
      *pitr = NULL;
    }
}

void
mailvar_print (int set)
{
  struct mailvar_variable **vartab;
  FILE *out = ofile;
  size_t i, count;
  int pagelines = util_get_crt ();
  int width = util_getcols ();
  int prettyprint = mailvar_get (NULL, "variable-pretty-print",
				 mailvar_type_boolean, 0) == 0;
  
  vartab = mailvar_make_array (set, &count);

  if (pagelines && count > pagelines)
    out = popen (getenv ("PAGER"), "w");

  for (i = 0; i < count; i++)
    {
      if (prettyprint)
	{
	  const struct mailvar_symbol *sym =
	    find_mailvar_symbol (vartab[i]->name);

	  if (sym)
	    {
	      if (sym->flags & MAILVAR_HIDDEN)
		continue;
	      if (sym->flags & MAILVAR_RDONLY)
		fprintf (out, "# %s:\n", _("Read-only variable"));
	      print_descr (out, gettext (sym->descr), 1, 3, width - 1, "# ");
	    }
	}
      switch (vartab[i]->type)
	{
	case mailvar_type_number:
	  fprintf (out, "%s=%d", vartab[i]->name, vartab[i]->value.number);
	  break;
	  
	case mailvar_type_string:
	  fprintf (out, "%s=\"%s\"", vartab[i]->name, vartab[i]->value.string);
	  break;
	  
	case mailvar_type_boolean:
	  if (!vartab[i]->value.bool)
	    fprintf (out, "no");
	  fprintf (out, "%s", vartab[i]->name);
	  break;
	  
	case mailvar_type_whatever:
	  fprintf (out, "%s %s", vartab[i]->name, _("oops?"));
	}
      fprintf (out, "\n");
    }
  free (vartab);

  if (out != ofile)
    pclose (out);
}


void
mailvar_variable_format (FILE *fp,
			 const struct mailvar_variable *var,
			 const char *defval)
{
  if (var)
    switch (var->type)
      {
      case mailvar_type_string:
	fprintf (fp, "%s", var->value.string);
	break;

      case mailvar_type_number:
	fprintf (fp, "%d", var->value.number);
	break;

      case mailvar_type_boolean:
	fprintf (fp, "%s", var->set ? "yes" : "no");
	break;

      default:
	if (defval)
	  fprintf (fp, "%s", defval);
	break;
      }
}


static char *typestr[] =
  {
    N_("untyped"),
    N_("numeric"),
    N_("string"),
    N_("boolean")
  };

static void
describe_symbol (FILE *out, int width, const struct mailvar_symbol *sym)
{
  int n;
  int i, t;
  const struct mailvar_symbol *ali;

  n = fprintf (out, "%s", sym->var.name);
  for (ali = sym + 1; ali->var.name && ali->flags & MAILVAR_ALIAS; ali++)
    {
      size_t len = strlen (ali->var.name) + 2;
      if (n + len > width)
	n = fprintf (out, "\n%s", ali->var.name);
      else
	n = fprintf (out, ", %s", ali->var.name);
    }
  fputc ('\n', out);
  
  fprintf (out, _("Type: "));
  for (i = 0, t = 0; i < sizeof (typestr) / sizeof (typestr[0]); i++)
    if (sym->flags & MAILVAR_TYPEMASK (i))
      {
	if (t++)
	  fprintf (out, " %s ", _("or"));
	fprintf (out, "%s", gettext (typestr[i]));
      }
  if (!t)
    fprintf (out, "%s", gettext (typestr[0]));
  fputc ('\n', out);
  
  fprintf (out, "%s", _("Current value: "));
  mailvar_variable_format (out, &sym->var, _("[not set]"));

  if (sym->flags & MAILVAR_RDONLY)
    fprintf (out, " [%s]", _("read-only"));
  fputc ('\n', out);
  
  print_descr (out, gettext (sym->descr ? sym->descr : N_("Not documented")),
	       1, 1, width - 1, NULL);
  fputc ('\n', out);
}

int
mail_variable (int argc, char **argv)
{
  int pagelines = util_get_crt ();
  int width = util_getcols ();
  FILE *out = ofile;

  if (pagelines)
    out = popen (getenv ("PAGER"), "w");
  
  if (argc == 1)
    {
      struct mailvar_symbol *sym;

      for (sym = mailvar_tab; sym->var.name; sym++)
	if (!(sym->flags & (MAILVAR_HIDDEN|MAILVAR_ALIAS)))
	  describe_symbol (out, width, sym);
    }
  else
    {
      int i;

      for (i = 1; i < argc; i++)
	{
	  struct mailvar_symbol *sym = find_mailvar_symbol (argv[i]);
	  if (!sym)
	    fprintf (out, "%s: unknown\n", argv[i]);
	  else
	    describe_symbol (out, width, sym);
	}
    }
  if (out != ofile)
    pclose (out);
  return 0;
}


#ifdef WITH_READLINE
static char *
mailvar_generator (int set, const char *text, int state)
{
  static struct mailvar_iterator *itr;
  const char *p;
  
  if (!state)
    p = mailvar_iterate_first (set, text, &itr);
  else
    p = mailvar_iterate_next (itr);

  if (!p)
    {
      mailvar_iterate_end (&itr);
      return NULL;
    }
  return strdup (p);
}

static char *
mailvar_set_generator (const char *text, int state)
{
  return mailvar_generator (0, text, state);
}

static char *
mailvar_unset_generator (const char *text, int state)
{
  return mailvar_generator (1, text, state);
}

char **
mailvar_set_compl (int argc, char **argv, int ws)
{
  ml_attempted_completion_over ();
  return rl_completion_matches (ws ? "" : argv[argc-1],
				/* Possible values for argv[0] are:
				   set, unset, variable */
				argv[0][0] == 'u' ? mailvar_unset_generator
				                  : mailvar_set_generator);
}

#endif
