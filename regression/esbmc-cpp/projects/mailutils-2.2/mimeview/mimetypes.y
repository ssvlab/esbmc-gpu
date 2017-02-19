%{
/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2007, 2009, 2010 Free Software Foundation, Inc.

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
  
#include <mailutils/cctype.h>
#include <mimeview.h>
#include <mimetypes-decl.h>
  
static void
yyprint (FILE *output, unsigned short toknum, YYSTYPE val)
{
  switch (toknum)
    {
    case IDENT:
    case IDENT_L:
    case STRING:
      fprintf (output, "[%lu] %s", (unsigned long) val.string.len,
	       val.string.ptr);
      break;

    case EOL:
    default:
      break;
    }
}

#define YYPRINT yyprint

static mu_list_t arg_list; /* For error recovery */

#define L_OR  0
#define L_AND 1

enum node_type
  {
    functional_node,
    binary_node,
    negation_node,
    suffix_node
  };

union argument
{
  struct mimetypes_string *string;
  unsigned number;
  int c;
};

typedef int (*builtin_t) (union argument *args);

struct node
{
  enum node_type type;
  union
  {
    struct
    {
      builtin_t fun;
      union argument *args;
    } function;
    struct node *arg;
    struct
    {
      int op;
      struct node *arg1;
      struct node *arg2;
    } bin; 
    struct mimetypes_string suffix;
  } v;
};

static struct node *make_binary_node (int op,
				      struct node *left, struct node *rigth);
static struct node *make_negation_node (struct node *p);

static struct node *make_suffix_node (struct mimetypes_string *suffix);
static struct node *make_functional_node (char *ident, mu_list_t list);

static int eval_rule (struct node *root);

struct rule_tab
{
  char *type;
  struct node *node;
};

static mu_list_t rule_list;

%}

%token <string> IDENT IDENT_L
%token <string> STRING
%token EOL BOGUS

%left ','
%left '+'

%type <string> string arg type
%type <list> arglist
%type <node> function stmt rule

%union {
  struct mimetypes_string string;
  mu_list_t list;
  int result;
  struct node *node;
}

%%

input    : list
         ;

list     : rule_line
         | list eol rule_line
         ; 

rule_line: /* empty */ 
         | type rule
           {
	     struct rule_tab *p = mimetypes_malloc (sizeof (*p));
	     if (!rule_list)
	       mu_list_create (&rule_list);
	     p->type = $1.ptr;
	     p->node = $2;
	     mu_list_append (rule_list, p);
	   }
	 | error eol
           {
	     if (arg_list)
	       mu_list_destroy (&arg_list);
	     arg_list = NULL;
	     reset_lex ();
	   }
         ; 

eol      : EOL
         | eol EOL
         ;

type     : IDENT '/' IDENT
           {
	     $$ = mimetypes_append_string2 (&$1, '/', &$3);
	   }
         ;

rule     : stmt
         | rule rule %prec ','
           {
	     $$ = make_binary_node (L_OR, $1, $2);
	   }
         | rule ',' rule
           {
	     $$ = make_binary_node (L_OR, $1, $3);
	   }
         | rule '+' rule
           {
	     $$ = make_binary_node (L_AND, $1, $3);
	   }
         ;

stmt     : '!' stmt
           {
	     $$ = make_negation_node ($2);
	   }
         | '(' rule ')'
           {
	     $$ = $2;
	   }
         | string
           {
	     $$ = make_suffix_node (&$1);
	   }
         | function
         ;

string   : STRING
         | IDENT
         ;

function : IDENT_L arglist ')'
           {
	     reset_lex ();
	     $$ = make_functional_node ($1.ptr, $2);
	     if (!$$)
	       YYERROR;
	   }
         ;

arglist  : arg
           {
	     mu_list_create (&arg_list);
	     $$ = arg_list;
	     mu_list_append ($$, mimetypes_string_dup (&$1));
	   }
         | arglist ',' arg
           {
	     mu_list_append ($1, mimetypes_string_dup (&$3));
	     $$ = $1;
	   }
         ;

arg      : string
         ;

%%

int
mimetypes_parse (const char *name)
{
  int rc;
  if (mimetypes_open (name))
    return 1;
  rc = yyparse ();
  mimetypes_close ();
  return rule_list == NULL;
}
  
void
mimetypes_gram_debug (int level)
{
  yydebug = level;
}


static struct node *
make_node (enum node_type type)
{
  struct node *p = mimetypes_malloc (sizeof *p);
  p->type = type;
  return p;
}

static struct node *
make_binary_node (int op, struct node *left, struct node *right)
{
  struct node *node = make_node (binary_node);

  node->v.bin.op = op;
  node->v.bin.arg1 = left;
  node->v.bin.arg2 = right;
  return node;
}

struct node *
make_negation_node (struct node *p)
{
  struct node *node = make_node (negation_node);
  node->v.arg = p;
  return node;
}

struct node *
make_suffix_node (struct mimetypes_string *suffix)
{
  struct node *node = make_node (suffix_node);
  node->v.suffix = *suffix;
  return node;
}

struct builtin_tab
{
  char *name;
  char *args;
  builtin_t handler;
};

/*        match("pattern")
            Pattern match on filename
*/
static int
b_match (union argument *args)
{
  return fnmatch (args[0].string->ptr, mimeview_file, 0) == 0;
}

/*       ascii(offset,length)
            True if bytes are valid printable ASCII (CR, NL, TAB,
            BS, 32-126)
*/
static int
b_ascii (union argument *args)
{
  int i;
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }

  for (i = 0; i < args[1].number; i++)
    {
      int c = getc (mimeview_fp);
      if (c == EOF)
	break;
      if (!mu_isascii (c))
	return 0;
    }
      
  return 1;
}

/*       printable(offset,length)
            True if bytes are printable 8-bit chars (CR, NL, TAB,
            BS, 32-126, 128-254)
*/
#define ISPRINT(c) ((c) &&\
                    (strchr ("\n\r\t\b",c) \
                     || (32<=(c) && (c)<=126) \
                     || (128<=(c) && (c)<=254)))
static int
b_printable (union argument *args)
{
  int i;

  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }

  for (i = 0; i < args[1].number; i++)
    {
      int c = getc (mimeview_fp);
      if (c == EOF)
	break;
      if (!ISPRINT ((unsigned)c))
	return 0;
    }
  return 1;
}

/*        string(offset,"string")
            True if bytes are identical to string
*/
static int
b_string (union argument *args)
{
  struct mimetypes_string *str = args[1].string;
  int i;
  
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }

  for (i = 0; i < str->len; i++)
    {
      int c = getc (mimeview_fp);
      if (c == EOF || (char)c != str->ptr[i])
	return 0;
    }
  return 1;
}

/*        istring(offset,"string")
            True if a case-insensitive comparison of the bytes is
            identical
*/
static int
b_istring (union argument *args)
{
  int i;
  struct mimetypes_string *str = args[1].string;
  
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }

  for (i = 0; i < str->len; i++)
    {
      int c = getc (mimeview_fp);
      if (c == EOF || mu_tolower (c) != mu_tolower (str->ptr[i]))
	return 0;
    }
  return 1;
}

/*       char(offset,value)
            True if byte is identical
*/
static int
b_char (union argument *args)
{
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }
  return getc (mimeview_fp) == args[1].number;
}

/*        short(offset,value)
            True if 16-bit integer is identical
	  FIXME: Byte order  
*/
static int
b_short (union argument *args)
{
  unsigned short val;
  int rc;
  
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }
  rc = fread (&val, sizeof val, 1, mimeview_fp);

  if (rc == -1)
    {
      mu_error ("fread: %s", mu_strerror (errno));
      return 0;
    }
  else if (rc == 0)
    return 0;
  return val == args[1].number;
}

/*        int(offset,value)
            True if 32-bit integer is identical
          FIXME: Byte order
*/
static int
b_int (union argument *args)
{
  unsigned int val;
  int rc;
  
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }
  rc = fread (&val, sizeof val, 1, mimeview_fp);
  if (rc == -1)
    {
      mu_error ("fread: %s", mu_strerror (errno));
      return 0;
    }
  else if (rc == 0)
    return 0;
  return val == args[1].number;
}

/*        locale("string")
            True if current locale matches string
*/
static int
b_locale (union argument *args)
{
  abort (); /* FIXME */
  return 0;
}

/*        contains(offset,range,"string")
            True if the range contains the string
*/
static int
b_contains (union argument *args)
{
  int i, count;
  char *buf;
  struct mimetypes_string *str = args[2].string;

  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }
  buf = xmalloc (args[1].number);
  count = fread (buf, 1, args[1].number, mimeview_fp);
  if (count == -1)
    {
      mu_error ("fread: %s", mu_strerror (errno));
    }
  else if (count > str->len)
    for (i = 0; i < count - str->len; i++)
      if (buf[i] == str->ptr[0] && memcmp (buf + i, str->ptr, str->len) == 0)
	{
	  free (buf);
	  return 1;
	}
  free (buf);
  return 0;
}

static struct builtin_tab builtin_tab[] = {
  { "match", "s", b_match },
  { "ascii", "dd", b_ascii },
  { "printable", "dd", b_printable },
  { "string", "ds", b_string },
  { "istring", "ds", b_istring },
  { "char", "dc", b_char },
  { "short", "dd", b_short },
  { "int", "dd", b_int },
  { "locale", "s", b_locale },
  { "contains", "dds", b_contains },
  { NULL }
};
  
struct node *
make_functional_node (char *ident, mu_list_t list)
{
  size_t count, i;
  struct builtin_tab *p;
  struct node *node;
  union argument *args;
  mu_iterator_t itr;
  
  for (p = builtin_tab; ; p++)
    {
      if (!p->name)
	{
	  char *s;
	  asprintf (&s, _("%s: unknown function"), ident);
	  yyerror (s);
	  free (s);
	  return NULL;
	}

      if (strcmp (ident, p->name) == 0)
	break;
    }

  mu_list_count (list, &count);
  i = strlen (p->args);

  if (count < i)
    {
      char *s;
      asprintf (&s, _("too few arguments in call to `%s'"), ident);
      yyerror (s);
      free (s);
      return NULL;
    }
  else if (count > i)
    {
      char *s;
      asprintf (&s, _("too many arguments in call to `%s'"), ident);
      yyerror (s);
      free (s);
      return NULL;
    }

  args = mimetypes_malloc (count * sizeof *args);
  
  mu_list_get_iterator (list, &itr);
  for (i = 0, mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr), i++)
    {
      struct mimetypes_string *data;
      char *tmp;
      
      mu_iterator_current (itr, (void **)&data);
      switch (p->args[i])
	{
	case 'd':
	  args[i].number = strtoul (data->ptr, &tmp, 0);
	  if (*tmp)
	    goto err;
	  break;
	  
	case 's':
	  args[i].string = data;
	  break;
	  
	case 'c':
	  args[i].c = strtoul (data->ptr, &tmp, 0);
	  if (*tmp)
	    goto err;
	  break;
	  
	default:
	  abort ();
	}
    }

  node = make_node (functional_node);
  node->v.function.fun = p->handler;
  node->v.function.args = args;
  return node;
  
 err:
  {
    char *s;
    asprintf (&s,
	      _("argument %lu has wrong type in call to `%s'"),
	      (unsigned long) i, ident);
    yyerror (s);
    free (s);
    return NULL;
  }
}

static int
check_suffix (char *suf)
{
  char *p = strrchr (mimeview_file, '.');
  if (!p)
    return 0;
  return strcmp (p+1, suf) == 0;
}

static int
eval_rule (struct node *root)
{
  int result;
  
  switch (root->type)
    {
    case functional_node:
      result = root->v.function.fun (root->v.function.args);
      break;
      
    case binary_node:
      result = eval_rule (root->v.bin.arg1);
      switch (root->v.bin.op)
	{
	case L_OR:
	  if (!result)
	    result |= eval_rule (root->v.bin.arg2);
	  break;
	  
	case L_AND:
	  if (result)
	    result &= eval_rule (root->v.bin.arg2);
	  break;
	  
	default:
	  abort ();
	}
      break;
      
    case negation_node:
      result = !eval_rule (root->v.arg);
      break;
      
    case suffix_node:
      result = check_suffix (root->v.suffix.ptr);
      break;

    default:
      abort ();
    }
  return result;
}

static int
evaluate (void *item, void *data)
{
  struct rule_tab *p = item;
  char **ptype = data;
    
  if (eval_rule (p->node))
    {
      *ptype = p->type;
      return 1;
    }
  return 0;
}

const char *
get_file_type ()
{
  const char *type = NULL;
  mu_list_do (rule_list, evaluate, &type);
  return type;
}
    
