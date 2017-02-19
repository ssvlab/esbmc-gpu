%{
/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2004, 2005, 2006, 2007, 2009, 2010 Free Software
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

#include <mh.h>
#include <regex.h>  
#include <pick.h>
  
static node_t *pick_node_create (node_type type, void *a, void *b);
static void set_cflags (char *str);
 
static regex_t *
regex_dup (regex_t *re)
{
  regex_t *rp = xmalloc (sizeof (*rp));
  *rp = *re;
  return rp;
}

int yyerror (const char *s);
int yylex (void); 
 
static node_t *parse_tree;
static int nesting_level;
static int reg_flags = REG_EXTENDED|REG_ICASE;
%}

%token <string> T_COMP T_DATEFIELD  T_STRING T_CFLAGS
%token T_LBRACE T_RBRACE T_BEFORE T_AFTER 
%left T_OR
%left T_AND
%left T_NOT

%union {
  char *string;
  node_t *node;
  regex_t regex;
};

%type <node> expr exprlist
%type <regex> regex

%%

input    : /* empty */
           {
	     parse_tree = NULL;
	   }
         | exprlist
           {
	     parse_tree = $1;
	   }
         ;

exprlist : expr
         | exprlist expr
           {
	     $$ = pick_node_create (node_and, $1, $2);
	   }
         ;

cflags   : /* empty */
         | T_CFLAGS
           {
	     set_cflags ($1);
	   }
         ;

regex    : cflags T_STRING
           {
	     int rc = regcomp (&$$, $2, reg_flags|REG_NOSUB);
	     if (rc)
	       {
		 char errbuf[512];
		 regerror (rc, &$$, errbuf, sizeof (errbuf));
		 mu_error ("error compiling regex \"%s\": %s",
			   $2, errbuf);
		 YYERROR;
	       }
	   }
         ;

expr     : lbrace exprlist rbrace
           {
	     $$ = $2;
	   }
         | cflags T_COMP regex
           {
	     $$ = pick_node_create (node_regex, $2, regex_dup (&$3));
	   }		      
         | regex
           {
	     $$ = pick_node_create (node_regex, NULL, regex_dup (&$1));
	   }		      
         | T_DATEFIELD
           {
	     $$ = pick_node_create (node_datefield, $1, NULL);
	   }
         | T_BEFORE T_STRING
           {
	     time_t t;
	     if (mu_parse_date ($2, &t, NULL))
	       {
		 mu_error (_("bad date format: %s"), $2);
		 exit (1);
	       }
	     $$ = pick_node_create (node_before, NULL, NULL);
	     $$->v.time = t;
	   }
         | T_AFTER T_STRING
           {
	     time_t t;
	     if (mu_parse_date ($2, &t, NULL))
	       {
		 mu_error (_("bad date format: %s"), $2);
		 exit (1);
	       }
	     $$ = pick_node_create (node_after, NULL, NULL);
	     $$->v.time = t;
	   }
         | expr T_AND expr
           {
	     $$ = pick_node_create (node_and, $1, $3);
	   }
         | expr T_OR expr
           {
	     $$ = pick_node_create (node_or, $1, $3);
	   }
	 | T_NOT expr
           {
	     $$ = pick_node_create (node_not, $2, NULL);
	   }
         ;

lbrace   : T_LBRACE
           {
	     nesting_level++;
	   }
         ;

rbrace   : T_RBRACE
           {
	     nesting_level--;
	   }
         ;

%%

/* Lexical analizer */

struct token
{
  int tok;
  char *val;
};

static mu_iterator_t iterator;

int
yylex ()
{
  struct token *tok;
  
  if (mu_iterator_is_done (iterator))
    return 0;
  mu_iterator_current (iterator, (void **)&tok);
  mu_iterator_next (iterator);
  yylval.string = tok->val;
  return tok->tok;
}

static char *
tokname (int tok)
{
  switch (tok)
    {
    case T_DATEFIELD:
      return "--datefield";
      
    case T_BEFORE:
      return "--before";
      
    case T_AFTER:
      return "--after";
      
    case T_LBRACE:
      return "--lbrace";
      
    case T_RBRACE:
      return "--rbrace";
      
    case T_OR:
      return "--or";
      
    case T_AND:
      return "--and";
      
    case T_NOT:
      return "--not";
    }
  return NULL;
}

int
yyerror (const char *s)
{
  int tok = yylex ();
  const char *str;
  
  if (!tok)
    str = _("end of input");
  else if (yylval.string)
    str = yylval.string;
  else
    str = tokname (tok);

  if (nesting_level)
    mu_error (_("%s near %s (missing closing brace?)"), s, str);
  else
    mu_error (_("%s near %s"), s, str);
  return 0;
}
  
void
pick_add_token (mu_list_t *list, int tok, char *val)
{
  struct token *tp;
  int rc;
  
  if (!*list && (rc = mu_list_create (list)))
    {
      mu_error(_("cannot create list: %s"), mu_strerror (rc));
      exit (1);
    }
  tp = xmalloc (sizeof (*tp));
  tp->tok = tok;
  tp->val = val;
  mu_list_append (*list, tp);
}

/* Main entry point */
int
pick_parse (mu_list_t toklist)
{
  int rc;
  
  if (!toklist)
    {
      parse_tree = NULL;
      return 0;
    }

  if (mu_list_get_iterator (toklist, &iterator))
    return -1;
  mu_iterator_first (iterator);
  rc = yyparse ();
  mu_iterator_destroy (&iterator);
  return rc;
}


/* Parse tree functions */

node_t *
pick_node_create (node_type type, void *a, void *b)
{
  node_t *node;

  node = xmalloc (sizeof (*node));
  node->type = type;
  node->v.gen.a = a;
  node->v.gen.b = b;
  return node;
}

struct eval_env
{
  mu_message_t msg;
  char *datefield;
};

static int
match_header (mu_message_t msg, char *comp, regex_t *regex)
{
  size_t i, count;
  mu_header_t hdr = NULL;
  char buf[128];
  
  mu_message_get_header (msg, &hdr);
  mu_header_get_field_count (hdr, &count);
  for (i = 1; i <= count; i++)
    {
      mu_header_get_field_name (hdr, i, buf, sizeof buf, NULL);
      if (mu_c_strcasecmp (buf, comp) == 0)
	{
	  mu_header_get_field_value (hdr, i, buf, sizeof buf, NULL);
	  if (regexec (regex, buf, 0, NULL, 0) == 0)
	    return 1;
	}
    }
  return 0;
}

static int
match_message (mu_message_t msg, regex_t *regex)
{
  mu_stream_t str = NULL;
  char buf[128];
  size_t n;
  
  mu_message_get_stream (msg, &str);
  mu_stream_seek (str, 0, SEEK_SET);
  while (mu_stream_sequential_readline (str, buf, sizeof buf, &n) == 0
	 && n > 0)
    {
      buf[n] = 0;
      if (regexec (regex, buf, 0, NULL, 0) == 0)
	return 1;
    }
  return 0;
}

static int
get_date_field (struct eval_env *env, time_t *t)
{
  mu_header_t hdr;
  char buf[128];
  
  if (mu_message_get_header (env->msg, &hdr))
    return 1;
  if (mu_header_get_value (hdr, env->datefield, buf, sizeof buf, NULL))
    return 1;
  return mu_parse_date (buf, t, NULL);
}

static int
pick_eval_node (node_t *node, struct eval_env *env)
{
  time_t t;
  
  switch (node->type)
    {
    case node_and:
      if (!pick_eval_node (node->v.op.larg, env))
	return 0;
      return pick_eval_node (node->v.op.rarg, env);
	
    case node_or:
      if (pick_eval_node (node->v.op.larg, env))
	return 1;
      return pick_eval_node (node->v.op.rarg, env);

    case node_not:
      return !pick_eval_node (node->v.op.larg, env);
      
    case node_regex:
      if (node->v.re.comp)
	return match_header (env->msg, node->v.re.comp, node->v.re.regex);
      else
	return match_message (env->msg, node->v.re.regex);
      
    case node_datefield:
      env->datefield = node->v.df.datefield;
      return 1;

    case node_before:
      if (get_date_field (env, &t))
	break;
      return t < node->v.time;
      
    case node_after:
      if (get_date_field (env, &t))
	break;
      return t > node->v.time;
    }

  return 0;
}

int
pick_eval (mu_message_t msg)
{
  struct eval_env env;
  
  if (!parse_tree)
    return 1;
  env.msg = msg;
  env.datefield = "date";
  return pick_eval_node (parse_tree, &env);
}

void
set_cflags (char *str)
{
  reg_flags = 0;
  for (; *str; str++)
    {
      switch (*str)
	{
	case 'b':
	case 'B':
	  reg_flags &= ~REG_EXTENDED;
	  break;

	case 'e':
	case 'E':
	  reg_flags |= REG_EXTENDED;
	  break;

	case 'c':
	case 'C':
	  reg_flags &= ~REG_ICASE;
	  break;
	  
	case 'i':
	case 'I':
	  reg_flags |= REG_ICASE;
	  break;

	default:
	  mu_error (_("Invalid regular expression flag: %c"), *str);
	  exit (1);
	}
    }
}
