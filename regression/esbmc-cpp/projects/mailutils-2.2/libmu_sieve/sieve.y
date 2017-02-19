%{
/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2006, 2007, 2008, 2009,
   2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif  
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sieve-priv.h>

mu_sieve_machine_t mu_sieve_machine;
int mu_sieve_error_count;

static void branch_fixup (size_t start, size_t end);
static int _sieve_default_error_printer (void *, const char *, va_list);
static int _sieve_default_parse_error (void *, const char *, int,
				       const char *, va_list);
%}

%union {
  char *string;
  size_t number;
  sieve_instr_t instr;
  mu_sieve_value_t *value;
  mu_list_t list;
  size_t pc;
  struct {
    size_t start;
    size_t end;
  } pclist;
  struct {
    char *ident;
    mu_list_t args;
  } command;
  struct {
    size_t begin;
    size_t cond;
    size_t branch;
  } branch;
}

%token <string> IDENT TAG
%token <number> NUMBER
%token <string> STRING MULTILINE
%token REQUIRE IF ELSIF ELSE ANYOF ALLOF NOT

%type <value> arg
%type <list> slist stringlist stringorlist arglist maybe_arglist
%type <command> command
%type <pclist> testlist
%type <pc> action test statement list elsif else cond begin if block
%type <branch> elsif_branch maybe_elsif else_part

%%

input        : /* empty */
             | list
               { /* to placate bison */ }
             ;

list         : statement
             | list statement
             ;

statement    : REQUIRE stringorlist ';'
               {
		 mu_sieve_require ($2);
		 /*  All the items in $2 are registered in memory_pool,
		     so we don't free them */
		 mu_list_destroy (&$2);
		 $$ = mu_sieve_machine->pc;
	       }
             | action ';'
	     /* 1  2     3       4    */ 
             | if cond block else_part
               {
		 mu_sieve_machine->prog[$2].pc = $4.begin - $2 - 1;
		 if ($4.branch)
		   branch_fixup ($4.branch, mu_sieve_machine->pc);
	       }		 
             ;

if           : IF
               {
		 $$ = mu_sieve_machine->pc;
	       }
             ;

else_part    : maybe_elsif
               {
		 if ($1.begin)
		   mu_sieve_machine->prog[$1.cond].pc =
		                  mu_sieve_machine->pc - $1.cond - 1;
		 else
		   {
		     $$.begin = mu_sieve_machine->pc;
		     $$.branch = 0;
		   }
	       }
             | maybe_elsif else block
               {
		 if ($1.begin)
		   {
		     mu_sieve_machine->prog[$1.cond].pc = $3 - $1.cond - 1;
		     mu_sieve_machine->prog[$2].pc = $1.branch;
		     $$.begin = $1.begin;
		     $$.branch = $2;
		   }
		 else
		   {
		     $$.begin = $3;
		     $$.branch = $2;
		   }
	       }
             ;

maybe_elsif  : /* empty */
               {
		 $$.begin = 0;
	       }
             | elsif_branch
             ;

elsif_branch : elsif begin cond block
               {
		 $$.begin = $2; 
		 $$.branch = $1;
		 $$.cond = $3;
	       }
             | elsif_branch elsif begin cond block
               {
		 mu_sieve_machine->prog[$1.cond].pc = $3 - $1.cond - 1;
		 mu_sieve_machine->prog[$2].pc = $1.branch;
		 $$.begin = $1.begin;
		 $$.branch = $2;
		 $$.cond = $4;
	       }
             ;

elsif        : ELSIF
               {
		 mu_sv_code_instr (_mu_sv_instr_branch);
		 $$ = mu_sieve_machine->pc;
		 mu_sv_code_number (0);
	       }
             ;

else         : ELSE
               {
		 mu_sv_code_instr (_mu_sv_instr_branch);
		 $$ = mu_sieve_machine->pc;
		 mu_sv_code_number (0);
	       }
             ;

block        : '{' list '}'
               {
		 $$ = $2;
	       }
             ;

testlist     : cond_expr
               {
		 $$.start = $$.end = mu_sieve_machine->pc;
		 if (mu_sv_code_instr (_mu_sv_instr_brz)
		     || mu_sv_code_number (0))
		   YYERROR;
	       }
             | testlist ',' cond_expr
               {
		 mu_sieve_machine->prog[$1.end+1].pc = mu_sieve_machine->pc;
		 $1.end = mu_sieve_machine->pc;
		 if (mu_sv_code_instr (_mu_sv_instr_brz)
		     || mu_sv_code_number (0))
		   YYERROR;
		 $$ = $1;
	       }
             ;

cond         : cond_expr
               {
		 mu_sv_code_instr (_mu_sv_instr_brz);
		 $$ = mu_sieve_machine->pc;
		 mu_sv_code_number (0);
	       }
             ;

cond_expr    : test
               { /* to placate bison */ }
             | ANYOF '(' testlist ')'
               {
		 mu_sv_code_anyof ($3.start);
	       }
             | ALLOF '(' testlist ')'
               {
		 mu_sv_code_allof ($3.start);
	       }
             | NOT cond_expr
               {
		 if (mu_sv_code_instr (_mu_sv_instr_not))
		   YYERROR;
	       }
             ;

begin        : /* empty */
               {
		 $$ = mu_sieve_machine->pc;
	       }
             ; 

test         : command
               {
		 mu_sieve_register_t *reg = 
		        mu_sieve_test_lookup (mu_sieve_machine, $1.ident);
		 $$ = mu_sieve_machine->pc;

		 if (!reg)
		   mu_sv_compile_error (&mu_sieve_locus,
					_("unknown test: %s"),
					$1.ident);
		 else if (!reg->required)
		   mu_sv_compile_error (&mu_sieve_locus,
					_("test `%s' has not been required"),
					$1.ident);
		 else if (mu_sv_code_test (reg, $1.args))
		   YYERROR;
	       }
             ;

command      : IDENT maybe_arglist
               {
		 $$.ident = $1;
		 $$.args = $2;
	       }
             ;

action       : command
               {
		 mu_sieve_register_t *reg = 
		        mu_sieve_action_lookup (mu_sieve_machine, $1.ident);
		 
		 $$ = mu_sieve_machine->pc;
		 if (!reg)
		   mu_sv_compile_error (&mu_sieve_locus,
					_("unknown action: %s"),
					$1.ident);
		 else if (!reg->required)
		   mu_sv_compile_error (&mu_sieve_locus,
					_("action `%s' has not been required"),
					$1.ident);
		 else if (mu_sv_code_action (reg, $1.args))
		   YYERROR;
	       }
             ;

maybe_arglist: /* empty */
               {
		 $$ = NULL;
	       }
             | arglist
	     ;

arglist      : arg
               {
		 mu_list_create (&$$);
		 mu_list_append ($$, $1);
	       }		 
             | arglist arg
               {
		 mu_list_append ($1, $2);
		 $$ = $1;
	       }
             ;

arg          : stringlist
               {
		 $$ = mu_sieve_value_create (SVT_STRING_LIST, $1);
	       }
             | STRING
               {
		 $$ = mu_sieve_value_create (SVT_STRING, $1);
               } 
             | MULTILINE
               {
		 $$ = mu_sieve_value_create (SVT_STRING, $1);
	       }
             | NUMBER
               {
		 $$ = mu_sieve_value_create (SVT_NUMBER, &$1);
	       }
             | TAG
               {
		 $$ = mu_sieve_value_create (SVT_TAG, $1);
	       }
             ;

stringorlist : STRING
               {
		 mu_list_create (&$$);
		 mu_list_append ($$, $1);
	       }
             | stringlist
             ;

stringlist   : '[' slist ']'
               {
		 $$ = $2;
	       }
             ;

slist        : STRING
               {
		 mu_list_create (&$$);
		 mu_list_append ($$, $1);
	       }
             | slist ',' STRING
               {
		 mu_list_append ($1, $3);
		 $$ = $1;
	       }
             ;

%%

int
yyerror (const char *s)
{
  mu_sv_compile_error (&mu_sieve_locus, "%s", s);
  return 0;
}

int
mu_sieve_machine_init (mu_sieve_machine_t *pmach, void *data)
{
  int rc;
  mu_sieve_machine_t mach;
  size_t level;
  
  mach = malloc (sizeof (*mach));
  if (!mach)
    return ENOMEM;
  memset (mach, 0, sizeof (*mach));
  rc = mu_list_create (&mach->memory_pool);
  if (rc)
    {
      free (mach);
      return rc;
    }
  
  mach->data = data;
  mach->error_printer = _sieve_default_error_printer;
  mach->parse_error_printer = _sieve_default_parse_error;

  level = mu_global_debug_level ("sieve");
  if (level)
    {
      mu_debug_create (&mach->debug, mach);
      mu_debug_set_level (mach->debug, level);
    }
  
  *pmach = mach;
  return 0;
}

int
mu_sieve_machine_inherit (mu_sieve_machine_t const parent,
			  mu_sieve_machine_t *pmach)
{
  mu_sieve_machine_t child;
  int rc;
  
  rc = mu_sieve_machine_init (&child, parent->data);
  if (rc)
    return rc;
  child->logger = parent->logger;
  child->debug = parent->debug;
  child->debug_level = parent->debug_level;
  child->debug_printer = parent->debug_printer;
  *pmach = child;
  return 0;
}

int
mu_sieve_machine_dup (mu_sieve_machine_t const in, mu_sieve_machine_t *out)
{
  int rc;
  mu_sieve_machine_t mach;
  
  mach = malloc (sizeof (*mach));
  if (!mach)
    return ENOMEM;
  memset (mach, 0, sizeof (*mach));
  rc = mu_list_create (&mach->memory_pool);
  if (rc)
    {
      free (mach);
      return rc;
    }
  mach->destr_list = NULL;
  mach->test_list = NULL;
  mach->action_list = NULL;
  mach->comp_list = NULL;

  mach->progsize = in->progsize;
  mach->prog = in->prog;

  mach->pc = 0;
  mach->reg = 0;
  mach->stack = NULL;

  mach->debug_level = in->debug_level;
  
  mach->data = in->data;
  mach->error_printer = in->error_printer;
  mach->parse_error_printer = in->parse_error_printer;
  mach->debug_printer = in->debug_printer;
  mach->logger = in->logger;
  mach->debug = in->debug;
  mach->daemon_email = in->daemon_email;

  *out = mach;
  return 0;
}

void
mu_sieve_set_error (mu_sieve_machine_t mach, mu_sieve_printf_t error_printer)
{
  mach->error_printer = error_printer ?
                           error_printer : _sieve_default_error_printer;
}

void
mu_sieve_set_parse_error (mu_sieve_machine_t mach, mu_sieve_parse_error_t p)
{
  mach->parse_error_printer = p ? p : _sieve_default_parse_error;
}

void
mu_sieve_set_debug (mu_sieve_machine_t mach, mu_sieve_printf_t debug)
{
  mach->debug_printer = debug;
}

void
mu_sieve_set_debug_object (mu_sieve_machine_t mach, mu_debug_t dbg)
{
  mach->debug = dbg;
}

void
mu_sieve_set_debug_level (mu_sieve_machine_t mach, int level)
{
  mach->debug_level = level;
}

void
mu_sieve_set_logger (mu_sieve_machine_t mach, mu_sieve_action_log_t logger)
{
  mach->logger = logger;
}

mu_mailer_t
mu_sieve_get_mailer (mu_sieve_machine_t mach)
{
  if (!mach->mailer)
    {
      mu_mailer_create (&mach->mailer, NULL);
      if (mach->debug)
	mu_mailer_set_debug (mach->mailer, mach->debug);
    }

  return mach->mailer;
}

void
mu_sieve_set_mailer (mu_sieve_machine_t mach, mu_mailer_t mailer)
{
  mu_mailer_destroy (&mach->mailer);
  mach->mailer = mailer;
}

#define MAILER_DAEMON_PFX "MAILER-DAEMON@"

char *
mu_sieve_get_daemon_email (mu_sieve_machine_t mach)
{
  if (!mach->daemon_email)
    {
      const char *domain = NULL;
      
      mu_get_user_email_domain (&domain);
      mach->daemon_email = mu_sieve_malloc (mach,
					 sizeof(MAILER_DAEMON_PFX) +
					 strlen (domain));
      sprintf (mach->daemon_email, "%s%s", MAILER_DAEMON_PFX, domain);
    }
  return mach->daemon_email;
}

void
mu_sieve_set_daemon_email (mu_sieve_machine_t mach, const char *email)
{
  mu_sieve_mfree (mach, (void *)mach->daemon_email);
  mach->daemon_email = mu_sieve_mstrdup (mach, email);
}

struct sieve_destr_record
{
  mu_sieve_destructor_t destr;
  void *ptr;
};

int
mu_sieve_machine_add_destructor (mu_sieve_machine_t mach,
				 mu_sieve_destructor_t destr,
				 void *ptr)
{
  struct sieve_destr_record *p;

  if (!mach->destr_list && mu_list_create (&mach->destr_list))
    return 1;
  p = mu_sieve_malloc (mach, sizeof (*p));
  if (!p)
    return 1;
  p->destr = destr;
  p->ptr = ptr;
  return mu_list_prepend (mach->destr_list, p);
}

static int
_run_destructor (void *data, void *unused)
{
  struct sieve_destr_record *p = data;
  p->destr (p->ptr);
  return 0;
}

void
mu_sieve_machine_destroy (mu_sieve_machine_t *pmach)
{
  mu_sieve_machine_t mach = *pmach;
  mu_mailer_destroy (&mach->mailer);
  mu_list_do (mach->destr_list, _run_destructor, NULL);
  mu_list_destroy (&mach->destr_list);
  mu_list_destroy (&mach->action_list);
  mu_list_destroy (&mach->test_list);
  mu_list_destroy (&mach->comp_list);
  mu_list_destroy (&mach->source_list);
  mu_sieve_slist_destroy (&mach->memory_pool);
  free (mach);
  *pmach = NULL;
}

static int
string_comp (const void *item, const void *value)
{
  return strcmp (item, value);
}

void
mu_sieve_machine_begin (mu_sieve_machine_t mach, const char *file)
{
  mu_sieve_machine = mach;
  mu_sieve_error_count = 0;
  mu_sv_code_instr (NULL);

  mu_list_create (&mach->source_list);
  mu_list_set_comparator (mach->source_list, string_comp);
  
  mu_sv_register_standard_actions (mach);
  mu_sv_register_standard_tests (mach);
  mu_sv_register_standard_comparators (mach);
}

void
mu_sieve_machine_finish (mu_sieve_machine_t mach)
{
  mu_sv_code_instr (NULL);
}

int
mu_sieve_compile (mu_sieve_machine_t mach, const char *name)
{
  int rc;
  
  mu_sieve_machine_begin (mach, name);

  if (mu_sv_lex_begin (name) == 0)
    {
      rc = yyparse ();
      if (mu_sieve_error_count)
	rc = 1;
      mu_sv_lex_finish ();
    }
  else
    rc = 1;
  
  mu_sieve_machine_finish (mach);
  return rc;
}

int
mu_sieve_compile_buffer (mu_sieve_machine_t mach,
			 const char *buf, int bufsize,
			 const char *fname, int line)
{
  int rc;
  
  mu_sieve_machine_begin (mach, fname);

  if (mu_sv_lex_begin_string (buf, bufsize, fname, line) == 0)
    {
      rc = yyparse ();
      if (mu_sieve_error_count)
	rc = 1;
      mu_sv_lex_finish ();
    }
  else
    rc = 1;
  
  mu_sieve_machine_finish (mach);
  return rc;
}

static void
_branch_fixup (size_t start, size_t end)
{
  size_t prev = mu_sieve_machine->prog[start].pc;
  if (!prev)
    return;
  branch_fixup (prev, end);
  mu_sieve_machine->prog[prev].pc = end - prev - 1;
}

static void
branch_fixup (size_t start, size_t end)
{
  _branch_fixup (start, end);
  mu_sieve_machine->prog[start].pc = end - start - 1;
}

static int
_sieve_default_error_printer (void *unused, const char *fmt, va_list ap)
{
  return mu_verror (fmt, ap);
}

static int
_sieve_default_parse_error (void *unused, const char *filename, int lineno,
			    const char *fmt, va_list ap)
{
  mu_debug_t debug;

  mu_diag_get_debug (&debug);
  if (filename)
    mu_debug_set_locus (debug, filename, lineno);
  mu_diag_vprintf (MU_DIAG_ERROR, fmt, ap);
  mu_diag_printf (MU_DIAG_ERROR, "\n");
  mu_debug_set_locus (debug, NULL, 0);
  return 0;
}

