%{
/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2004, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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
#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
#include <obstack.h>

int yyerror (const char *s);
int yylex ();
 
static mh_format_t format;     /* Format structure being built */
static size_t pc;              /* Program counter. Poins to current
				  cell in format.prog */
static struct obstack stack;   /* Temporary token storage */

#define FORMAT_INC 64          /* Increase format.prog by that many
				  cells each time pc reaches
				  format.progsize */

static size_t mh_code_op (mh_opcode_t op);
static size_t mh_code_string (char *string);
static size_t mh_code_number (int num);
static size_t mh_code_builtin (mh_builtin_t *bp, int argtype);
static void branch_fixup (size_t pc, size_t tgt); 

  /* Lexical tie-ins */
static int in_escape;       /* Set when inside an escape sequence */
static int want_function;   /* Set when expecting function name */
static int want_arg;        /* Expecting function argument */
%}

%union {
  char *str;
  int num;
  int type;
  struct {
    size_t cond;
    size_t end;
  } elif_list;
  size_t pc;
  mh_builtin_t *builtin;
};
%token <num> NUMBER
%token <str> STRING
%token <builtin> FUNCTION
%token IF ELIF ELSE FI
%token OBRACE CBRACE OCURLY CCURLY
%token <num> FMTSPEC
%token BOGUS
%type <type> cond_expr component funcall item argument escape literal
%type <elif_list> elif_part elif_list fi
%type <pc> cond end else elif else_part zlist list pitem 
%type <builtin> function

%%

input     : list
            {
	      /* nothing: to shut bison up */
	    }
          ;

list      : pitem 
          | list pitem
          ;

pitem     : item
            {
	      switch ($1)
		{
		case mhtype_none:
		  break;
		  
		case mhtype_num:
		  mh_code_op (mhop_num_asgn);
		  mh_code_op (mhop_num_print);
		  break;
		  
		case mhtype_str:
		  mh_code_op (mhop_str_asgn);
		  mh_code_op (mhop_str_print);
		  break;
		  
		default:
		  yyerror (_("INTERNAL ERROR: unexpected item type (please report)"));
		  abort ();
		}
	      $$ = pc;
	    }
          ;

item      : literal
          | escape
            {
	      in_escape = 0;
	    }
          ;

literal   : STRING
            {
	      mh_code_string ($1);
	      $$ = mhtype_str;
	    }
          | NUMBER
            {
	      mh_code_number ($1);
	      $$ = mhtype_num;
	    }	      
          ;

escape    : component
          | funcall
          | cntl
            {
	      $$ = mhtype_none;
	    }
          ;

component : fmtspec OCURLY STRING CCURLY
            {
	      if (mu_c_strcasecmp ($3, "body") == 0)
		{
		  mh_code_op (mhop_body);
		}
	      else
		{
		  mh_code_string ($3);
		  mh_code_op (mhop_header);
		}
	      $$ = mhtype_str;
	    }
          ;

obrace    : OBRACE
            {
	      in_escape++;
	    }
          ;

cbrace    : CBRACE
            {
	      in_escape--;
	    }
          ;

funcall   : fmtspec obrace { want_function = 1;} function { want_function = 0; want_arg = 1;} argument cbrace
            {
	      if ($4)
		{
		  if (!mh_code_builtin ($4, $6))
		    YYERROR;
		  $$ = $4->type;
		}
	      else
		{
		  switch ($6)
		    {
		    default:
		      break;
		  
		    case mhtype_num:
		      mh_code_op (mhop_num_asgn);
		      break;
		  
		    case mhtype_str:
		      mh_code_op (mhop_str_asgn);
		      break;
		    }
		  $$ = mhtype_none;
		}
	    }
          ;

fmtspec   : /* empty */
          | FMTSPEC
            {
	      mh_code_op (mhop_fmtspec);
	      mh_code_op ($1);
	    }
          ;

function  : FUNCTION
          | STRING
            {
	      if (strcmp ($1, "void") == 0)
		{
		  $$ = NULL;
		}
	      else
		{
		  yyerror (_("undefined function"));
		  mu_error ($1);
		  YYERROR;
		}
	    }
          ;

argument  : /* empty */
            {
	      $$ = mhtype_none;
	    }
          | literal
	  | escape
          ;

/*           1   2    3    4      5         6     7 */
cntl      : if cond zlist end elif_part else_part fi
            {
	      size_t start_pc = 0, end_pc = 0;

	      /* Fixup first condition */
	      if ($5.cond)
		MHI_NUM(format.prog[$2]) = $5.cond - $2;
	      else if ($6)
		MHI_NUM(format.prog[$2]) = $6 - $2;
	      else
		MHI_NUM(format.prog[$2]) = $7.cond - $2;

	      /* Link all "false" lists */
	      if ($5.cond)
		{
		  start_pc = $5.end;
		  end_pc = $5.end;
		  while (MHI_NUM(format.prog[end_pc]))
		    end_pc = MHI_NUM(format.prog[end_pc]);
		}

	      if (start_pc)
		MHI_NUM(format.prog[end_pc]) = $4;
	      else
		start_pc = $4;

	      /* Now, fixup the end branches */
	      branch_fixup (start_pc, $7.end);
	      MHI_NUM(format.prog[start_pc]) = $7.end - start_pc;
	    }
          ;

zlist     : /* empty */
            {
	      $$ = pc;
	    }
          | list
          ;

if        : IF
            {
	      in_escape++;
	    }
          ;

fi        : FI
            {
	      /* False branch of an if-block */
	      $$.cond = mh_code_op (mhop_num_asgn);
	      /* Jump over the true branch */
	      mh_code_op (mhop_branch);
	      mh_code_op (2);
	      /* True branch */
	      $$.end = mh_code_op (mhop_num_asgn);
	    }
          ;

elif      : ELIF
            {
	      in_escape++;
	      $$ = pc;
	    }
          ;

end       : /* empty */
            {
	      mh_code_op (mhop_branch);
	      $$ = mh_code_op (0);
	    }
          ;

cond      : cond_expr
            {
	      in_escape--;
	      if ($1 == mhtype_str)
		mh_code_op (mhop_str_branch);
	      else
		mh_code_op (mhop_num_branch);
	      $$ = mh_code_op (0);
	    }
          ;

cond_expr : component
          | funcall
          ;

elif_part : /* empty */
            {
	      $$.cond = 0;
	      $$.end = 0;
	    }
          | elif_list end
            {
	      $$.cond = $1.cond;
	      MHI_NUM(format.prog[$2]) = $1.end;
	      $$.end = $2;
	    }
          ;

elif_list : elif cond zlist
            {
	      $$.cond = $1;
	      MHI_NUM(format.prog[$2]) = pc - $2 + 2;
	      $$.end = 0;
	    }
          | elif_list end elif cond zlist
            {
	      MHI_NUM(format.prog[$4]) = pc - $4 + 2;
	      $$.cond = $1.cond;
	      MHI_NUM(format.prog[$2]) = $1.end;
	      $$.end = $2;
	    }
          ;

else_part : /* empty */
            {
	      $$ = 0;
	    }
          | else list 
	  ;

else      : ELSE
            {
	      $$ = pc;
	    }
          ;

%%

static char *start;
static char *curp;

int
yyerror (const char *s)
{
  int len;
  mu_error ("%s: %s", start, s);
  len = curp - start;
  mu_error ("%*.*s^", len, len, "");
  return 0;
}

#define isdelim(c) (strchr("%<>?|(){} ",c) != NULL)

static int percent;
static int backslash(int c);

int
yylex ()
{
  /* Reset the tie-in */
  int expect_arg = want_arg;
  want_arg = 0;
  
  if (yydebug)
    fprintf (stderr, "[lex at %10.10s]\n", curp);
  if (*curp == '%')
    {
      curp++;
      percent = 1;
      if (mu_isdigit (*curp) || *curp == '-')
	{
	  int num = 0;
	  int flags = 0;

	  if (*curp == '-')
	    {
	      curp++;
	      flags = MH_FMT_RALIGN;
	    }
	  if (*curp == '0')
	    flags |= MH_FMT_ZEROPAD;
	  while (*curp && mu_isdigit (*curp))
	    num = num * 10 + *curp++ - '0';
	  yylval.num = num | flags;
	  return FMTSPEC;
	}
    }

  if (percent)
    {
      percent = 0;
      switch (*curp++)
	{
	case '<':
	  return IF;
	case '>':
	  return FI;
	case '?':
	  return ELIF;
	case '|':
	  return ELSE;
	case '%':
	  return '%';
	case '(':
	  return OBRACE;
	case '{':
	  return OCURLY;
	default:
	  return BOGUS;
      }
    }

  if (in_escape)
    {
      while (*curp && (*curp == ' ' || *curp == '\n'))
	curp++;
      switch (*curp)
	{
	case '(':
	  curp++;
	  return OBRACE;
	case '{':
	  curp++;
	  return OCURLY;
	case '0':case '1':case '2':case '3':case '4':
	case '5':case '6':case '7':case '8':case '9':
	  yylval.num = strtol (curp, &curp, 0);
	  return NUMBER;
	}
    }
  
  switch (*curp)
    {
    case ')':
      curp++;
      return CBRACE;
    case '}':
      curp++;
      return CCURLY;
    case 0:
      return 0;
    }

  do
    {
      if (*curp == '\\')
	{
	  int c = backslash (*++curp);
	  obstack_1grow (&stack, c);
	}
      else
	obstack_1grow (&stack, *curp);
      curp++;
    }
  while (*curp && (expect_arg ? *curp != ')' : !isdelim(*curp)));

  obstack_1grow (&stack, 0);
  yylval.str = obstack_finish (&stack);

  if (want_function)
    {
      int rest;
      mh_builtin_t *bp = mh_lookup_builtin (yylval.str, &rest);
      if (bp)
	{
	  curp -= rest;
	  yylval.builtin = bp;
	  while (*curp && mu_isspace (*curp))
	    curp++;
	  return FUNCTION;
	}
    }
  
  return STRING;
}

void
mh_format_debug (int val)
{
  yydebug = val;
}

int
mh_format_parse (char *format_str, mh_format_t *fmt)
{
  int rc;
  char *p = getenv ("MHFORMAT_DEBUG");

  if (p)
    yydebug = 1;
  start = curp = format_str;
  obstack_init (&stack);
  format.prog = NULL;
  format.progsize = 0;
  pc = 0;
  mh_code_op (mhop_stop);

  in_escape = 0; 
  percent = 0;

  rc = yyparse ();
  mh_code_op (mhop_stop);
  obstack_free (&stack, NULL);
  if (rc)
    {
      mh_format_free (&format);
      return 1;
    }
  *fmt = format;
  return 0;
}

int
backslash(int c)
{
  static char transtab[] = "b\bf\fn\nr\rt\t";
  char *p;
  
  for (p = transtab; *p; p += 2)
    {
      if (*p == c)
	return p[1];
    }
  return c;
}

void
branch_fixup (size_t epc, size_t tgt)
{
  size_t prev = MHI_NUM(format.prog[epc]);
  if (!prev)
    return;
  branch_fixup (prev, tgt);
  MHI_NUM(format.prog[prev]) = tgt - prev;
}


/* Make sure there are at least `count' entries available in the prog
   buffer */
void
prog_reserve (size_t count)
{
  if (pc + count >= format.progsize)
    {
      size_t inc = (count + 1) / FORMAT_INC + 1;
      format.progsize += inc * FORMAT_INC;
      format.prog = xrealloc (format.prog,
			      format.progsize * sizeof format.prog[0]);
    }
}

size_t
mh_code_string (char *string)
{
  int length = strlen (string) + 1;
  size_t count = (length + sizeof (mh_instr_t)) / sizeof (mh_instr_t);
  size_t start_pc = pc;
  
  mh_code_op (mhop_str_arg);
  prog_reserve (count);
  MHI_NUM(format.prog[pc++]) = count;
  memcpy (MHI_STR(format.prog[pc]), string, length);
  pc += count;
  return start_pc;
}
 
size_t
mh_code (mh_instr_t *instr)
{
  prog_reserve (1);
  format.prog[pc] = *instr;
  return pc++;
}

size_t
mh_code_op (mh_opcode_t op)
{
  mh_instr_t instr;
  MHI_OPCODE(instr) = op;
  return mh_code(&instr);
}

size_t
mh_code_number (int num)
{
  mh_instr_t instr;
  size_t ret = mh_code_op (mhop_num_arg);
  MHI_NUM(instr) = num;
  mh_code (&instr);
  return ret;
}

size_t
mh_code_builtin (mh_builtin_t *bp, int argtype)
{
  mh_instr_t instr;
  size_t start_pc = pc;
  if (bp->argtype != argtype)
    {
      if (argtype == mhtype_none)
	{
	  if (bp->optarg)
	    {
	      switch (bp->argtype)
		{
		case mhtype_num:
		  mh_code_op (mhop_num_to_arg);
		  break;
		  
		case mhtype_str:
		  if (bp->optarg == MHA_OPT_CLEAR)
		    mh_code_string ("");
		  /* mhtype_none means that the argument was an escape,
		     which has left its string value (if any) in the
		     arg_str register. Therefore, there's no need to
		     code mhop_str_to_arg */
		  break;
		  
		default:
		  yyerror (_("INTERNAL ERROR: unknown argtype (please report)"));
		  abort ();
		}
	    }
	  else
	    {
	      mu_error (_("missing argument for %s"), bp->name);
	      return 0;
	    }
	}
      else
	{
	  switch (bp->argtype)
	    {
	    case mhtype_none:
	      mu_error (_("extra arguments to %s"), bp->name);
	      return 0;
	      
	    case mhtype_num:
	      mh_code_op (mhop_str_to_num);
	      break;
	      
	    case mhtype_str:
	      mh_code_op (mhop_num_to_str);
	      break;
	    }
	}
    }

  mh_code_op (mhop_call);
  MHI_BUILTIN(instr) = bp->fun;
  mh_code (&instr);
  return start_pc;
}

