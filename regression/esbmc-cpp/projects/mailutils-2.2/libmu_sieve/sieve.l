%top {
/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2004, 2005, 2007, 2008, 2009,
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
}

%{
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  
#include <sys/file.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <mailutils/cctype.h>
#include <mailutils/argcv.h>
#include <sieve-priv.h>
#include <sieve-gram.h>
  
mu_sieve_locus_t mu_sieve_locus;
ino_t sieve_source_inode;

static mu_list_t string_list;
static char *multiline_delimiter;
static int strip_tabs; 

static int number (void);
static int string (void);
static void line_begin (void);
static void line_add (char *text, size_t len);
static void line_finish (void);
static void multiline_begin (void);
static void multiline_add (char *);
static void multiline_finish (void);
static char *multiline_strip_tabs (char *text);
static void ident (const char *text);
static void sieve_include (void);
static void sieve_searchpath (void);
static char *str_unescape (char *text, size_t len);
static int isemptystr (char *text);

static const char *input_string_ptr;
static int input_string_level; 
 
static int
fillbuf(char *buf, int max_size)
{
  if (input_string_ptr)
    {
      int n = input_string_level;
      if (n > max_size)
	n = max_size;
      memcpy (buf, input_string_ptr, n);
      input_string_ptr += n;
      input_string_level -= n;
      return n;
    }
  else if (feof (yyin))
    return 0;
  else
    return fread (buf, 1, max_size, yyin);
} 
 
#undef YY_INPUT
#define YY_INPUT(buf,result,max_size) result = fillbuf(buf, max_size)
#define LEX_BUFFER_STATE YY_BUFFER_STATE
#define SET_BUFFER_STATE(s) do { \
        (s) = YY_CURRENT_BUFFER; \
        yy_switch_to_buffer(yy_create_buffer(yyin, YY_BUF_SIZE)); \
} while (0)
#define RESTORE_BUFFER_STATE(s) do { \
        yy_delete_buffer(YY_CURRENT_BUFFER); \
        yy_switch_to_buffer(s); \
} while (0)

struct buffer_ctx {
  struct buffer_ctx *prev;
  mu_sieve_locus_t locus;
  ino_t i_node;
  FILE *yyin;
  LEX_BUFFER_STATE state;
};

static struct buffer_ctx *context_stack;

static struct buffer_ctx *ctx_lookup (ino_t ino);
static int push_source (const char *name);
static int pop_source (void);

struct buffer_ctx *
ctx_lookup (ino_t ino)
{
  struct buffer_ctx *ctx;

  for (ctx = context_stack; ctx; ctx = ctx->prev)
    if (ctx->i_node == ino)
      break;
  return ctx;
}
        
int
push_source (const char *name)
{
  FILE *fp;
  struct buffer_ctx *ctx;
  struct stat st;
        
  if (stat (name, &st))
    {
      mu_sv_compile_error (&mu_sieve_locus,
                           _("cannot stat `%s': %s"), name, strerror (errno));
      return 1;
    }

  if (mu_sieve_locus.source_file && st.st_ino == sieve_source_inode)
    {
      yyerror (_("recursive inclusion"));
      return 1;
    }
  if ((ctx = ctx_lookup (st.st_ino)))
    {
      yyerror (_("recursive inclusion"));
      if (ctx->prev)
	mu_sv_compile_error (&ctx->prev->locus,
                             _("`%s' already included here"),
                             name);
      else
	mu_sv_compile_error (&mu_sieve_locus, 
                             _("`%s' already included at top level"),
		             name);
      return 1;
    }
                
  fp = fopen (name, "r");
  if (!fp)
    {
      mu_sv_compile_error (&mu_sieve_locus, 
                           _("cannot open file `%s': %s"),
			   name, strerror (errno));
      return 1;
    }

  /* Push current context */
  if (mu_sieve_locus.source_file)
    {
      ctx = mu_sieve_alloc (sizeof (*ctx));
      ctx->locus = mu_sieve_locus;
      ctx->i_node = sieve_source_inode;
      ctx->yyin = yyin;
      ctx->prev = context_stack;
      context_stack = ctx;
      
      /* Switch to the new context */
      yyin = fp;
      SET_BUFFER_STATE (ctx->state);
    }
  else
    yyrestart (fp);
  
  mu_sieve_locus.source_file = strdup (name);
  mu_sieve_locus.source_line = 1;
  sieve_source_inode = st.st_ino;

  mu_sv_change_source ();
  
  return 0;
}

int
pop_source ()
{
  struct buffer_ctx *ctx;

  if (yyin)
    fclose (yyin);

  if (mu_sieve_locus.source_file)
    free (mu_sieve_locus.source_file);
  if (!context_stack)
    {
      yyin = NULL;
      mu_sieve_locus.source_file = NULL;
      return 1;
    }
  /* Restore previous context */
  mu_sieve_locus = context_stack->locus;
  mu_sieve_locus.source_line++; /* #include rule did not increment it */
  sieve_source_inode = context_stack->i_node;
  RESTORE_BUFFER_STATE (context_stack->state);
  ctx = context_stack->prev;
  free (context_stack);
  context_stack = ctx;

  mu_sv_change_source ();

  return 0;
}
%}
%x COMMENT ML STR

WS [ \t][ \t]*
IDENT [a-zA-Z_][a-zA-Z_0-9]+
SIZESUF [kKmMgG]

%%
         /* C-style comments */
"/*"         BEGIN(COMMENT);
<COMMENT>[^*\n]*        /* eat anything that's not a '*' */
<COMMENT>"*"+[^*/\n]*   /* eat up '*'s not followed by '/'s */
<COMMENT>\n             ++mu_sieve_locus.source_line;
<COMMENT>"*"+"/"        BEGIN(INITIAL);
         /* Preprocessor directives (an extension) */
#[ \t]*include.*\n      { sieve_include (); }
#[ \t]*searchpath.*\n          { sieve_searchpath (); } 
         /* End-of-line comments */
#.*\n   { mu_sieve_locus.source_line++; }
#.*     /* end-of-file comment */;
         /* Reserved words */
require return REQUIRE; 
if	return IF;      
elsif	return ELSIF;  
else	return ELSE;    
anyof	return ANYOF;   
allof	return ALLOF;   
not     return NOT;
         /* Identifiers */
{IDENT}  { ident (yytext); return IDENT; } 
:{IDENT} { ident (yytext + 1); return TAG; }
         /* Numbers */
0[0-7]*{SIZESUF}*                     { return number (); }
0x[0-9a-fA-F][0-9a-fA-F]+{SIZESUF}*   { return number (); }
[1-9][0-9]*{SIZESUF}*                 { return number (); }
         /* Quoted strings */
\"[^\\"\n]*\"                 { return string (); }
\"[^\\"\n]*\\.    { BEGIN(STR);
                   line_begin ();
		   line_add (str_unescape (yytext + 1, yyleng - 1), 0); }
<STR>[^\\"\n]*\\. { line_add (str_unescape (yytext, yyleng), 0); }
<STR>[^\\"\n]*\" { BEGIN(INITIAL);
                   if (yyleng > 1) 
                     line_add (yytext, yyleng - 1); 
                   line_finish ();
		   return STRING; }
         /* Multiline strings */
text:-?[ \t]*#.*\n       { BEGIN(ML);
                           multiline_begin ();
			   mu_sieve_locus.source_line++; }
text:-?[ \t]*\n          { BEGIN(ML);
                           multiline_begin ();
			   mu_sieve_locus.source_line++; }
text:-?\\?{IDENT}[ \t]*#.*\n { BEGIN(ML); multiline_begin ();
                               mu_sieve_locus.source_line++; }
text:-?\\?{IDENT}[ \t]*\n    { BEGIN(ML); multiline_begin ();
                               mu_sieve_locus.source_line++; }
<ML>#[ \t]*include.*\n    { if (multiline_delimiter[0] == '\\')
                              {
				mu_sieve_locus.source_line++;
                                multiline_add (NULL);
			      }
                            else
                              sieve_include (); }
<ML>.*\n { char *p = multiline_strip_tabs (yytext);
           mu_sieve_locus.source_line++;
	   
           if (strncmp (p, multiline_delimiter, strlen (multiline_delimiter))
	        == 0
	       && isemptystr (p + strlen (multiline_delimiter)))
	     {
	       free (multiline_delimiter);
	       multiline_delimiter = NULL;
	       BEGIN(INITIAL);
	       multiline_finish ();
	       return MULTILINE;
	     }
	    multiline_add (NULL); } 
{WS}     ;
         /* Other tokens */
\n { mu_sieve_locus.source_line++; } 
. return yytext[0];

%%

int
yywrap ()
{
  return pop_source();
}

static char *
get_file_name (char *p, char *endp, int *usepath)
{
  char exp, *name, *startp;
  int n;
  
  if (usepath)
    *usepath = 0;
  switch (*p)
    {
    case '"':
      exp = '"';
      break;

    case '<':
      exp = '>';
      if (usepath)
	*usepath = 1;
      break;

    default:
      yyerror (_("preprocessor syntax"));
      return NULL;
    }

  for (startp = ++p; p < endp && *p != exp; p++)
    ;

  if (*p != exp)
    {
      yyerror (_("missing closing quote in preprocessor statement"));
      return NULL;
    }
  
  n = p - startp;
  name = mu_sieve_alloc (n + 1);
  memcpy (name, startp, n);
  name[n] = 0;
  return name;
}

static int
_try_include (void *item, void *data)
{
  char **dir = data;
  char *name = malloc (strlen (item) + 1 + strlen (*dir) + 1);

  if (!name)
    return 0;
  sprintf (name, "%s/%s", (char*) item, *dir);
  if (access (name, R_OK) == 0)
    {
      *(char**) data = name;
      return 1;
    }
  free (name);
  return 0;
}

static void
sieve_include ()
{
  char *p, *endp = yytext + yyleng, *name;
  int usepath;
  
  p = strstr (yytext, "include");
  for (p += 7; p < endp && mu_isspace (*p); p++)
    ;

  name = get_file_name (p, endp, &usepath);
  if (!name)
    return;
  
  if (usepath && name[0] != '/' && memcmp (name, "..", 2))
    {
      char *p = name;
      if (mu_sieve_include_path
	  && mu_list_do (mu_sieve_include_path, _try_include, &p))
	{
	  push_source (p);
	  free (name);
	  free (p);
	  return;
	}
    }

  push_source (name);
  free (name);
}

static void
sieve_searchpath ()
{
  int append = 0;
  char *p, *endp = yytext + yyleng, *name;
  
  p = strstr (yytext, "searchpath");
  for (p += 10; p < endp && mu_isspace (*p); p++)
    ;
  if (strcmp (p, "add") == 0)
    {
      append = 1;
      for (p += 3; p < endp && mu_isspace (*p); p++)
	;
    }
  name = get_file_name (p, endp, NULL);
  if (name)
    {
      mu_sv_load_add_dir (mu_sieve_machine, name);
      free (name);
    }
}

int
mu_sv_lex_begin (const char *name)
{
  return push_source (name);
}

int
mu_sv_lex_begin_string (const char *buf, int bufsize,
			const char *fname, int line)
{
  if (!fname)
    return 1;
  yyrestart (NULL);

  input_string_ptr = buf;
  input_string_level = bufsize;
  mu_sieve_locus.source_file = strdup (fname);
  mu_sieve_locus.source_line = line;
  sieve_source_inode = 0;

  mu_sv_change_source ();
  return 0;
}

void
mu_sv_lex_finish ()
{
  while (pop_source () == 0)
    ;
}
  
static int
number ()
{
  char *p;
  yylval.number = strtoul (yytext, &p, 0);
  switch (*p)
    {
    case 'k':
    case 'K':
      yylval.number *= 1024L;
      break;
      
    case 'm':
    case 'M':
      yylval.number *= 1024*1024L;
      break;
      
    case 'g':
    case 'G':
      yylval.number *= 1024*1024*1024L;
    }
  return NUMBER;
}

static int
string ()
{
  yylval.string = mu_sieve_malloc (mu_sieve_machine, yyleng - 1);
  memcpy (yylval.string, yytext + 1, yyleng - 2);
  yylval.string[yyleng - 2] = 0;
  return STRING; 
}

static int
isemptystr (char *text)
{
  for (; *text && mu_isspace (*text); text++)
    ;
  return *text == 0;
}

static char *
multiline_strip_tabs (char *text)
{
  if (strip_tabs)
    for (; *text == '\t'; text++)
      ;
  return text;
}

static void
line_add (char *text, size_t len)
{
  char *s;
  
  if (len == 0)
    len = strlen (text);
  s = malloc (len + 1);
  if (!s)
    {
      yyerror (_("not enough memory"));
      exit (1);
    }
  memcpy (s, text, len);
  s[len] = 0;
  mu_list_append (string_list, s);
}

static void
multiline_add (char *s)
{
  if (!s)
    s = multiline_strip_tabs (yytext);
  line_add (s, 0);
}

static void
line_begin ()
{
  int status;

  if (string_list)
    mu_sieve_slist_destroy (&string_list);
  status = mu_list_create (&string_list);
  if (status)
    {
      mu_sv_compile_error (&mu_sieve_locus,
                           "mu_list_create: %s", mu_strerror (status));
      exit (1);
    }
}

static void
multiline_begin ()
{
  char *p = yytext + 5; /* past the text: keyword */

  if (*p == '-')
    {
      strip_tabs = 1;
      p++;
    }
  else
    strip_tabs = 0;

  if (!mu_isspace (*p))
    {
      char *endp;
      int len;
      
      for (endp = p; *endp; endp++)
	if (mu_isspace (*endp))
	  break;

      len = endp - p;
      multiline_delimiter = mu_sieve_alloc (len + 1);
      memcpy (multiline_delimiter, p, len);
      multiline_delimiter[len] = 0;
    }
  else
    {
      multiline_delimiter = strdup (".");
      if (!multiline_delimiter)
	{
	  yyerror (_("not enough memory"));
	  exit (1);
	}
    }

  line_begin ();
}

static void
line_finish ()
{
  mu_iterator_t itr;
  int length = 0;
  char *p;
  
  if (!string_list || mu_list_get_iterator (string_list, &itr))
    return;

  /* Count number of characters in the multiline */
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
    {
      char *s;
      mu_iterator_current (itr, (void **)&s);
      length += strlen (s);
    }

  /* Copy the contents */
  yylval.string = mu_sieve_malloc (mu_sieve_machine, length + 1);
  p = yylval.string;
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      char *s;
      mu_iterator_current (itr, (void **)&s);
      strcpy (p, s);
      p += strlen (s);
      free (s);
    }
  *p = 0;
  mu_iterator_destroy (&itr);
  mu_list_destroy (&string_list);
}

static void
multiline_finish ()
{
  line_finish ();
}

static void
ident (const char *text)
{
  yylval.string = strdup (text);
  if (!yylval.string)
    {
      yyerror (_("not enough memory"));
      exit (1);
    }
}

/* Escapes the last character from yytext */
static char *
str_unescape (char *text, size_t len)
{
  char *str = mu_sieve_alloc (len);
  memcpy (str, text, len - 2);
  str[len - 2] = mu_argcv_unquote_char (text[len - 1]);
  str[len - 1] = 0;
  return str;
}
