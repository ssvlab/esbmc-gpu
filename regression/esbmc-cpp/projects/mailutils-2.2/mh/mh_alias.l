%top {
/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2005, 2006, 2007, 2009, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
}

%{
#include <mh.h>
#include <mh_alias.h>
#include <sys/stat.h>
#include <mailutils/cctype.h>
#include <mu_asprintf.h>
  
char *ali_filename;  
size_t ali_line_num;  
ino_t ali_source_inode;

void
va_ali_parse_error_loc (const char *name, size_t line,
			      const char *fmt, va_list ap)
{
  char *buf;

  vasprintf (&buf, fmt, ap);
  if (name)
    mu_error ("%s:%lu: %s", name, (unsigned long) line,  buf);
  else
    mu_error ("%s", buf);
  free (buf);
}

void
ali_parse_error_loc (const char *name, size_t line, const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  va_ali_parse_error_loc (name, line, fmt, ap);
  va_end (ap);
}

void
ali_parse_error (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  va_ali_parse_error_loc (ali_filename, ali_line_num, fmt, ap);
  va_end (ap);
}

int
yyerror (char *s)
{
  ali_parse_error ("%s", s);
  return 0;
}
 
#ifdef FLEX_SCANNER
#define xinput() (yyin ? getc(yyin) : EOF)
#undef YY_INPUT
#define YY_INPUT(buf,result,max_size)  do { \
        int i;                                  \
        for (i = 0; i < max_size; i++) {        \
                int ch = xinput();              \
                if (ch == EOF)                  \
                        break;                  \
                buf[i] = ch;                    \
        }                                       \
        result = i;                             \
} while (0) 
#define LEX_BUFFER_STATE YY_BUFFER_STATE
#define SET_BUFFER_STATE(s) do { \
        (s) = YY_CURRENT_BUFFER; \
        yy_switch_to_buffer(yy_create_buffer(yyin, YY_BUF_SIZE)); \
} while (0)
#define RESTORE_BUFFER_STATE(s) do { \
        yy_delete_buffer(YY_CURRENT_BUFFER); \
        yy_switch_to_buffer(s); \
} while (0)

#else
/* AT&T Lex */
                                               
static void lex_set_buffer (FILE *fp);
static void lex_delete_buffer (LEX_BUFFER_STATE buf);
static int xinput (void); 
static int xunput (void);
        
#undef unput
#define unput(c) xunput(c)
#undef input
#define input() xinput()

#define LEX_BUF_SIZE 16384
#define LEX_PUTBACK_SIZE 32
                                               
typedef struct {
  FILE *yyin;
  char *buffer;
  size_t bufsize;
  size_t level;
  char *ptr;
  char *putback;
  size_t pb_size;
  size_t pb_level;
} LEX_BUFFER_STATE;
LEX_BUFFER_STATE current_buffer;
 
#define SET_BUFFER_STATE(s) do { \
        (s) = current_buffer;    \
        lex_set_buffer(yyin);    \
} while (0)
#define RESTORE_BUFFER_STATE(s) do { \
        lex_delete_buffer(current_buffer); \
        current_buffer = (s); \
        yyin = current_buffer.yyin;  \
} while (0)
                                    
void
lex_set_buffer (FILE *fp)
{
  char *buf;
  size_t size;
        
  for (size = LEX_BUF_SIZE; size > 1; size /= 2)
    if (buf = malloc (size))
      break;
  
  if (!buf)
    {
      ali_parse_error (_("not enough memory"));
      abort ();
    }

  current_buffer.yyin = yyin;
  current_buffer.buffer = buf;
  current_buffer.bufsize = size;
  current_buffer.level = 0;
  current_buffer.ptr = current_buffer.buffer;
  current_buffer.pb_size = current_buffer.pb_level = 0;
  current_buffer.putback = NULL;
}
                
void
lex_delete_buffer (LEX_BUFFER_STATE buf)
{
  free (buf.buffer);
  if (buf.putback)
    free (buf.putback);
}

int
xinput ()
{
  if (!yyin)
    return EOF;
  
  if (current_buffer.pb_level) 
    return current_buffer.putback[--current_buffer.pb_level];

  if (current_buffer.level <= 0)
    {
      int n;

      if (feof (yyin))
	return 0;
      n = fread (current_buffer.buffer, 1,
		 current_buffer.bufsize, yyin);
      if (n <= 0)
	return 0;
      current_buffer.level = n;
      current_buffer.ptr = current_buffer.buffer;
    }
  current_buffer.level--;
  return *current_buffer.ptr++;
}

int
xunput (int c)
{
  if (current_buffer.pb_level == current_buffer.pb_size)
    {
      char *putback;
      current_buffer.pb_size += LEX_PUTBACK_SIZE;
      putback = xmalloc (current_buffer.pb_size);
      memcpy (putback, current_buffer.putback,
	      current_buffer.pb_level);
      free (current_buffer.putback);
      current_buffer.putback = putback;
    }
  current_buffer.putback[current_buffer.pb_level++] = c;
  return c;
}
                
#endif                                         

struct buffer_ctx {
  struct buffer_ctx *prev;
  char *filename;
  int line;
  ino_t i_node;
  FILE *yyin;
  LEX_BUFFER_STATE state;
};

static struct buffer_ctx *context_stack;

static struct buffer_ctx *
ctx_lookup (ino_t ino)
{
  struct buffer_ctx *ctx;

  for (ctx = context_stack; ctx; ctx = ctx->prev)
    if (ctx->i_node == ino)
      break;
  return ctx;
}
        
static int
push_source (const char *name, int fail)
{
  FILE *fp;
  struct buffer_ctx *ctx;
  struct stat st;
  char *filename;

  filename = mh_expand_name (NULL, name, 0);
  if (stat (filename, &st))
    {
      if (fail)
	ali_parse_error (_("can't stat `%s': %s"), filename, strerror (errno));
      free (filename);
      return 1;
    }

  if (ali_filename && st.st_ino == ali_source_inode)
    {
      ali_parse_error (_("recursive inclusion"));
      free (filename);
      return 1;
    }
  if ((ctx = ctx_lookup (st.st_ino)))
    {
      ali_parse_error (_("recursive inclusion"));
      if (ctx->prev)
	ali_parse_error_loc (ctx->prev->filename, ctx->prev->line,
                             _("`%s' already included here"),
                             filename);
      else
	ali_parse_error (_("`%s' already included at top level"),
			 filename);
      free (filename);
      return 1;
    }
                
  fp = fopen (filename, "r");
  if (!fp)
    {
      ali_parse_error (_("can't open `%s': %s"), filename, strerror (errno));
      free (filename);
      return 1;
    }
  
  /* Push current context */
  if (ali_filename)
    {
      ctx = xmalloc (sizeof (*ctx));
      ctx->filename = ali_filename;
      ctx->line = ali_line_num;
      ctx->i_node = ali_source_inode;
      ctx->yyin = yyin;
      ctx->prev = context_stack;
      context_stack = ctx;
      
      /* Switch to the new context */
      yyin = fp;
      SET_BUFFER_STATE (ctx->state);
    }
  else
    {
#ifdef FLEX_SCANNER
      yyrestart (fp);
#else           
      yyin = fp;
      lex_set_buffer (yyin);
#endif
    }
  ali_filename = filename;
  ali_line_num = 1;
  ali_source_inode = st.st_ino;
  return 0;
}

static int
pop_source ()
{
  struct buffer_ctx *ctx;

  if (yyin)
    fclose (yyin);
#ifndef FLEX_SCANNER
  lex_delete_buffer (current_buffer);
#endif
  if (ali_filename)
    free (ali_filename);
  ali_filename = NULL;
  if (!context_stack)
    {
      yyin = NULL;
      return 1;
    }
  /* Restore previous context */
  ali_filename = context_stack->filename;
  ali_line_num = context_stack->line + 1; /* < line did not increment it */
  ali_source_inode = context_stack->i_node;
  RESTORE_BUFFER_STATE (context_stack->state);
  ctx = context_stack->prev;
  free (context_stack);
  context_stack = ctx;
  return 0;
}

extern int yyparse (void);

%}
WS [ \t]+
WORD [^ \t\n,:;<+=\*]+
SPEC [,:;+=\*]
%s VERBATIM
%%
\\\n           { ali_line_num++; }
\n             { ali_line_num++; return '\n';}
^[ \t]*\;.*\n  ali_line_num++; 
^[ \t]*{WORD}\*  { char *p;
                   for (p = yytext; p < yytext + yyleng; p++)
		     if (!mu_isspace (*p))
		       break;
		   yylval.string = strdup (p);
		   return STRING;}
{WS}           ;
{WORD}         { yylval.string = strdup (yytext); return STRING;}
^{WS}?"<"{WS}?{WORD} {
                 char *p;
                 for (p = yytext; p < yytext + yyleng && mu_isblank(*p); p++)
		   ;
                 for (p++; p < yytext + yyleng; p++)
		   if (!mu_isspace (*p))
		     break;
                 push_source (p, 1); }
{SPEC}         return yytext[0];
<VERBATIM>[^ \t\n,:;+=\*][^\n,]* {
                    yylval.string = xmalloc (yyleng + 1);
 		    memcpy(yylval.string, yytext, yyleng);
		    yylval.string[yyleng] = 0;
		    return STRING;}
.              { char *p;
                 asprintf (&p,
			   _("Stray character %03o in alias file"), yytext[0]);
                 yyerror (p);
		 free (p); }
%%
		 
int
yywrap ()
{
  return pop_source();
}

/* Parses the named alias file */
int
mh_alias_read (char *name, int fail)
{
  extern int yydebug;
  char *p = getenv("ALI_YYDEBUG");

  if (p && *p > '0' && *p < '9')
    yydebug = 1;
  
  if (push_source (name, fail))
    return 1;
  if (yydebug)
    fprintf (stderr, "Starting parse of %s\n", name);
  return yyparse ();
}

void
ali_verbatim (int enable)
{
  if (enable)
    BEGIN(VERBATIM);
  else
    BEGIN(INITIAL);
}
  
