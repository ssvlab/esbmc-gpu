%top {
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
}

%{
#include <unistd.h>
#include <stdio.h>
#include <sys/stat.h>
#include <mimeview.h>
#include <mimetypes-decl.h>  
#include <mu_asprintf.h>

static int line_num;
static char *file_name;
static int file_name_alloc; 
 
static struct obstack stack;
static int prev_state;
 
static unsigned 
digit_to_number (char c)
{
  return (unsigned) (c >= '0' && c <= '9' ? c-'0' :
                     c >= 'A' && c <= 'Z' ? c-'A'+10 :
                     c-'a'+10);
}
%}
%x ARGS HEX
X [0-9a-fA-F]
IDENT [a-zA-Z_\.][a-zA-Z0-9_\.-]*
WS [ \t]*
%%
     /* Comments */
<INITIAL>#.*\n       { line_num++; }
<INITIAL>#.*         /* end-of-file comment */;
     /* Tokens */
\\\n                 { line_num++; }
\n                   { line_num++; return EOL; }
{WS}                 ;
{IDENT}              {
  obstack_grow (&stack, yytext, yyleng);
  yylval.string.len = obstack_object_size (&stack);
  obstack_1grow (&stack, 0);
  yylval.string.ptr = obstack_finish (&stack);
  return IDENT;
}
<INITIAL>{IDENT}"("           {
  obstack_grow (&stack, yytext, yyleng-1);
  yylval.string.len = obstack_object_size (&stack);
  obstack_1grow (&stack, 0);
  yylval.string.ptr = obstack_finish (&stack);
  BEGIN(ARGS);
  return IDENT_L;
}
<INITIAL,ARGS>\"[^\\"\n]*\"        {
  obstack_grow (&stack, yytext+1, yyleng-2);
  yylval.string.len = obstack_object_size (&stack);
  obstack_1grow (&stack, 0);
  yylval.string.ptr = obstack_finish (&stack);
  return STRING;
}
<INITIAL,ARGS>"<" {
  prev_state = YYSTATE;
  BEGIN(HEX);
}
<ARGS>[^ \t<\\\n),]+/[),] {
  obstack_grow (&stack, yytext, yyleng);
  yylval.string.len = obstack_object_size (&stack);
  obstack_1grow (&stack, 0);
  yylval.string.ptr = obstack_finish (&stack);
  return STRING;
}
<ARGS>[^ \t<\\\n),]+< {
  obstack_grow (&stack, yytext, yyleng);
  prev_state = YYSTATE;
  BEGIN(HEX);
}
<INITIAL>[^ \t<\\\n)+,&]/[ \t\\\n)+,&] {
  obstack_grow (&stack, yytext, yyleng);
  yylval.string.len = obstack_object_size (&stack);
  obstack_1grow (&stack, 0);
  yylval.string.ptr = obstack_finish (&stack);
  return STRING;
}
<ARGS>[^ \t<\\\n),]/[ \t\\\n] {
  obstack_grow (&stack, yytext, yyleng);
  yylval.string.len = obstack_object_size (&stack);
  obstack_1grow (&stack, 0);
  yylval.string.ptr = obstack_finish (&stack);
  return STRING;
}
<HEX>{X}{X} {
  int c = digit_to_number (yytext[0])*16 + digit_to_number (yytext[1]);
  obstack_1grow (&stack, c);
}
<HEX>">"/[ \t\\\n,)] {
  BEGIN(prev_state);
  yylval.string.len = obstack_object_size (&stack);
  obstack_1grow (&stack, 0);
  yylval.string.ptr = obstack_finish (&stack);
  return STRING;
}
<HEX>">" {
  BEGIN(prev_state);
}
  /* Special cases: && and ||. Docs don't say anything about them, but
     I've found them in my mime.types file...         --Sergey */
"&&"  return '+';
"||"  return ',';
  /* Operators */
"!"|"+"|","|"("|")"|"/"  return yytext[0];
<ARGS>","        return yytext[0];
<ARGS>")"        { BEGIN(INITIAL); return yytext[0]; }
<INITIAL,ARGS,HEX>. {
 fprintf (stderr, "Invalid character '%c', state %d\n", yytext[0], YYSTATE);
 abort();
}
%%

void
mimetypes_lex_debug (int level)
{
  yy_flex_debug = level;
}

int
mimetypes_open (const char *name)
{
  struct stat st;
  if (stat (name, &st))
    {
      mu_error (_("cannot stat `%s': %s"), name, mu_strerror (errno));
      return -1;
    }
  
  if (S_ISDIR (st.st_mode))
    {
      asprintf (&file_name, "%s/mime.types", name);
      file_name_alloc = 1;
    }
  else
    {
      file_name = (char*) name;
      file_name_alloc = 0;
    }
  
  yyin = fopen (file_name, "r");
  if (!yyin)
    {
      mu_error (_("Cannot open `%s': %s"), file_name, mu_strerror (errno));
      if (file_name_alloc)
	{
	  free (file_name);
	  file_name_alloc = 0;
	}
      return -1;
    }
  line_num = 1;
  obstack_init (&stack);
  return 0;
}

void
mimetypes_close ()
{
  fclose (yyin);
  if (file_name_alloc)
    {
      free (file_name);
      file_name_alloc = 0;
    }
}

int
yyerror (char *s)
{
  mu_error ("%s:%d: %s", file_name, line_num, s);
  return 0;
}

int
yywrap ()
{
  return 1;
}

struct mimetypes_string
mimetypes_append_string2 (struct mimetypes_string *s1,
			  char c,
			  struct mimetypes_string *s2)
{
  struct mimetypes_string r;

  r.len = s1->len + s2->len + 1;
  obstack_grow (&stack, s1->ptr, s1->len);
  obstack_1grow (&stack, c);
  obstack_grow (&stack, s2->ptr, s2->len);
  obstack_1grow (&stack, 0);
  r.ptr = obstack_finish (&stack);
  return r;
}

struct mimetypes_string *     
mimetypes_string_dup (struct mimetypes_string *s)
{
  obstack_grow (&stack, s, sizeof *s);
  return obstack_finish (&stack);
}

void *
mimetypes_malloc (size_t size)
{
  return obstack_alloc(&stack, size);
}

void
reset_lex ()
{
  BEGIN(INITIAL);
}


