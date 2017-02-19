/* clexer -- show lexical structure of a C file.
   Copyright (C) 2010 Sergey Poznyakoff

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>. */

%option noyywrap

%top {
#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
}

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

enum state
  {
    s_init,
    s_multiline,
    s_cookie_multiline,
    s_cookie
  };

enum token
  {
    t_token = 256,
    t_open,
    t_close,
    t_cookie
  };

const char *progname;
char *infilename;
char *outfilename;
int guile_snarfer_mode;
enum state state;
FILE *outfile;
unsigned line_no = 1;

static void
advance_line (const char *text)
{
  for (; *text; text++)
    if (*text == '\n')
      line_no++;
}      

#define outstr(text)                            \
  do						\
    if (state != s_init)			\
      fprintf (outfile, "%s\n", text);		\
  while (0)
 
#define outtok(tname,tval)			\
  do						\
    if (state != s_init)			\
      fprintf (outfile, "(%s . \"%s\")\n",	\
	       tname, tval);			\
  while (0)
  
#define RETTOK(t) { outstr (#t); return t_token; }
#define RETTXT(t,s) { outtok (#t, s); return t_token; }
  
%}

N [0-9]
X [0-9a-fA-F]
O [0-7]
ID [a-zA-Z_][a-zA-Z_0-9]*
IQ (l|L|ll|LL|lL|Ll|u|U)
FQ [fFlL]
E  [Ee][+-]?{N}+
WS [ \t\v\f]

%%
\/\*(\n|[^*]|\*[^/])*\*\/   { advance_line (yytext);
                              outtok ("comment", yytext); }
#.*\n                       { outstr ("hash"); line_no++; }
\n                          { outstr ("eol"); line_no++; }
{WS}+                       ;
\\                          ;
{ID}                        RETTXT (id, yytext);
0[xX]{X}+{IQ}?              RETTXT (int_hex, yytext + 2);
0{O}+{IQ}?                  RETTXT (int_oct, yytext + 1);
{N}+{IQ}?                   RETTXT (int_dec, yytext);
L?\'[^\\']\' |               
L?\'\\[^0xX]\' |
L?\'\\{O}{1,3}\' |
L?\'\\[xX]{X}{1,2}          RETTXT (char, yytext);
{N}+E{FQ}? |
{N}*"."{N}+({E})?{FQ}? |
{N}+"."{N}*({E})?{FQ}?      RETTXT (flo_dec, yytext);
L?\"([^\\\"]|\\.|\\\n)*\"   { outstr (yytext); advance_line (yytext); }

"^"{WS}*"^"                 { fputs ("snarf_cookie\n", outfile);
                              return t_cookie; }
"{"                         { outstr ("brace_open"); return t_open; }
"}"                         { outstr ("brace_close"); return t_close; }
","                         RETTOK (comma); 
":"                         RETTOK (colon); 
"="                         RETTOK (assign);
"("                         RETTOK (paren_open); 
")"                         RETTOK (paren_close); 
"["                         RETTOK (bracket_open);
"]"                         RETTOK (bracket_close);
"&"                         RETTOK (amp); 
"^"                         RETTOK (caret);
"|"                         RETTOK (pipe);
"?"                         RETTOK (question);
"!"                         RETTOK (bang);
"~"                         RETTOK (tilde);
"-"                         RETTOK (minus);
"+"                         RETTOK (plus); 
"*"                         RETTOK (star); 
"/"                         RETTOK (slash);
"%"                         RETTOK (percent);
"..."                       RETTOK (ellipsis);
"->"                        RETTOK (ptr);
"."                         RETTOK (dot);
"*="                        RETTOK (mul_assign);
"/="                        RETTOK (div_assign);
"%="                        RETTOK (mod_assign);
"+="                        RETTOK (add_assign);
"-="                        RETTOK (sub_assign);
"<<="                       RETTOK (shift_left_assign);
">>="                       RETTOK (shift_right_assign);
"&="                        RETTOK (logand_assign);
"|="                        RETTOK (logior_assign); 
"^="                        RETTOK (logxor_assign);
"||"                        RETTOK (or);
"&&"                        RETTOK (and);
"=="                        RETTOK (eq);
"!="                        RETTOK (neq);
">="                        RETTOK (ge);
">>"                        RETTOK (right_shift);
">"                         RETTOK (gt);
"<="                        RETTOK (le);
"<<"                        RETTOK (left_shift);
"<"                         RETTOK (lt);
"++"                        RETTOK (inc);
"--"                        RETTOK (dec);
";"                         RETTOK (semicolon);
.                           { fprintf (stderr,
                                       (isascii (yytext[0]) &&
                                        isprint (yytext[0])) ?
                                         "%s:%u: stray character %c\n" :
                                         "%s:%u: stray character \\%03o",
                                       infilename,
                                       line_no, (unsigned char) yytext[0]);
                              return t_token; }

%%

void
usage()
{
  printf ("\
usage: clexer [OPTIONS] [FILE]\n\
\n\
Show lexical structure of a C source. When given the --snarfer option,\n\
display only lexical tokens produced by Guile snarfer. The output may be\n\
piped to the `guile-tools snarf-check-and-output-texi' command.\n\
\n\
OPTIONS are:\n\
  -s, --snarfer      filter out Guile docstrings\n\
  -o, --output=FILE  write output to FILE\n\
  -h, --help         show this help summary\n\
\n");
  printf ("Report bugs to <gray+clexer@gnu.org.ua>\n");
}

int
main (int argc, char **argv)
{
  int i;
  enum token tok, last_tok;
  
  progname = argv[0];

  for (i = 1; i < argc; i++)
    {
      if (!strcmp (argv[i], "-s") ||
 	  !strcmp (argv[i], "--snarfer") ||
	  !strcmp (argv[i], "--guile-snarfer"))
	{
	  guile_snarfer_mode = 1;
	}
      else if (!strcmp (argv[i], "-h") ||
	       !strcmp (argv[i], "--help"))
	{
	  usage ();
	  exit (0);
	}
#if 0
      else if (!strcmp (argv[i], "-v") ||
	       !strcmp (argv[i], "--version"))
	{
	  version ();
	  exit (0);
	}
#endif
      else if (!strncmp (argv[i], "-o", 2))
	{
	  if (argv[i][2])
	    outfilename = argv[i] + 2;
	  else if (++i < argc)
	    outfilename = argv[i];
	  else
	    {
	      fprintf (stderr, "%s: option '-o' requires an argument\n",
		       progname);
	      exit (1);
	    }
	}
      else if (!strncmp (argv[i], "--output=", 9))
	outfilename = argv[i] + 9;
      else if (!strcmp (argv[i], "--output"))
	{
	  if (++i < argc)
	    outfilename = argv[++i];
	  else
	    {
	      fprintf (stderr, "%s: option '--output' requires an argument\n",
		       progname);
	      exit (1);
	    }
	}
      else if (!strcmp (argv[i], "--"))
	{
	  if (++i < argc)
	    infilename = argv[i];
	  break;
	}
      else if (argv[i][0] == '-')
	{
	  fprintf (stderr, "%s: unknown option: %s\n",
		   progname, argv[i]);
	  exit (1);
	}
      else
	{
	  infilename = argv[i];
	}
    }
  
  if (infilename && strcmp (infilename, "-"))
    {
      FILE *fp = fopen (infilename, "r");
      if (!fp)
	{
	  fprintf (stderr, "%s: cannot open file `%s': %s\n",
		   progname, infilename, strerror (errno));
	  exit (1);
	}
      yyrestart (fp);
    }
  else
    infilename = "stdin";

  if (outfilename)
    {
      outfile = fopen (outfilename, "w");
      if (!outfile)
	{
	  fprintf (stderr, "%s: cannot open file `%s': %s\n",
		   progname, outfilename, strerror (errno));
	  exit (1);
	}
    }
  else
    outfile = stdout;

  state = guile_snarfer_mode ? s_init : s_cookie;
  last_tok = t_token;
  while ((tok = yylex ()) > 0)
    {
      if (!guile_snarfer_mode)
	continue;
      switch (tok)
	{
	case t_token:
	  break;
	  
	case t_open:
	  if (last_tok == t_cookie && state == s_cookie)
	    state = s_multiline;
	  break;
	  
	case t_close:
	  if (last_tok == t_cookie && state == s_cookie_multiline)
	    state = s_init;
	  break;
	  
	case t_cookie:
	  switch (state)
	    {
	    case s_init:
	      state = s_cookie;
	      break;

	    case s_multiline:
	    case s_cookie_multiline:
	      state = s_cookie_multiline;
	      break;

	    case s_cookie:
	      state = s_init;
	    }
	}
      last_tok = tok;
    }

  fclose (outfile);
  exit (0);
}
