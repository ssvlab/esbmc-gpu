/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2007, 2010 Free Software Foundation, Inc.

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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#include <mailutils/mailutils.h>
#include <xalloc.h>
#include <fnmatch.h>
#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
#include <obstack.h>  

struct mimetypes_string
{
  char *ptr;
  size_t len;
};

int mimetypes_yylex (void);
int mimetypes_yyerror (char *s);

int mimetypes_open (const char *name);
void mimetypes_close (void);
int mimetypes_parse (const char *name);
void mimetypes_gram_debug (int level);
void mimetypes_lex_debug (int level);
void mimetypes_lex_init (void);
void reset_lex (void);
void *mimetypes_malloc (size_t size);

struct mimetypes_string mimetypes_append_string2 (struct mimetypes_string *s1,
						  char c,
						  struct mimetypes_string *s2);
struct mimetypes_string *mimetypes_string_dup (struct mimetypes_string *s);

const char *get_file_type (void);

extern char *mimeview_file;
extern FILE *mimeview_fp;    
extern int debug_level;

#define DEBUG(l,f) if (debug_level > (l)) printf f
