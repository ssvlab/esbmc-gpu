/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2007, 2009, 2010
   Free Software Foundation, Inc.

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

%{
#include "mail.h"

#include <stdio.h>
#include <stdlib.h>

#include <xalloc.h>

/* Defined in <limits.h> on some systems, but redefined in <regex.h>
   if we are using GNU's regex. So, undef it to avoid duplicate definition
   warnings. */

#ifdef RE_DUP_MAX
# undef RE_DUP_MAX
#endif
#include <regex.h>

struct header_data
{
  char *header;
  char *expr;
};

static msgset_t *msgset_select (int (*sel) (mu_message_t, void *),
				     void *closure, int rev,
				     unsigned int max_matches);
static int select_header (mu_message_t msg, void *closure);
static int select_body (mu_message_t msg, void *closure);
static int select_type (mu_message_t msg, void *closure);
static int select_sender (mu_message_t msg, void *closure);
static int select_deleted (mu_message_t msg, void *closure);
static int check_set (msgset_t **pset);
 
int yyerror (const char *);
int yylex  (void);

static int msgset_flags = MSG_NODELETED;
static msgset_t *result;
%}

%union {
  char *string;
  int number;
  int type;
  msgset_t *mset;
}

%token <type> TYPE
%token <string> IDENT REGEXP HEADER BODY
%token <number> NUMBER
%type <mset> msgset msgspec msgexpr msg rangeset range partno number
%type <string> header

%%

input    : /* empty */
           {
	     result = msgset_make_1 (get_cursor ());
	   }
         | '.'
           {
	     result = msgset_make_1 (get_cursor ());
	   }
         | msgset
           {
	     result = $1;
	   }
         | '^'
           {
	     result = msgset_select (select_deleted, NULL, 0, 1);
	   }
         | '$'
           {
	     result = msgset_select (select_deleted, NULL, 1, 1);
	   }
         | '*'
           {
	     result = msgset_select (select_deleted, NULL, 0, total);
	   }
         | '-'
           {
	     result = msgset_select (select_deleted, NULL, 1, 1);
	   }
         | '+'
           {
	     result = msgset_select (select_deleted, NULL, 0, 1);
	   }
         ;

msgset   : msgexpr
         | msgset ',' msgexpr
           {
	     $$ = msgset_append ($1, $3);
	   }
         | msgset msgexpr
           {
	     $$ = msgset_append ($1, $2);
	   }
         ;

msgexpr  : msgspec
           {
	     $$ = $1;
	     if (check_set (&$$))
	       YYABORT;
	   }
         | '{' msgset '}'
           {
	     $$ = $2;
	   }
         | '!' msgexpr
           {
	     $$ = msgset_negate ($2);
	   }
         ;

msgspec  : msg
         | msg '[' rangeset ']'
           {
	     $$ = msgset_expand ($1, $3);
	     msgset_free ($1);
	     msgset_free ($3);
	   }
         | range
         ;

msg      : header REGEXP /* /.../ */
           {
	     struct header_data hd;
	     hd.header = $1;
	     hd.expr   = $2;
	     $$ = msgset_select (select_header, &hd, 0, 0);
	     if ($1)
	       free ($1);
	     free ($2);
	   }
         | BODY
           {
	     $$ = msgset_select (select_body, $1, 0, 0);
	   }
         | TYPE  /* :n, :d, etc */
           {
	     if (strchr ("dnorTtu", $1) == NULL)
	       {
		 yyerror (_("unknown message type"));
		 YYERROR;
	       }
	     $$ = msgset_select (select_type, (void *)&$1, 0, 0);
	   }
         | IDENT /* Sender name */
           {
	     $$ = msgset_select (select_sender, (void *)$1, 0, 0);
	     free ($1);
	   }
         ;

header   : /* empty */
           {
	     $$ = NULL;
	   }
         | HEADER
           {
	     $$ = $1;
	   }
         ;

rangeset : range
         | rangeset ',' range
           {
	     $$ = msgset_append ($1, $3);
	   }
         | rangeset range
           {
	     $$ = msgset_append ($1, $2);
	   }
         ;

range    : number
         | NUMBER '-' number
           {
	     if ($3->npart == 1)
	       {
		 $$ = msgset_range ($1, $3->msg_part[0]);
	       }
	     else
	       {
		 $$ = msgset_range ($1, $3->msg_part[0]-1);
		 if (!$$)
		   YYERROR;
		 msgset_append ($$, $3);
	       }
	   }
         | NUMBER '-' '*'
           {
	     $$ = msgset_range ($1, total);
	   }
         ;

number   : partno
         | partno '[' rangeset ']'
           {
	     $$ = msgset_expand ($1, $3);
	     msgset_free ($1);
	     msgset_free ($3);
	   }
         ;

partno   : NUMBER
           {
	     $$ = msgset_make_1 ($1);
	   }
         | '(' rangeset ')'
           {
	     $$ = $2;
	   }
         ;
%%

static int xargc;
static char **xargv;
static int cur_ind;
static char *cur_p;

int
yyerror (const char *s)
{
  fprintf (stderr, "%s: ", xargv[0]);
  fprintf (stderr, "%s", s);
  if (!cur_p)
    fprintf (stderr, _(" near end"));
  else if (*cur_p == 0)
    {
      int i =  (*cur_p == 0) ? cur_ind + 1 : cur_ind;
      if (i == xargc)
	fprintf (stderr, _(" near end"));
      else
	fprintf (stderr, _(" near %s"), xargv[i]);
    }
  else
    fprintf (stderr, _(" near %s"), cur_p);
  fprintf (stderr, "\n");
  return 0;
}

int
yylex()
{
  if (cur_ind == xargc)
    return 0;
  if (!cur_p)
    cur_p = xargv[cur_ind];
  if (*cur_p == 0)
    {
      cur_ind++;
      cur_p = NULL;
      return yylex ();
    }

  if (mu_isdigit (*cur_p))
    {
      yylval.number = strtoul (cur_p, &cur_p, 10);
      return NUMBER;
    }

  if (mu_isalpha (*cur_p))
    {
      char *p = cur_p;
      int len;

      while (*cur_p && *cur_p != ',' && *cur_p != ':') 
	cur_p++;
      len = cur_p - p + 1;
      yylval.string = xmalloc (len);
      memcpy (yylval.string, p, len-1);
      yylval.string[len-1] = 0;
      if (*cur_p == ':')
	{
	  ++cur_p;
	  return HEADER;
	}
      return IDENT;
    }

  if (*cur_p == '/')
    {
      char *p = ++cur_p;
      int len;

      while (*cur_p && *cur_p != '/')
	cur_p++;
      len = cur_p - p + 1;
      cur_p++;
      yylval.string = xmalloc (len);
      memcpy (yylval.string, p, len-1);
      yylval.string[len-1] = 0;
      return REGEXP;
    }

  if (*cur_p == ':')
    {
      cur_p++;
      if (*cur_p == '/')
	{
	  char *p = ++cur_p;
	  int len;

	  while (*cur_p && *cur_p != '/')
	    cur_p++;
	  len = cur_p - p + 1;
	  cur_p++;
	  yylval.string = xmalloc (len);
	  memcpy (yylval.string, p, len-1);
	  yylval.string[len-1] = 0;
	  return BODY;
	}
      yylval.type = *cur_p++;
      return TYPE;
    }

  return *cur_p++;
}

int
msgset_parse (const int argc, char **argv, int flags, msgset_t **mset)
{
  int rc;
  xargc = argc;
  xargv = argv;
  msgset_flags = flags;
  cur_ind = 1;
  cur_p = NULL;
  result = NULL;
  rc = yyparse ();
  if (rc == 0)
    {
      if (result == NULL)
	{
	  util_noapp ();
	  rc = 1;
	}
      else
	*mset = result;
    }
  return rc;
}

void
msgset_free (msgset_t *msg_set)
{
  msgset_t *next;

  if (!msg_set)
    return;
  while (msg_set)
    {
      next = msg_set->next;
      if (msg_set->msg_part)
	free (msg_set->msg_part);
      free (msg_set);
      msg_set = next;
    }
}

size_t
msgset_count (msgset_t *set)
{
  size_t count = 0;
  for (; set; set = set->next)
    count++;
  return count;
}

/* Create a message set consisting of a single msg_num and no subparts */
msgset_t *
msgset_make_1 (size_t number)
{
  msgset_t *mp;

  if (number == 0)
    return NULL;
  mp = xmalloc (sizeof (*mp));
  mp->next = NULL;
  mp->npart = 1;
  mp->msg_part = xmalloc (sizeof mp->msg_part[0]);
  mp->msg_part[0] = number;
  return mp;
}

msgset_t *
msgset_dup (const msgset_t *set)
{
  msgset_t *mp;
  mp = xmalloc (sizeof (*mp));
  mp->next = NULL;
  mp->npart = set->npart;
  mp->msg_part = xcalloc (mp->npart, sizeof mp->msg_part[0]);
  memcpy (mp->msg_part, set->msg_part, mp->npart * sizeof mp->msg_part[0]);
  return mp;
}

msgset_t *
msgset_append (msgset_t *one, msgset_t *two)
{
  msgset_t *last;

  if (!one)
    return two;
  for (last = one; last->next; last = last->next)
    ;
  last->next = two;
  return one;
}

int
msgset_member (msgset_t *set, size_t n)
{
  for (; set; set = set->next)
    if (set->msg_part[0] == n)
      return 1;
  return 0;
}

msgset_t *
msgset_negate (msgset_t *set)
{
  size_t i;
  msgset_t *first = NULL, *last = NULL;

  for (i = 1; i <= total; i++)
    {
      if (!msgset_member (set, i))
	{
	  msgset_t *mp = msgset_make_1 (i);
	  if (!first)
	    first = mp;
	  else
	    last->next = mp;
	  last = mp;
	}
    }
  return first;
}

msgset_t *
msgset_range (int low, int high)
{
  int i;
  msgset_t *mp, *first = NULL, *last = NULL;

  if (low == high)
    return msgset_make_1 (low);

  if (low >= high)
    {
      yyerror (_("range error"));
      return NULL;
    }

  for (i = 0; low <= high; i++, low++)
    {
      mp = msgset_make_1 (low);
      if (!first)
	first = mp;
      else
	last->next = mp;
      last = mp;
    }
  return first;
}

msgset_t *
msgset_expand (msgset_t *set, msgset_t *expand_by)
{
  msgset_t *i, *j;
  msgset_t *first = NULL, *last = NULL, *mp;

  for (i = set; i; i = i->next)
    for (j = expand_by; j; j = j->next)
      {
	mp = xmalloc (sizeof *mp);
	mp->next = NULL;
	mp->npart = i->npart + j->npart;
	mp->msg_part = xcalloc (mp->npart, sizeof mp->msg_part[0]);
	memcpy (mp->msg_part, i->msg_part, i->npart * sizeof i->msg_part[0]);
	memcpy (mp->msg_part + i->npart, j->msg_part,
		j->npart * sizeof j->msg_part[0]);

	if (!first)
	  first = mp;
	else
	  last->next = mp;
	last = mp;
      }
  return first;
}

msgset_t *
msgset_select (int (*sel) (mu_message_t, void *), void *closure, int rev,
	       unsigned int max_matches)
{
  size_t i, match_count = 0;
  msgset_t *first = NULL, *last = NULL, *mp;
  mu_message_t msg = NULL;

  if (max_matches == 0)
    max_matches = total;

  if (rev)
    {
      for (i = total; i > 0; i--)
	{
	  mu_mailbox_get_message (mbox, i, &msg);
	  if ((*sel)(msg, closure))
	    {
	      mp = msgset_make_1 (i);
	      if (!first)
		first = mp;
	      else
		last->next = mp;
	      last = mp;
	      if (++match_count == max_matches)
		break;
	    }
	}
    }
  else
    {
      for (i = 1; i <= total; i++)
	{
	  mu_mailbox_get_message (mbox, i, &msg);
	  if ((*sel)(msg, closure))
	    {
	      mp = msgset_make_1 (i);
	      if (!first)
		first = mp;
	      else
		last->next = mp;
	      last = mp;
	      if (++match_count == max_matches)
		break;
	    }
	}
    }
  return first;
}

int
select_header (mu_message_t msg, void *closure)
{
  struct header_data *hd = (struct header_data *)closure;
  mu_header_t hdr;
  char *contents;
  const char *header = hd->header ? hd->header : MU_HEADER_SUBJECT;

  mu_message_get_header (msg, &hdr);
  if (mu_header_aget_value (hdr, header, &contents) == 0)
    {
      if (mailvar_get (NULL, "regex", mailvar_type_boolean, 0) == 0)
	{
	  /* Match string against the extended regular expression(ignoring
	     case) in pattern, treating errors as no match.
	     Return 1 for match, 0 for no match.
	  */
          regex_t re;
          int status;
	  int flags = REG_EXTENDED;

	  if (mu_islower (header[0]))
	    flags |= REG_ICASE;
          if (regcomp (&re, hd->expr, flags) != 0)
	    {
	      free (contents);
	      return 0;
	    }
          status = regexec (&re, contents, 0, NULL, 0);
          free (contents);
	  regfree (&re);
          return status == 0;
	}
      else
	{
	  int rc;
	  mu_strupper (contents);
	  rc = strstr (contents, hd->expr) != NULL;
	  free (contents);
	  return rc;
	}
    }
  return 0;
}

int
select_body (mu_message_t msg, void *closure)
{
  char *expr = closure;
  int noregex = mailvar_get (NULL, "regex", mailvar_type_boolean, 0);
  regex_t re;
  int status;
  mu_body_t body = NULL;
  mu_stream_t stream = NULL;
  size_t size = 0, lines = 0;
  char buffer[128];
  size_t n = 0;
  off_t offset = 0;

  if (noregex)
    mu_strupper (expr);
  else if (regcomp (&re, expr, REG_EXTENDED | REG_ICASE) != 0)
    return 0;

  mu_message_get_body (msg, &body);
  mu_body_size (body, &size);
  mu_body_lines (body, &lines);
  mu_body_get_stream (body, &stream);
  status = 0;
  while (status == 0
	 && mu_stream_readline (stream, buffer, sizeof(buffer)-1, offset, &n) == 0
	 && n > 0)
    {
      offset += n;
      if (noregex)
	{
	  mu_strupper (buffer);
	  status = strstr (buffer, expr) != NULL;
	}
      else
	status = regexec (&re, buffer, 0, NULL, 0);
    }

  if (!noregex)
    regfree (&re);

  return status;
}

int
select_sender (mu_message_t msg MU_ARG_UNUSED, void *closure MU_ARG_UNUSED)
{
  /* char *sender = (char*) closure; */
  /* FIXME: all messages from sender argv[i] */
  /* Annoying we can use mu_address_create() for that
     but to compare against what? The email ?  */
  return 0;
}

int
select_type (mu_message_t msg, void *closure)
{
  int type = *(int*) closure;
  mu_attribute_t attr= NULL;

  mu_message_get_attribute (msg, &attr);

  switch (type)
    {
    case 'd':
      return mu_attribute_is_deleted (attr);
    case 'n':
      return mu_attribute_is_recent (attr);
    case 'o':
      return mu_attribute_is_seen (attr);
    case 'r':
      return mu_attribute_is_read (attr);
    case 'u':
      return !mu_attribute_is_read (attr);
    case 't':
      return mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_TAGGED);
    case 'T':
      return !mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_TAGGED);
    }
  return 0;
}

int
select_deleted (mu_message_t msg, void *closure MU_ARG_UNUSED)
{
  mu_attribute_t attr= NULL;
  int rc;

  mu_message_get_attribute (msg, &attr);
  rc = mu_attribute_is_deleted (attr);
  return strcmp (xargv[0], "undelete") == 0 ? rc : !rc;
}

int
check_set (msgset_t **pset)
{
  int flags = msgset_flags;
  int rc = 0;
  
  if (msgset_count (*pset) == 1)
    flags ^= MSG_SILENT;
  if (flags & MSG_NODELETED)
    {
      msgset_t *p = *pset, *prev = NULL;
      msgset_t *delset = NULL;

      while (p)
	{
	  msgset_t *next = p->next;
	  if (util_isdeleted (p->msg_part[0]))
	    {
	      if ((flags & MSG_SILENT) && (prev || next))
		{
		  /* Mark subset as deleted */
		  p->next = delset;
		  delset = p;
		  /* Remove it from the set */
		  if (prev)
		    prev->next = next;
		  else
		    *pset = next;
		}
	      else
		{
		  util_error (_("%lu: Inappropriate message (has been deleted)"),
			      (unsigned long) p->msg_part[0]);
		  /* Delete entire set */
		  delset = *pset;
		  *pset = NULL;
		  rc = 1;
		  break;
		}
	    }
	  else
	    prev = p;
	  p = next;
	}

      if (delset)
	msgset_free (delset);

      if (!*pset)
	rc = 1;
    }

  return rc;
}

#if 0
void
msgset_print (msgset_t *mset)
{
  int i;
  printf ("(");
  printf ("%d .", mset->msg_part[0]);
  for (i = 1; i < mset->npart; i++)
    {
      printf (" %d", mset->msg_part[i]);
    }
  printf (")\n");
}

int
main(int argc, char **argv)
{
  msgset_t *mset = NULL;
  int rc = msgset_parse (argc, argv, &mset);

  for (; mset; mset = mset->next)
    msgset_print (mset);
  return 0;
}
#endif
