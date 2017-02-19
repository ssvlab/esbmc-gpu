%{
/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2004, 2005, 2006, 2007, 2010 Free Software
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
#include <pwd.h>
#include <grp.h>
#include <sys/types.h>
  
struct mh_alias
{
  char *name;
  mu_list_t rcpt_list;
  int inclusive;
};

static mu_list_t alias_list;

static mu_list_t
list_create_or_die ()
{
  int status;
  mu_list_t list;

  status = mu_list_create (&list);
  if (status)
    {
      ali_parse_error (_("can't create list: %s"), mu_strerror (status));
      exit (1);
    }
  return list;
}

static char *
ali_list_to_string (mu_list_t *plist)
{
  size_t n;
  char *string;
  
  mu_list_count (*plist, &n);
  if (n == 1)
    {
      mu_list_get (*plist, 0, (void **)&string);
    }
  else
    {
      char *p;
      size_t length = 0;
      mu_iterator_t itr;
      mu_list_get_iterator (*plist, &itr);
      for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next(itr))
	{
	  char *s;
	  mu_iterator_current (itr, (void**) &s);
	  length += strlen (s) + 1;
	}
  
      string = xmalloc (length + 1);
      p = string;
      for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next(itr))
	{
	  char *s;
	  mu_iterator_current (itr, (void**) &s);
	  strcpy (p, s);
	  p += strlen (s);
	  *p++ = ' ';
	}
      *--p = 0;
      mu_iterator_destroy (&itr);
    }
  mu_list_destroy (plist);
  return string;
}

static mu_list_t unix_group_to_list (char *name);
static mu_list_t unix_gid_to_list (char *name);
static mu_list_t unix_passwd_to_list (void);

int yyerror (char *s);
int yylex (void);

%}

%union {
  char *string;
  mu_list_t list;
  struct mh_alias *alias;
}

%token <string> STRING
%type <list>  address_list address_group string_list
%type <string> address
%type <alias> alias

%%

input        : /* empty */
             | alias_list
             | alias_list nl
             | nl alias_list
             | nl alias_list nl
             ;

alias_list   : alias
               {
		 if (!alias_list)
		   alias_list = list_create_or_die ();
		 mu_list_append (alias_list, $1);
	       }
             | alias_list nl alias
               {
		 mu_list_append (alias_list, $3);
	       }
             ;

nl           : '\n'
             | nl '\n'
             ;

alias        : STRING ':' { ali_verbatim (1); } address_group
               {
		 ali_verbatim (0);
		 $$ = xmalloc (sizeof (*$$));
		 $$->name = $1;
		 $$->rcpt_list = $4;
		 $$->inclusive = 0;
	       }
             | STRING ';' { ali_verbatim (1); } address_group
               {
		 ali_verbatim (0);
		 $$ = xmalloc (sizeof (*$$));
		 $$->name = $1;
		 $$->rcpt_list = $4;
		 $$->inclusive = 1;
	       }
             ;

address_group: address_list
             | '=' STRING
               {
		 $$ = unix_group_to_list ($2);
		 free ($2);
	       }
             | '+' STRING
               {
		 $$ = unix_gid_to_list ($2);
		 free ($2);
	       }
             | '*'
               {
		 $$ = unix_passwd_to_list ();
	       }
             ;

address_list : address
               {
		 $$ = list_create_or_die ();
		 mu_list_append ($$, $1);
	       }
             | address_list ',' address
               {
		 mu_list_append ($1, $3);
		 $$ = $1;
	       }
             ;

address      : string_list
               {
		 $$ = ali_list_to_string (&$1);
	       }
             ;

string_list  : STRING
               {
		 mu_list_create(&$$);
		 mu_list_append($$, $1);
	       }
             | string_list STRING
               {
		 mu_list_append($1, $2);
		 $$ = $1;
	       }
             ;

%%

static mu_list_t
ali_list_dup (mu_list_t src)
{
  mu_list_t dst;
  mu_iterator_t itr;

  if (mu_list_create (&dst))
    return NULL;

  if (mu_list_get_iterator (src, &itr))
    {
      mu_list_destroy (&dst);
      return NULL;
    }
  
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
    {
      void *data;
      mu_iterator_current (itr, (void **)&data);
      mu_list_append (dst, data);
    }
  mu_iterator_destroy (&itr);
  return dst;
}

static int
ali_member (mu_list_t list, const char *name)
{
  mu_iterator_t itr;
  int found = 0;

  if (mu_list_get_iterator (list, &itr))
    return 0;
  for (mu_iterator_first (itr); !found && !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      char *item;
      mu_address_t tmp;
      
      mu_iterator_current (itr, (void **)&item);
      if (strcmp (item, name) == 0)
	found = 1;
      else if (mu_address_create (&tmp, item) == 0)
	{
	  found = mu_address_contains_email (tmp, name);
	  mu_address_destroy (&tmp);
	}
    }
  mu_iterator_destroy (&itr);
  return found;
}

int
aliascmp (const char *pattern, const char *name)
{
  int len = strlen (pattern);

  if (len > 1 && pattern[len - 1] == '*')
    return strncmp (pattern, name, len - 2);
  return strcmp (pattern, name);
}

static int mh_alias_get_internal (const char *name, mu_iterator_t start,
				  mu_list_t *return_list, int *inclusive);

int
alias_expand_list (mu_list_t name_list, mu_iterator_t orig_itr, int *inclusive)
{
  mu_iterator_t itr;

  if (mu_list_get_iterator (name_list, &itr))
    return 1;
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
    {
      char *name;
      mu_list_t exlist;
      
      mu_iterator_current (itr, (void **)&name);
      if (mh_alias_get_internal (name, orig_itr, &exlist, inclusive) == 0)
	{
	  /* Insert exlist after name */
	  mu_iterator_ctl (itr, mu_itrctl_insert_list, exlist);
	  mu_list_destroy (&exlist);
	  /* Remove name */
	  mu_iterator_ctl (itr, mu_itrctl_delete, NULL);
	}
    }
  mu_iterator_destroy (&itr);
  return 0;
}  

/* Look up the named alias. If found, return the list of recipient
   names associated with it */
static int
mh_alias_get_internal (const char *name,
		       mu_iterator_t start, mu_list_t *return_list,
		       int *inclusive) 
{
  mu_iterator_t itr;
  int rc = 1;

  if (!start)
    {
      if (mu_list_get_iterator (alias_list, &itr))
	return 1;
      mu_iterator_first (itr);
    }
  else
    {
      mu_iterator_dup (&itr, start);
      mu_iterator_next (itr);
    }
	
  for (; !mu_iterator_is_done (itr); mu_iterator_next (itr))
    {
      struct mh_alias *alias;
      mu_iterator_current (itr, (void **)&alias);
      if (aliascmp (alias->name, name) == 0)
	{
	  if (inclusive)
	    *inclusive |= alias->inclusive;
	  *return_list = ali_list_dup (alias->rcpt_list);
	  alias_expand_list (*return_list, itr, inclusive);
	  rc = 0;
	  break;
	}
    }
  
  mu_iterator_destroy (&itr);
  return rc;
}

int
mh_alias_get (const char *name, mu_list_t *return_list)
{
  return mh_alias_get_internal (name, NULL, return_list, NULL);
}

int
mh_alias_get_address (const char *name, mu_address_t *paddr, int *incl)
{
  mu_iterator_t itr;
  mu_list_t list;

  if (incl)
    *incl = 0;
  if (mh_alias_get_internal (name, NULL, &list, incl))
    return 1;
  if (mu_list_is_empty (list))
    {
      mu_list_destroy (&list);
      return 1;
    }
  
  if (mu_list_get_iterator (list, &itr) == 0)
    {
      for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
	{
	  char *item;
	  mu_address_t a;
	  char *ptr = NULL; 

	  mu_iterator_current (itr, (void **)&item);
	  if (mu_address_create (&a, item))
	    {
	      mu_error (_("Error expanding aliases -- invalid address `%s'"),
			item);
	    }
	  else
	    {
	      if (incl && *incl)
		mu_address_set_personal (a, 1, name);
	      mu_address_union (paddr, a);
	      mu_address_destroy (&a);
	    }
	  if (ptr)
	    free (ptr);
	}
      mu_iterator_destroy (&itr);
    }
  mu_list_destroy (&list);
  return 0;
}

/* Look up the given user name in the aliases. Return the list of
   alias names this user is member of */
int
mh_alias_get_alias (const char *uname, mu_list_t *return_list)
{
  mu_iterator_t itr;
  int rc = 1;
  
  if (mu_list_get_iterator (alias_list, &itr))
    return 1;
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
    {
      struct mh_alias *alias;
      mu_iterator_current (itr, (void **)&alias);
      if (ali_member (alias->rcpt_list, uname))
	{
	  if (*return_list == NULL && mu_list_create (return_list))
	    break;
	  mu_list_append (*return_list, alias->name);
	  rc = 0;
	}
    }
  
  mu_iterator_destroy (&itr);
  return rc;
}

void
mh_alias_enumerate (mh_alias_enumerator_t fun, void *data)
{
  mu_iterator_t itr;
  int rc = 0;
  
  if (mu_list_get_iterator (alias_list, &itr))
    return ;
  for (mu_iterator_first (itr);
       rc == 0 && !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      struct mh_alias *alias;
      mu_list_t tmp;
      
      mu_iterator_current (itr, (void **)&alias);

      tmp = ali_list_dup (alias->rcpt_list);
      alias_expand_list (tmp, itr, NULL);

      rc = fun (alias->name, tmp, data);
      mu_list_destroy (&tmp);
    }
  mu_iterator_destroy (&itr);
}

static mu_list_t
unix_group_to_list (char *name)
{
  struct group *grp = getgrnam (name);
  mu_list_t lst = list_create_or_die ();
  
  if (grp)
    {
      char **p;

      for (p = grp->gr_mem; *p; p++)
	mu_list_append (lst, strdup (*p));
    }      
  
  return lst;
}

static mu_list_t
unix_gid_to_list (char *name)
{
  struct group *grp = getgrnam (name);
  mu_list_t lst = list_create_or_die ();

  if (grp)
    {
      struct passwd *pw;
      setpwent();
      while ((pw = getpwent ()))
	{
	  if (pw->pw_gid == grp->gr_gid)
	    mu_list_append (lst, strdup (pw->pw_name));
	}
      endpwent();
    }
  return lst;
}

static mu_list_t
unix_passwd_to_list ()
{
  mu_list_t lst = list_create_or_die ();
  struct passwd *pw;

  setpwent();
  while ((pw = getpwent ()))
    {
      if (pw->pw_uid > 200)
	mu_list_append (lst, strdup (pw->pw_name));
    }
  endpwent();
  return lst;
}

int
mh_read_aliases ()
{
  const char *p;
  
  p = mh_global_profile_get ("Aliasfile", NULL);
  if (p)
    {
      int argc;
      char **argv;
      int rc = mu_argcv_get (p, NULL, NULL, &argc, &argv);
      if (rc == 0)
	{
	  int i;
	  for (i = 0; i < argc; i++) 
	    mh_alias_read (argv[i], 1);
	}
      mu_argcv_free (argc, argv);
    }
  mh_alias_read (DEFAULT_ALIAS_FILE, 0);
  return 0;
}
