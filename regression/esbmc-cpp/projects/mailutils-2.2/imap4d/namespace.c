/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2005, 2007, 2008, 2009, 2010 Free Software
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

#include "imap4d.h"

typedef int (*nsfp_t) (void *closure, int ns, char *path, int delim);

mu_list_t namespace[NS_MAX];

static const char *
printable_pathname (const char *str)
{
  if (strncmp (str, imap4d_homedir, strlen (imap4d_homedir)) == 0)
    {
      str += strlen (imap4d_homedir);
      if (str[0] == '/')
	str++;
    }
  return str;
}

static int
print_namespace_fun (void *item, void *data)
{
  int *pcount = data;
  const char *dir = printable_pathname (item);
  char *suf = (dir[0] && dir[strlen (dir) - 1] != '/') ? "/" : "";
  if ((*pcount)++)
    util_send (" ");
  util_send ("(\"%s%s\" \"/\")", dir, suf);
  return 0;
}

static void
print_namespace (int nsid)
{
  mu_list_t list = namespace[nsid];
  if (!list)
    util_send ("NIL");
  else
    {
      int count;
      count = 0;
      util_send ("(");
      mu_list_do (list, print_namespace_fun, &count);
      util_send (")");
    }
}

struct ns_closure
{
  int id;
  nsfp_t fun;
  void *closure;
};

static int
_enum_fun (void *item, void *data)
{
  struct ns_closure *nsp = data;
  return nsp->fun (nsp->closure, nsp->id, (char*) item, '/');
}
  
static int
namespace_enumerate (int id, nsfp_t f, void *closure)
{
  struct ns_closure nsc;

  nsc.id = id;
  nsc.fun = f;
  nsc.closure = closure;
  return namespace[id] == 0 ? 0 :
          mu_list_do (namespace[id], _enum_fun, &nsc);
}

static int
namespace_enumerate_all (nsfp_t f, void *closure)
{
  return namespace_enumerate (NS_PRIVATE, f, closure)
    || namespace_enumerate (NS_OTHER, f, closure)
    || namespace_enumerate (NS_SHARED, f, closure);
}

/*
5. NAMESPACE Command

   Arguments: none

   Response:  an untagged NAMESPACE response that contains the prefix
                 and hierarchy delimiter to the server's Personal
                 Namespace(s), Other Users' Namespace(s), and Shared
                 Namespace(s) that the server wishes to expose. The
                 response will contain a NIL for any namespace class
                 that is not available. Namespace_Response_Extensions
                 MAY be included in the response.
                 Namespace_Response_Extensions which are not on the IETF
                 standards track, MUST be prefixed with an "X-".
*/

int
imap4d_namespace (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  if (imap4d_tokbuf_argc (tok) != 2)
    return util_finish (command, RESP_BAD, "Invalid arguments");

  util_send ("* NAMESPACE ");

  print_namespace (NS_PRIVATE);
  util_send (" ");
  print_namespace (NS_OTHER);
  util_send (" ");
  print_namespace (NS_SHARED);
  util_send ("\r\n");

  return util_finish (command, RESP_OK, "Completed");
}


struct namespace_info
{
  char *name;
  int namelen;
  int ns;
  int exact;
};

static int
check_namespace (void *closure, int ns, char *path, int delim)
{
  struct namespace_info *p = (struct namespace_info *) closure;
  int len = strlen (path);
  if ((len == 0 && p->namelen == len)
      || (len > 0 && strncmp (path, p->name, strlen (path)) == 0))
    {
      p->ns = ns;
      p->exact = len == p->namelen;
      return 1;
    }
  return 0;
}

static int
risky_pattern (const char *pattern, int delim)
{
  for (; *pattern && *pattern != delim; pattern++)
    {
      if (*pattern == '%' || *pattern == '*')
	return 1;
    }
  return 0;
}

char *
namespace_checkfullpath (const char *name, const char *pattern, 
			 const char *delim, int *nspace)
{
  struct namespace_info info;
  const char *p;
  char *path = NULL;
  char *scheme = NULL;

  p = strchr (name, ':');
  if (p)
    {
      size_t size = p - name + 1;
      scheme = malloc (size + 1);
      if (!scheme)
	return NULL;
      memcpy (scheme, name, size);
      scheme[size] = 0;
      name = p + 1;
    }

  path = util_getfullpath (name, delim);
  if (!path)
    {
      free (scheme);
      return path;
    }
  info.name = path;
  info.namelen = strlen (path);
  if (!namespace_enumerate_all (check_namespace, &info))
    {
      free (scheme);
      free (path);
      return NULL;
    }

  if (pattern &&
      info.ns == NS_OTHER && info.exact && risky_pattern (pattern, '/'))
    {
      free (scheme);
      free (path);
      return NULL;
    }

  if (nspace)
    *nspace = info.ns;
  if (scheme)
    {
      char *pathstr = malloc (strlen (scheme) + strlen (path) + 2);
      if (pathstr)
	{
	  strcpy (pathstr, scheme);
	  strcat (pathstr, path);
	}
      free (scheme);
      free (path);
      path = pathstr;
    }
  return path;
}

char *
namespace_getfullpath (const char *name, const char *delim, int *nspace)
{
  char *ret;
  if (mu_c_strcasecmp (name, "INBOX") == 0 && auth_data->change_uid)
    {
      ret = strdup (auth_data->mailbox);
      if (nspace)
	*nspace = NS_PRIVATE;
    }
  else
    ret = namespace_checkfullpath (name, NULL, delim, nspace);
  return ret;
}

int
namespace_init_session (char *path)
{
  if (!namespace[NS_PRIVATE])
    mu_list_create (&namespace[NS_PRIVATE]);
  mu_list_prepend (namespace[NS_PRIVATE],
		   mu_strdup (mu_normalize_path (path)));
  return 0;
}

static int
normalize_fun (void *item, void *data)
{
  char *name = item;
  mu_list_t list = data;
  return mu_list_append (list,
			 mu_strdup (mu_normalize_path (name)));
}

void
namespace_init ()
{
  int i;

  for (i = 0; i < NS_MAX; i++)
    {
      if (namespace[i])
	{
	  mu_list_t list;
	  mu_list_create (&list);
	  mu_list_set_destroy_item (list, mu_list_free_item);
	  mu_list_do (namespace[i], normalize_fun, list);
	  mu_list_set_destroy_item (namespace[i], mu_list_free_item);
	  mu_list_destroy (&namespace[i]);
	  namespace[i] = list;
	}
    }
}
