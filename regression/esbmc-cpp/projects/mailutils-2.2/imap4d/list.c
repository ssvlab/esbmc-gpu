/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2005, 2006, 2007, 2008, 2009, 2010
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

#include "imap4d.h"
#include <dirent.h>
#include <pwd.h>

static int
imap4d_match (const char *name, void *pat, int flags)
{
  return util_wcard_match (name, pat, "/");
}

struct refinfo
{
  char *refptr;   /* Original reference */
  size_t reflen;  /* Length of the original reference */
  size_t pfxlen;  /* Length of the current prefix */
  size_t homelen; /* Length of homedir */
  char *buf;
  size_t bufsize;
};

static int
list_fun (mu_folder_t folder, struct mu_list_response *resp, void *data)
{
  char *name;
  struct refinfo *refinfo = data;
  size_t size;
  
  name = resp->name;
  size = strlen (name);
  if (size == refinfo->homelen + 6
      && memcmp (name, imap4d_homedir, refinfo->homelen) == 0
      && memcmp (name + refinfo->homelen + 1, "INBOX", 5) == 0)
    return 0;
     
  util_send ("* %s", "LIST (");
  if ((resp->type & (MU_FOLDER_ATTRIBUTE_FILE|MU_FOLDER_ATTRIBUTE_DIRECTORY))
       == (MU_FOLDER_ATTRIBUTE_FILE|MU_FOLDER_ATTRIBUTE_DIRECTORY))
    /* nothing */;
  else if (resp->type & MU_FOLDER_ATTRIBUTE_FILE)
    util_send ("\\NoInferiors");
  else if (resp->type & MU_FOLDER_ATTRIBUTE_DIRECTORY)
    util_send ("\\NoSelect");
  
  util_send (") \"%c\" ", resp->separator);

  name = resp->name + refinfo->pfxlen;
  size = strlen (name) + refinfo->reflen + 1;
  if (size > refinfo->bufsize)
    {
      if (refinfo->buf == NULL)
	{
	  refinfo->bufsize = size;
	  refinfo->buf = malloc (refinfo->bufsize);
	  if (!refinfo->buf)
	    {
	      mu_error ("%s", mu_strerror (errno));
	      return 1;
	    }
	  memcpy (refinfo->buf, refinfo->refptr, refinfo->reflen);
	}
      else
	{
	  char *p = realloc (refinfo->buf, size);
	  if (!p)
	    {
	      mu_error ("%s", mu_strerror (errno));
	      return 1;
	    }
	  refinfo->buf = p;
	  refinfo->bufsize = size;
	}
    }

  if ((refinfo->reflen == 0 || refinfo->refptr[refinfo->reflen - 1] == '/')
      && name[0] == '/')
    name++;
  strcpy (refinfo->buf + refinfo->reflen, name);
  name = refinfo->buf;
  
  if (strpbrk (name, "\"{}"))
    util_send ("{%lu}\r\n%s\r\n", (unsigned long) strlen (name), name);
  else if (is_atom (name))
    util_send ("%s\r\n", name);
  else
    util_send ("\"%s\"\r\n", name);
  return 0;
}

/*
6.3.8.  LIST Command

   Arguments:  reference name
               mailbox name with possible wildcards

   Responses:  untagged responses: LIST

   Result:     OK - list completed
               NO - list failure: can't list that reference or name
               BAD - command unknown or arguments invalid
*/

/*
  1- IMAP4 insists: the reference argument present in the
  interpreted form SHOULD prefix the interpreted form.  It SHOULD
  also be in the same form as the reference name argument.  This
  rule permits the client to determine if the returned mailbox name
  is in the context of the reference argument, or if something about
  the mailbox argument overrode the reference argument.
  
  ex:
  Reference         Mailbox         -->  Interpretation
  ~smith/Mail        foo.*          -->  ~smith/Mail/foo.*
  archive            %              --> archive/%
  #news              comp.mail.*     --> #news.comp.mail.*
  ~smith/Mail        /usr/doc/foo   --> /usr/doc/foo
  archive            ~fred/Mail     --> ~fred/Mail/ *

  2- The character "*" is a wildcard, and matches zero or more characters
  at this position.  The character "%" is similar to "*",
  but it does not match the hierarchy delimiter.  */

int
imap4d_list (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  char *ref;
  char *wcard;
  const char *delim = "/";

  if (imap4d_tokbuf_argc (tok) != 4)
    return util_finish (command, RESP_BAD, "Invalid arguments");
  
  ref = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);
  wcard = imap4d_tokbuf_getarg (tok, IMAP4_ARG_2);

  /* If wildcard is empty, it is a special case: we have to
     return the hierarchy.  */
  if (*wcard == '\0')
    {
      util_out (RESP_NONE, "LIST (\\NoSelect) \"%s\" \"%s\"", delim,
		(*ref) ? delim : "");
    }
  /* There is only one mailbox in the "INBOX" hierarchy ... INBOX.  */
  else if (mu_c_strcasecmp (ref, "INBOX") == 0
	   || (ref[0] == 0 && mu_c_strcasecmp (wcard, "INBOX") == 0))
    {
      util_out (RESP_NONE, "LIST (\\NoInferiors) NIL INBOX");
    }
  else
    {
      int status;
      mu_folder_t folder;
      char *cwd;
      char *p, *q;
      struct refinfo refinfo;
      
      switch (*wcard)
	{
	  /* Absolute Path in wcard, dump the old ref.  */
	case '/':
	  {
	    ref = calloc (2, 1);
	    ref[0] = *wcard;
	    wcard++;
	  }
	  break;

	  /* Absolute Path, but take care of things like ~guest/Mail,
	     ref becomes ref = ~guest.  */
	case '~':
	  {
	    char *s = strchr (wcard, '/');
	    if (s)
	      {
		ref = calloc (s - wcard + 1, 1);
		memcpy (ref, wcard, s - wcard);
		ref [s - wcard] = '\0';
		wcard = s + 1;
	      }
	    else
	      {
		ref = strdup (wcard);
		wcard += strlen (wcard);
	      }
	  }
	  break;

	default:
	  ref = strdup (ref);
	}

      /* Move any directory not containing a wildcard into the reference
	 So (ref = ~guest, wcard = Mail/folder1/%.vf) -->
	 (ref = ~guest/Mail/folder1, wcard = %.vf).  */
      for (p = wcard; (q = strpbrk (p, "/%*")) && *q == '/'; p = q + 1)
	;

      if (p > wcard)
	{
	  size_t seglen = p - wcard;
	  size_t reflen = strlen (ref);
	  int addslash = (reflen > 0 && ref[reflen-1] != '/'); 
	  size_t len = seglen + reflen + addslash + 1;

	  ref = realloc (ref, len);
	  if (addslash)
	    ref[reflen++] = '/';
	  memcpy (ref + reflen, wcard, seglen);
	  ref[reflen + seglen] = 0;
	  wcard += seglen;
	}

      /* Allocates.  */
      cwd = namespace_checkfullpath (ref, wcard, delim, NULL);
      if (!cwd)
	{
	  free (ref);
	  return util_finish (command, RESP_NO,
			      "The requested item could not be found.");
	}
      status = mu_folder_create (&folder, cwd);
      if (status)
	{
	  free (ref);
	  free (cwd);
	  return util_finish (command, RESP_NO,
			      "The requested item could not be found.");
	}
      mu_folder_set_match (folder, imap4d_match);

      memset (&refinfo, 0, sizeof refinfo);

      refinfo.refptr = ref;
      refinfo.reflen = strlen (ref);
      refinfo.pfxlen = strlen (cwd);
      refinfo.homelen = strlen (imap4d_homedir);

      /* The special name INBOX is included in the output from LIST, if
	 INBOX is supported by this server for this user and if the
	 uppercase string "INBOX" matches the interpreted reference and
	 mailbox name arguments with wildcards as described above.  The
	 criteria for omitting INBOX is whether SELECT INBOX will return
	 failure; it is not relevant whether the user's real INBOX resides
	 on this or some other server. */

      if (!*ref && (imap4d_match ("INBOX", wcard, 0) == 0
		    || imap4d_match ("inbox", wcard, 0) == 0))
	util_out (RESP_NONE, "LIST (\\NoInferiors) NIL INBOX");

      mu_folder_enumerate (folder, NULL, wcard, 0, 0, NULL,
			   list_fun, &refinfo);
      mu_folder_destroy (&folder);
      free (refinfo.buf);
      free (cwd);
      free (ref);
    }

  return util_finish (command, RESP_OK, "Completed");
}

