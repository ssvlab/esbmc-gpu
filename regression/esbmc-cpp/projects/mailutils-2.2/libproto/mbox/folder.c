/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2004, 2005, 2006, 2007, 2008,
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

#include <errno.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <glob.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>

#include <folder0.h>
#include <registrar0.h>

#include <mailutils/auth.h>
#include <mailutils/url.h>
#include <mailutils/stream.h>
#include <mailutils/mutil.h>
#include <mailutils/errno.h>
#include <mailutils/debug.h>

/* We export url parsing and the initialisation of
   the mailbox, via the register entry/record.  */

static int
_mbox_is_scheme (mu_record_t record, mu_url_t url, int flags)
{
  int rc = 0;
  
  if (mu_url_is_scheme (url, record->scheme))
    return MU_FOLDER_ATTRIBUTE_FILE & flags;
  
  if (mu_scheme_autodetect_p (url))
    {
      struct stat st;
      const char *path;

      mu_url_sget_path (url, &path);
      if (stat (path, &st) < 0)
	return 0;

      if (S_ISREG (st.st_mode) || S_ISCHR (st.st_mode))
	{
	  if (st.st_size == 0)
	    {
	      rc |= MU_FOLDER_ATTRIBUTE_FILE;
	    }
	  else if (flags & MU_FOLDER_ATTRIBUTE_FILE)
	    {
#if 0
	      /* This effectively sieves out all non-mailbox files,
		 but it makes mu_folder_enumerate crawl, which is
		 intolerable for imap4d LIST command. */
	      int fd = open (path, O_RDONLY);
	      if (fd != -1)
		{
		  char buf[5];
		  if (read (fd, buf, 5) == 5)
		    if (memcmp (buf, "From ", 5) == 0)
		      rc |= MU_FOLDER_ATTRIBUTE_FILE;
		  close (fd);
		}
#else
	      rc |= MU_FOLDER_ATTRIBUTE_FILE;
#endif
	    }
	}
	  
      if ((flags & MU_FOLDER_ATTRIBUTE_DIRECTORY)
	  && S_ISDIR (st.st_mode))
	rc |= MU_FOLDER_ATTRIBUTE_DIRECTORY;
    }
  return rc;
}

static struct _mu_record _mbox_record =
{
  MU_MBOX_PRIO,
  MU_MBOX_SCHEME,
  mu_url_expand_path, /* URL init.  */
  _mailbox_mbox_init, /* Mailbox init.  */
  NULL, /* Mailer init.  */
  _folder_mbox_init, /* Folder init.  */
  NULL, /* No need for an back pointer.  */
  _mbox_is_scheme, /* _is_scheme method.  */
  NULL, /* _get_url method.  */
  NULL, /* _get_mailbox method.  */
  NULL, /* _get_mailer method.  */
  NULL  /* _get_folder method.  */
};
mu_record_t mu_mbox_record = &_mbox_record;

/* lsub/subscribe/unsubscribe are not needed.  */
static void folder_mbox_destroy    (mu_folder_t);
static int folder_mbox_open        (mu_folder_t, int);
static int folder_mbox_close       (mu_folder_t);
static int folder_mbox_delete      (mu_folder_t, const char *);
static int folder_mbox_rename      (mu_folder_t , const char *, const char *);
static int folder_mbox_list        (mu_folder_t, const char *, void *, int,
				    size_t, mu_list_t, mu_folder_enumerate_fp,
				    void *);
static int folder_mbox_subscribe   (mu_folder_t, const char *);
static int folder_mbox_unsubscribe (mu_folder_t, const char *);
static int folder_mbox_lsub        (mu_folder_t, const char *, const char *,
				    mu_list_t);


static char *get_pathname       (const char *, const char *);

static int folder_mbox_get_authority (mu_folder_t folder, mu_authority_t * pauth);

struct _fmbox
{
  char *dirname;
  char **subscribe;
  size_t sublen;
};
typedef struct _fmbox *fmbox_t;


int
_folder_mbox_init (mu_folder_t folder)
{
  fmbox_t dfolder;
  int status = 0;

  /* We create an authority so the API is uniform across the mailbox
     types. */
  status = folder_mbox_get_authority (folder, NULL);
  if (status != 0)
    return status;

  dfolder = folder->data = calloc (1, sizeof (*dfolder));
  if (dfolder == NULL)
    return ENOMEM;

  status = mu_url_aget_path (folder->url, &dfolder->dirname);
  if (status == MU_ERR_NOENT)
    {
      dfolder->dirname = malloc (2);
      if (dfolder->dirname == NULL)
	status = ENOMEM;
      else
	{
	  strcpy (dfolder->dirname, ".");
	  status = 0;
	}
    }
  
  if (status)  
    {
      free (dfolder);
      folder->data = NULL;
      return status;
    }

  folder->_destroy = folder_mbox_destroy;

  folder->_open = folder_mbox_open;
  folder->_close = folder_mbox_close;

  folder->_list = folder_mbox_list;
  folder->_lsub = folder_mbox_lsub;
  folder->_subscribe = folder_mbox_subscribe;
  folder->_unsubscribe = folder_mbox_unsubscribe;
  folder->_delete = folder_mbox_delete;
  folder->_rename = folder_mbox_rename;
  return 0;
}

void
folder_mbox_destroy (mu_folder_t folder)
{
  if (folder->data)
    {
      fmbox_t fmbox = folder->data;
      if (fmbox->dirname)
	free (fmbox->dirname);
      if (fmbox->subscribe)
	free (fmbox->subscribe);
      free (folder->data);
      folder->data = NULL;
    }
}

/* Noop. */
static int
folder_mbox_open (mu_folder_t folder, int flags MU_ARG_UNUSED)
{
  fmbox_t fmbox = folder->data;
  if (flags & MU_STREAM_CREAT)
    {
      return (mkdir (fmbox->dirname, S_IRWXU) == 0) ? 0 : errno;
    }
  return 0;
}

/*  Noop.  */
static int
folder_mbox_close (mu_folder_t folder MU_ARG_UNUSED)
{
  return 0;
}

static int
folder_mbox_delete (mu_folder_t folder, const char *filename)
{
  fmbox_t fmbox = folder->data;
  if (filename)
    {
      int status = 0;
      char *pathname = get_pathname (fmbox->dirname, filename);
      if (pathname)
	{
	  if (remove (pathname) != 0)
	    status = errno;
	  free (pathname);
	}
      else
	status = ENOMEM;
      return status;
    }
  return EINVAL;
}

static int
folder_mbox_rename (mu_folder_t folder, const char *oldpath,
		    const char *newpath)
{
  fmbox_t fmbox = folder->data;
  if (oldpath && newpath)
    {
      int status = 0;
      char *pathold = get_pathname (fmbox->dirname, oldpath);
      if (pathold)
	{
	  char *pathnew = get_pathname (fmbox->dirname, newpath);
	  if (pathnew)
	    {
	      if (rename (pathold, pathnew) != 0)
		status = errno;
	      free (pathnew);
	    }
	  else
	    status = ENOMEM;
	  free (pathold);
	}
      else
	status = ENOMEM;
      return status;
    }
  return EINVAL;
}

struct inode_list           /* Inode/dev number list used to cut off
			       recursion */
{
  struct inode_list *next;
  ino_t inode;
  dev_t dev;
};

struct search_data
{
  mu_list_t result;
  mu_folder_enumerate_fp enumfun;
  void *enumdata;
  char *dirname;
  size_t dirlen;
  void *pattern;
  int flags;
  size_t max_level;
  size_t errcnt;
  mu_folder_t folder;
};

static int
inode_list_lookup (struct inode_list *list, struct stat *st)
{
  for (; list; list = list->next)
    if (list->inode == st->st_ino && list->dev == st->st_dev)
      return 1;
  return 0;
}

static int
list_helper (struct search_data *data, mu_record_t record,
	     const char *dirname, size_t level,
	     struct inode_list *ilist)
{
  DIR *dirp;
  struct dirent *dp;
  int stop = 0;
  
  if (data->max_level && level > data->max_level)
    return 0;

  dirp = opendir (dirname);
  if (dirp == NULL)
    {
      MU_DEBUG2 (data->folder->debug, MU_DEBUG_ERROR,
		 "list_helper cannot open directory %s: %s",
		 dirname, mu_strerror (errno));
      data->errcnt++;
      return 1;
    }

  if (!record)
    {
      int type;
      mu_registrar_lookup (dirname, MU_FOLDER_ATTRIBUTE_ALL,
			   &record, &type);
    }
  
  while ((dp = readdir (dirp)))
    {
      char const *ename = dp->d_name;
      char *fname;
      struct stat st;
      
      if (ename[ename[0] != '.' ? 0 : ename[1] != '.' ? 1 : 2] == 0)
	continue;
      fname = get_pathname (dirname, ename);
      if (stat (fname, &st) == 0)
	{
	  int f;
	  if (S_ISDIR (st.st_mode))
	    f = MU_FOLDER_ATTRIBUTE_DIRECTORY;
	  else if (S_ISREG (st.st_mode))
	    f = MU_FOLDER_ATTRIBUTE_FILE;
	  else
	    f = 0;
	  if (mu_record_list_p (record, ename, f))
	    {
	      if (data->folder->_match == NULL
		  || data->folder->_match (fname + data->dirlen +
					   ((data->dirlen > 1
					     && data->dirname[data->dirlen-1] != '/') ?
					    1 : 0),
					   data->pattern,
					   data->flags) == 0)
		{
		  char *refname = fname;
		  int type = 0;
		  struct mu_list_response *resp;
		  mu_record_t rec = NULL;
		  
		  resp = malloc (sizeof (*resp));
		  if (resp == NULL)
		    {
		      MU_DEBUG1 (data->folder->debug, MU_DEBUG_ERROR,
				 "list_helper: %s", mu_strerror (ENOMEM));
		      data->errcnt++;
		      free (fname);
		      continue;
		    }
		  
		  mu_registrar_lookup (refname, MU_FOLDER_ATTRIBUTE_ALL,
				       &rec, &type);
		  
		  resp->name = fname;
		  resp->level = level;
		  resp->separator = '/';
		  resp->type = type;
		  
		  if (resp->type == 0)
		    {
		      free (resp->name);
		      free (resp);
		      continue;
		    }
		  
		  if (data->enumfun)
		    {
		      if (data->enumfun (data->folder, resp, data->enumdata))
			{
			  free (resp->name);
			  free (resp);
			  stop = 1;
			  break;
			}
		    }
		  
		  if (data->result)
		    {
		      fname = NULL;
		      mu_list_append (data->result, resp);
		    }
		  else
		    free (resp);
		  
		  if ((type & MU_FOLDER_ATTRIBUTE_DIRECTORY)
		      && !inode_list_lookup (ilist, &st))
		    {
		      struct inode_list idata;
		      
		      idata.inode = st.st_ino;
		      idata.dev   = st.st_dev;
		      idata.next  = ilist;
		      stop = list_helper (data, rec, refname, level + 1,
					  &idata);
		    }
		}
	      else if (S_ISDIR (st.st_mode))
		{
		  struct inode_list idata;
		  
		  idata.inode = st.st_ino;
		  idata.dev   = st.st_dev;
		  idata.next  = ilist;
		  stop = list_helper (data, NULL, fname, level + 1, &idata);
		}
	    }
	}
      else
	{
	  MU_DEBUG2 (data->folder->debug, MU_DEBUG_ERROR,
		     "list_helper cannot stat %s: %s",
		     fname, mu_strerror (errno));
	}
      free (fname);
    }
  closedir (dirp);
  return stop;
}

static int
folder_mbox_list (mu_folder_t folder, const char *ref,
		  void *pattern,
		  int flags,
		  size_t max_level,
		  mu_list_t flist,
		  mu_folder_enumerate_fp enumfun, void *enumdata)
{
  fmbox_t fmbox = folder->data;
  struct inode_list iroot;
  struct search_data sdata;
  
  memset (&iroot, 0, sizeof iroot);
  sdata.dirname = get_pathname (fmbox->dirname, ref);
  sdata.dirlen = strlen (sdata.dirname);
  sdata.result = flist;
  sdata.enumfun = enumfun;
  sdata.enumdata = enumdata;
  sdata.pattern = pattern;
  sdata.flags = flags;
  sdata.max_level = max_level;
  sdata.folder = folder;
  sdata.errcnt = 0;
  list_helper (&sdata, NULL, sdata.dirname, 0, &iroot);
  free (sdata.dirname);
  /* FIXME: error code */
  return 0;
}

static int
folder_mbox_lsub (mu_folder_t folder, const char *ref MU_ARG_UNUSED,
		  const char *name,
		  mu_list_t flist)
{
  fmbox_t fmbox = folder->data;
  int status;
  
  if (name == NULL || *name == '\0')
    name = "*";

  if (fmbox->sublen > 0)
    {      
      size_t i;

      for (i = 0; i < fmbox->sublen; i++)
	{
	  if (fmbox->subscribe[i]
	      && fnmatch (name, fmbox->subscribe[i], 0) == 0)
	    {
	      struct mu_list_response *resp;
	      resp = malloc (sizeof (*resp));
	      if (resp == NULL)
		{
		  status = ENOMEM;
		  break;
		}
	      else if ((resp->name = strdup (fmbox->subscribe[i])) == NULL)
		{
		  free (resp);
		  status = ENOMEM;
		  break;
		}
	      resp->type = MU_FOLDER_ATTRIBUTE_FILE;
	      resp->level = 0;
	      resp->separator = '/';
	    }
	}
    }
  return status;
}

static int
folder_mbox_subscribe (mu_folder_t folder, const char *name)
{
  fmbox_t fmbox = folder->data;
  char **tmp;
  size_t i;
  for (i = 0; i < fmbox->sublen; i++)
    {
      if (fmbox->subscribe[i] && strcmp (fmbox->subscribe[i], name) == 0)
	return 0;
    }
  tmp = realloc (fmbox->subscribe, (fmbox->sublen + 1) * sizeof (*tmp));
  if (tmp == NULL)
    return ENOMEM;
  fmbox->subscribe = tmp;
  fmbox->subscribe[fmbox->sublen] = strdup (name);
  if (fmbox->subscribe[fmbox->sublen] == NULL)
    return ENOMEM;
  fmbox->sublen++;
  return 0;
}

static int
folder_mbox_unsubscribe (mu_folder_t folder, const char *name)
{
  fmbox_t fmbox = folder->data;
  size_t i;
  for (i = 0; i < fmbox->sublen; i++)
    {
      if (fmbox->subscribe[i] && strcmp (fmbox->subscribe[i], name) == 0)
	{
	  free (fmbox->subscribe[i]);
	  fmbox->subscribe[i] = NULL;
	  return 0;
	}
    }
  return MU_ERR_NOENT;
}

static char *
get_pathname (const char *dirname, const char *basename)
{
  char *pathname = NULL;

  /* Skip eventual protocol designator.
     FIXME: Actually, any valid URL spec should be allowed as dirname ... */
  if (strncmp (dirname, MU_MBOX_SCHEME, MU_MBOX_SCHEME_LEN) == 0)
    dirname += MU_MBOX_SCHEME_LEN;
  else if (strncmp (dirname, MU_FILE_SCHEME, MU_FILE_SCHEME_LEN) == 0)
    dirname += MU_FILE_SCHEME_LEN;
  
  /* null basename gives dirname.  */
  if (basename == NULL)
    pathname = (dirname) ? strdup (dirname) : strdup (".");
  /* Absolute.  */
  else if (basename[0] == '/')
    pathname = strdup (basename);
  /* Relative.  */
  else
    {
      size_t baselen = strlen (basename);
      size_t dirlen = strlen (dirname);
      while (dirlen > 0 && dirname[dirlen-1] == '/')
	dirlen--;
      pathname = calloc (dirlen + baselen + 2, sizeof (char));
      if (pathname)
	{
	  memcpy (pathname, dirname, dirlen);
	  pathname[dirlen] = '/';
	  strcpy (pathname + dirlen + 1, basename);
	}
    }
  return pathname;
}

static int
folder_mbox_get_authority (mu_folder_t folder, mu_authority_t *pauth)
{
  int status = 0;
  if (folder->authority == NULL)
    {
	status = mu_authority_create_null (&folder->authority, folder);
    }
  if (!status && pauth)
    *pauth = folder->authority;
  return status;
}

