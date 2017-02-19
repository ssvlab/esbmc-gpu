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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <mailutils/mailutils.h>

static int
enumfun (mu_folder_t folder, struct mu_list_response *resp, void *data)
{
  printf ("%c%c %c %4d %s\n",
	  (resp->type & MU_FOLDER_ATTRIBUTE_DIRECTORY) ? 'd' : '-',
	  (resp->type & MU_FOLDER_ATTRIBUTE_FILE) ? 'f' : '-',
	  resp->separator,
	  resp->level,
	  resp->name);
  return 0;
}

int
ls_folders (char *fname, char *ref, char *pattern, int level)
{
  int status;
  mu_folder_t folder;
  mu_list_t flist;
  size_t count;
  
  status = mu_folder_create (&folder, fname);
  if (status)
    {
      mu_error ("mu_folder_create failed: %s", mu_strerror (status));
      return 1;
    }
  
  status = mu_folder_open (folder, MU_STREAM_READ);
  if (status)
    {
      mu_error ("mu_folder_create failed: %s", mu_strerror (status));
      return 1;
    }

  status = mu_folder_enumerate (folder, ref, pattern, 0, level, &flist,
				enumfun, NULL);
  
  switch (status)
    {
    case 0:
      mu_list_count (flist, &count);
      printf ("Number of folders: %lu\n", (unsigned long) count);
      mu_list_destroy (&flist);
      break;
    case MU_ERR_NOENT:
      printf ("No folders matching %s %s in %s\n", ref, pattern, fname);
      return 0;

    default:
      mu_error ("mu_folder_list failed: %s", mu_strerror (status));
    }
  return 0;
}

int
main (int argc, char *argv[])
{
  char *folder;
  char *ref = NULL;
  char *pattern = "*";
  int level = 0;

  switch (argc)
    {
    case 5:
      level = atoi (argv[4]);
    case 4:
      pattern = argv[3];
    case 3:
      ref = argv[2];
    case 2:
      folder = argv[1];
      break;
    default:
      mu_error ("usage: lsf folder [ref] [pattern] [recursion-level]\n");
      return 1;
    }

  mu_register_all_mbox_formats ();

  return ls_folders (folder, ref, pattern, level);
}
