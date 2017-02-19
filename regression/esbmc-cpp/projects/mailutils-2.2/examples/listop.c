/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2004, 2005, 2007, 2010 Free Software Foundation,
   Inc.

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
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <mailutils/argcv.h>
#include <mailutils/mailutils.h>

void
usage(int code)
{
  printf ("usage: listop [item..]\n");
  exit (code);
}

void
lperror (char *text, int rc)
{
  fprintf (stderr, "%s: %s\n", text, mu_strerror (rc));
  exit (1);
}

void
print (mu_list_t list)
{
  mu_iterator_t itr;
  size_t count;
  int rc;
  
  rc = mu_list_get_iterator (list, &itr);
  if (rc)
    lperror ("mu_list_get_iterator", rc);

  rc = mu_list_count (list, &count);
  if (rc)
    lperror ("mu_iterator_current", rc);

  printf ("# items: %lu\n", (unsigned long) count);
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      char *text;

      rc = mu_iterator_current (itr, (void**) &text);
      if (rc)
	lperror ("mu_iterator_current", rc);
      printf ("%s\n", text);
    }
  mu_iterator_destroy (&itr);
}

void
count (mu_list_t list)
{
  size_t n;
  int rc;

  rc = mu_list_count (list, &n);
  if (rc)
    lperror ("mu_iterator_current", rc);
  else
    printf ("%lu\n", (unsigned long) n);
}

void
next (mu_iterator_t itr, char *arg)
{
  int skip = arg ? strtoul (arg, NULL, 0) :  1;

  if (skip == 0)
    fprintf (stderr, "next arg?\n");
  while (skip--)
    mu_iterator_next (itr);
}

void
delete (mu_list_t list, int argc, char **argv)
{
  int rc;

  if (argc == 1)
    {
      fprintf (stderr, "del arg?\n");
      return;
    }

  while (--argc)
    {
      rc = mu_list_remove (list, *++argv);
      if (rc)
	fprintf (stderr, "mu_list_remove(%s): %s\n", *argv, mu_strerror (rc));
    }
}

void
add (mu_list_t list, int argc, char **argv)
{
  int rc;
  
  if (argc == 1)
    {
      fprintf (stderr, "add arg?\n");
      return;
    }

  while (--argc)
    {
      rc = mu_list_append (list, strdup (*++argv));
      if (rc)
	fprintf (stderr, "mu_list_append: %s\n", mu_strerror (rc));
    }
}

void
prep (mu_list_t list, int argc, char **argv)
{
  int rc;
  
  if (argc == 1)
    {
      fprintf (stderr, "add arg?\n");
      return;
    }

  while (--argc)
    {
      rc = mu_list_prepend (list, strdup (*++argv));
      if (rc)
	fprintf (stderr, "mu_list_append: %s\n", mu_strerror (rc));
    }
}

static mu_list_t
read_list (int argc, char **argv)
{
  int rc;
  mu_list_t list;
  
  rc = mu_list_create (&list);
  if (rc)
    {
      fprintf (stderr, "creating temp list: %s\n", mu_strerror (rc));
      return NULL;
    }
  mu_list_set_destroy_item (list, mu_list_free_item);
  for (; argc; argc--, argv++)
    {
      rc = mu_list_append (list, strdup (*argv));
      if (rc)
	{
	  mu_list_destroy (&list);
	  fprintf (stderr, "adding to temp list: %s\n", mu_strerror (rc));
	  break;
	}
    }
  return list;
}

void
ins (mu_list_t list, int argc, char **argv)
{
  int an;
  int rc;
  char *item;
  int insert_before = 0;
  
  if (argc < 3)
    {
      fprintf (stderr, "ins [before] item new_item [new_item*]?\n");
      return;
    }

  an = 1;
  if (strcmp (argv[1], "before") == 0)
    {
      an++;
      insert_before = 1;
    }
  else if (strcmp (argv[1], "after") == 0)
    {
      an++;
      insert_before = 0;
    }

  item = argv[an++];
  
  if (an + 1 == argc)
    rc = mu_list_insert (list, item, strdup (argv[an]), insert_before);
  else
    {
      mu_list_t tmp = read_list (argc - an, argv + an);
      if (!tmp)
	return;
      rc = mu_list_insert_list (list, item, tmp, insert_before);
      mu_list_destroy (&tmp);
    }

  if (rc)
    lperror ("mu_list_insert", rc);
}
  
void
repl (mu_list_t list, int argc, char **argv)
{
  int rc;
  
  if (argc != 3)
    {
      fprintf (stderr, "repl src dst?\n");
      return;
    }

  rc = mu_list_replace (list, argv[1], strdup (argv[2]));
  if (rc)
    fprintf (stderr, "mu_list_replace: %s\n", mu_strerror (rc));
}

void
ictl_tell (mu_iterator_t itr, int argc)
{
  size_t pos;
  int rc;

  if (argc)
    {
      fprintf (stderr, "ictl tell?\n");
      return;
    }
  
  rc = mu_iterator_ctl (itr, mu_itrctl_tell, &pos);
  if (rc)
    lperror ("mu_iterator_ctl", rc);
  printf ("%lu\n", (unsigned long) pos);
}

void
ictl_del (mu_iterator_t itr, int argc)
{
  int rc;

  if (argc)
    {
      fprintf (stderr, "ictl del?\n");
      return;
    }
  rc = mu_iterator_ctl (itr, mu_itrctl_delete, NULL);
  if (rc)
    lperror ("mu_iterator_ctl", rc);
}

void
ictl_repl (mu_iterator_t itr, int argc, char **argv)
{
  int rc;
  
  if (argc != 1)
    {
      fprintf (stderr, "ictl repl item?\n");
      return;
    }

  rc = mu_iterator_ctl (itr, mu_itrctl_replace, strdup (argv[0]));
  if (rc)
    lperror ("mu_iterator_ctl", rc);
}

void
ictl_dir (mu_iterator_t itr, int argc, char **argv)
{
  int rc;
  int dir;
  
  if (argc > 1)
    {
      fprintf (stderr, "ictl dir [backwards|forwards]?\n");
      return;
    }
  if (argc == 1)
    {
      if (strcmp (argv[0], "backwards") == 0)
	dir = 1;
      else if (strcmp (argv[0], "forwards") == 0)
	dir = 0;
      else
	{
	  fprintf (stderr, "ictl dir [backwards|forwards]?\n");
	  return;
	}
      rc = mu_iterator_ctl (itr, mu_itrctl_set_direction, &dir);
      if (rc)
	lperror ("mu_iterator_ctl", rc);
    }
  else
    {
      rc = mu_iterator_ctl (itr, mu_itrctl_qry_direction, &dir);
      if (rc)
	lperror ("mu_iterator_ctl", rc);
      printf ("%s\n", dir ? "backwards" : "forwards");
    }
}
  
void
ictl_ins (mu_iterator_t itr, int argc, char **argv)
{
  int rc;
  
  if (argc < 1)
    {
      fprintf (stderr, "ictl ins item [item*]?\n");
      return;
    }

  if (argc == 1)
    rc = mu_iterator_ctl (itr, mu_itrctl_insert, strdup (argv[0]));
  else
    {
      mu_list_t tmp = read_list (argc, argv);
      if (!tmp)
	return;
      rc = mu_iterator_ctl (itr, mu_itrctl_insert_list, tmp);
      mu_list_destroy (&tmp);
    }
}

void
ictl (mu_iterator_t itr, int argc, char **argv)
{
  if (argc == 1)
    {
      fprintf (stderr, "ictl tell|del|repl|ins?\n");
      return;
    }
  
  if (strcmp (argv[1], "tell") == 0)
    ictl_tell (itr, argc - 2);
  else if (strcmp (argv[1], "del") == 0)
    ictl_del (itr, argc - 2);
  else if (strcmp (argv[1], "repl") == 0)
    ictl_repl (itr, argc - 2, argv + 2);
  else if (strcmp (argv[1], "ins") == 0)
    ictl_ins (itr, argc - 2, argv + 2);
  else if (strcmp (argv[1], "dir") == 0)
    ictl_dir (itr, argc - 2, argv + 2);
  else
    fprintf (stderr, "unknown subcommand\n");
}
    
#define NITR 4

int
iter (int *pnum, int argc, char **argv)
{
  int n;
  
  if (argc != 2)
    {
      fprintf (stderr, "iter num?\n");
      return 1;
    }

  n = strtoul (argv[1], NULL, 0);
  if (n < 0 || n >= NITR)
    {
      fprintf (stderr, "iter [0-3]?\n");
      return 1;
    }
  *pnum = n;
  return 0;
}

void
find (mu_iterator_t itr, char *arg)
{
  char *text;
  
  if (!arg)
    {
      fprintf (stderr, "find item?\n");
      return;
    }

  mu_iterator_current (itr, (void**)&text);
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
    {
      char *item;

      mu_iterator_current (itr, (void**)&item);
      if (strcmp (arg, item) == 0)
	return;
    }

  fprintf (stderr, "%s not in list\n", arg);
}

void
help ()
{
  printf ("count\n");
  printf ("next [count]\n");
  printf ("first\n");
  printf ("find item\n");
  printf ("del item [item*]\n");
  printf ("add item [item*]\n");
  printf ("prep item [item*]\n");
  printf ("repl old_item new_item\n");
  printf ("ins [before|after] item new_item [new_item*]\n");
  printf ("ictl tell\n");
  printf ("ictl del\n");
  printf ("ictl repl item\n");
  printf ("ictl ins item [item*]\n");
  printf ("ictl dir [backwards|forwards]\n");
  printf ("print\n");
  printf ("quit\n");
  printf ("iter num\n");
  printf ("help\n");
  printf ("NUMBER\n");
}

void
shell (mu_list_t list)
{
  int num = 0;
  mu_iterator_t itr[NITR];
  int rc;

  memset (&itr, 0, sizeof itr);
  num = 0;
  while (1)
    {
      char *text;
      char buf[80];
      int argc;
      char **argv;

      if (!itr[num])
	{
	  rc = mu_list_get_iterator (list, &itr[num]);
	  if (rc)
	    lperror ("mu_list_get_iterator", rc);
	  mu_iterator_first (itr[num]);
	}
      
      rc = mu_iterator_current (itr[num], (void**) &text);
      if (rc)
	lperror ("mu_iterator_current", rc);

      printf ("%d:(%s)> ", num, text ? text : "NULL");
      if (fgets (buf, sizeof buf, stdin) == NULL)
	return;

      rc = mu_argcv_get (buf, "", "#", &argc, &argv);
      if (rc)
	lperror ("mu_argcv_get", rc);

      if (argc > 0)
	{
	  if (strcmp (argv[0], "count") == 0)
	    count (list);
	  else if (strcmp (argv[0], "next") == 0)
	    next (itr[num], argv[1]);
	  else if (strcmp (argv[0], "first") == 0)
	    mu_iterator_first (itr[num]);
	  else if (strcmp (argv[0], "del") == 0)
	    delete (list, argc, argv);
	  else if (strcmp (argv[0], "add") == 0)
	    add (list, argc, argv);
	  else if (strcmp (argv[0], "prep") == 0)
	    prep (list, argc, argv);
	  else if (strcmp (argv[0], "ins") == 0)
	    ins (list, argc, argv);
	  else if (strcmp (argv[0], "repl") == 0)
	    repl (list, argc, argv);
	  else if (strcmp (argv[0], "ictl") == 0)
	    ictl (itr[num], argc, argv);
	  else if (strcmp (argv[0], "print") == 0)
	    print (list);
	  else if (strcmp (argv[0], "quit") == 0)
	    return;
	  else if (strcmp (argv[0], "iter") == 0)
	    {
	      int n;
	      if (iter (&n, argc, argv) == 0 && !itr[n])
		{
		  rc = mu_list_get_iterator (list, &itr[n]);
		  if (rc)
		    lperror ("mu_list_get_iterator", rc);
		  mu_iterator_first (itr[n]);
		}
	      num = n;
	    }
	  else if (strcmp (argv[0], "close") == 0)
	    {
	      int n;
	      if (iter (&n, argc, argv) == 0)
		{
		  mu_iterator_destroy (&itr[n]);
		  if (n == num && ++num == NITR)
		    num = 0;
		}
	    }
	  else if (strcmp (argv[0], "find") == 0)
	    find (itr[num], argv[1]);
	  else if (strcmp (argv[0], "help") == 0)
	    help ();
	  else if (argc == 1)
	    {
	      char *p;
	      size_t n = strtoul (argv[0], &p, 0);
	      if (*p != 0)
		fprintf (stderr, "?\n");
	      else
		{
		  rc = mu_list_get (list, n, (void**) &text);
		  if (rc)
		    fprintf (stderr, "mu_list_get: %s\n", mu_strerror (rc));
		  else
		    printf ("%s\n", text);
		}
	    }
	  else
	    fprintf (stderr, "?\n");
	}
      mu_argcv_free (argc, argv);
    }
}

static int
string_comp (const void *item, const void *value)
{
  return strcmp (item, value);
}

int
main (int argc, char **argv)
{
  mu_list_t list;
  int rc;
  
  while ((rc = getopt (argc, argv, "h")) != EOF)
    switch (rc)
      {
      case 'h':
	usage (0);
	
      default:
	usage (1);
      }

  argc -= optind;
  argv += optind;

  rc = mu_list_create (&list);
  if (rc)
    lperror ("mu_list_create", rc);
  mu_list_set_comparator (list, string_comp);
  mu_list_set_destroy_item (list, mu_list_free_item);
  
  while (argc--)
    {
      rc = mu_list_append (list, *argv++);
      if (rc)
	lperror ("mu_list_append", rc);
    }

  shell (list);
  
  return 0;
}
