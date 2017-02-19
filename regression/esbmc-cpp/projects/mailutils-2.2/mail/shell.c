/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2005, 2007, 2009, 2010 Free Software
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

#include "mail.h"

static void
expand_bang (char **pbuf)
{
  char *last = NULL;
  char *tmp, *p, *q;
  size_t count = 0;
  
  mailvar_get (&last, "gnu-last-command", mailvar_type_string, 0);

  for (p = *pbuf; *p; p++)
    if (*p == '!')
      count++;

  if (count == 0)
    return;

  if (!last)
    {
      util_error (_("No previous command"));
      return;
    }

  tmp = xmalloc (strlen (*pbuf) + count * (strlen (last) - 1) + 1);
  for (p = *pbuf, q = tmp; *p; )
    {
      if (*p == '!')
	{
	  strcpy (q, last);
	  q += strlen (q);
	  p++;
	}
      else
	*p++ = *q++;
    }
  *q = 0;
  
  free (*pbuf);
  *pbuf = tmp;
}

int
mail_execute (int shell, int argc, char **argv)
{
  pid_t pid;
  char *buf = NULL;
  char *argv0 = NULL;

  if (argc)
    {
      argv0 = argv[0]; 
  
      /* Skip leading whitespace from argv[0] */
      while (mu_isspace (**argv))
	(*argv)++;
    }

  /* Expand arguments if required */
  if (mailvar_get (NULL, "bang", mailvar_type_boolean, 0) == 0)
    {
      int i;

      for (i = 0; i < argc; i++)
	expand_bang (argv + i);
    }

  /* Construct command line and save it to gnu-last-command variable */
  mu_argcv_string (argc, argv, &buf);
  mailvar_set ("gnu-last-command", buf, mailvar_type_string, 
             MOPTF_QUIET|MOPTF_OVERWRITE);

  /* Do actual work */
  
  pid = fork ();  
  if (pid == 0)
    {
      if (shell)
	{
	  if (argc == 0)
	    {
	      argv = xmalloc (sizeof (argv[0]) * 2);
	      argv[0] = getenv ("SHELL");
	      argv[1] = NULL;
	      argc = 1;
	    }
	  else
	    {
	      /* 1(shell) + 1 (-c) + 1(arg) + 1 (null) = 4  */
	      argv = xmalloc (4 * (sizeof (argv[0])));
	  
	      argv[0] = getenv ("SHELL");
	      argv[1] = "-c";
	      argv[2] = buf;
	      argv[3] = NULL;

	      argc = 3;
	    }
	}
      
      execvp (argv[0], argv);
      exit (1);
    }
  else
    {
      if (argv0) /* Restore argv[0], else mu_argcv_free will coredump */
	argv[0] = argv0;
      free (buf);
      if (pid > 0)
	{
	  while (waitpid (pid, NULL, 0) == -1)
	    /* do nothing */;
	  return 0;
	}
      else /* if (pid < 0) */
	{
	  mu_error ("fork failed: %s", mu_strerror (errno));
	  return 1;
	}
    }
}

/*
 * sh[ell] [command] -- GNU extension
 * ![command] -- GNU extension
 */

int
mail_shell (int argc, char **argv)
{
  if (argv[0][0] == '!' && strlen (argv[0]) > 1)
    {
      argv[0][0] = ' ';
      return mail_execute (1, argc, argv);
    }
  else if (argc > 1)
    {
      return mail_execute (0, argc-1, argv+1);
    }
  else
    {
      return mail_execute (1, 0, NULL);
    }
  return 1;
}


