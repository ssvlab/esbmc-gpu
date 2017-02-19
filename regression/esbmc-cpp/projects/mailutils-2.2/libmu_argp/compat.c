/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2009, 2010 Free Software Foundation, Inc.

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
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include "mailutils/cctype.h"
#include "mailutils/libargp.h"
#include "mailutils/argcv.h"
#include "mailutils/mutil.h"
#ifdef WITH_TLS
# include "mailutils/tls.h"
#endif
#ifdef WITH_GSASL
# include "mailutils/gsasl.h"
#endif
#include "mailutils/sieve.h"

#ifndef MU_COMPAT_CONFIG_FILE
# define MU_COMPAT_CONFIG_FILE SYSCONFDIR "/mailutils.rc"
#endif

#ifndef MU_COMPAT_USER_CONFIG_FILE
# define MU_COMPAT_USER_CONFIG_FILE "~/.mailutils"
#endif

static int
member (const char *array[], const char *text, size_t len)
{
  int i;
  for (i = 0; array[i]; i++)
    if (strncmp (array[i], text, len) == 0)
      return 1;
  return 0;
}

/* Appends applicable options found in file NAME to argv. If progname
   is NULL, all the options found are assumed to apply. Otherwise they
   apply only if the line starts with ":something", and something is
   found in the CAPA array, or the line starts with PROGNAME.
*/
void
read_rc (const char *progname, const char *name, const char *capa[],
	 int *argc, char ***argv)
{
  FILE *fp;
  char *linebuf = NULL;
  char *buf = NULL;
  size_t n = 0;
  int x_argc = *argc;
  char **x_argv = *argv;
  char* rcfile = mu_tilde_expansion (name, "/", NULL);

  if (!rcfile)
    return;
  
  fp = fopen (rcfile, "r");
  if (!fp)
    {
      free(rcfile);
      return;
    }
  
  while (getline (&buf, &n, fp) > 0)
    {
      char *kwp, *p;
      int len;
      
      for (kwp = buf; *kwp && mu_isspace (*kwp); kwp++)
	;

      if (*kwp == '#' || *kwp == 0)
	continue;

      len = strlen (kwp);
      if (kwp[len-1] == '\n')
	kwp[--len] = 0;

      if (kwp[len-1] == '\\' || linebuf)
	{
	  int cont;
	  
	  if (kwp[len-1] == '\\')
	    {
	      kwp[--len] = 0;
	      cont = 1;
	    }
	  else
	    cont = 0;
	  
	  if (!linebuf)
	    linebuf = calloc (len + 1, 1);
	  else
	    linebuf = realloc (linebuf, strlen (linebuf) + len + 1);
	  
	  if (!linebuf)
	    {
	      fprintf (stderr, _("%s: not enough memory\n"), progname);
	      exit (1);
	    }
	  
	  strcpy (linebuf + strlen (linebuf), kwp);
	  if (cont)
	    continue;
	  kwp = linebuf;
	}

      len = 0;
      if (progname)
	{
	  for (p = kwp; *p && !mu_isspace (*p); p++)
	    len++;
	}
      else
	p = kwp; /* Use the whole line. */

      if (progname == NULL
	  || (kwp[0] == ':' && member (capa, kwp+1, len-1))
	  || strncmp (progname, kwp, len) == 0
	  )
	{
	  int i, n_argc = 0;
	  char **n_argv;
              
	  if (mu_argcv_get (p, "", NULL, &n_argc, &n_argv))
	    {
	      mu_argcv_free (n_argc, n_argv);
	      if (linebuf)
		free (linebuf);
	      linebuf = NULL;
	      continue;
	    }
	  x_argv = realloc (x_argv,
			    (x_argc + n_argc) * sizeof (x_argv[0]));
	  if (!x_argv)
	    {
	      fprintf (stderr, _("%s: not enough memory\n"), progname);
	      exit (1);
	    }
	  
	  for (i = 0; i < n_argc; i++)
	    x_argv[x_argc++] = mu_tilde_expansion (n_argv[i], "/", NULL);
	  
	  free (n_argv);
	}
      if (linebuf)
	free (linebuf);
      linebuf = NULL;
    }
  fclose (fp);
  free(rcfile);

  *argc = x_argc;
  *argv = x_argv;
}


void
mu_create_argcv (const char *capa[],
		 int argc, char **argv, int *p_argc, char ***p_argv)
{
  char *progname;
  int x_argc;
  char **x_argv;
  int i;
  int rcdir = 0;

  progname = strrchr (argv[0], '/');
  if (progname)
    progname++;
  else
    progname = argv[0];

  x_argv = malloc (sizeof (x_argv[0]));
  if (!x_argv)
    {
      fprintf (stderr, _("%s: not enough memory\n"), progname);
      exit (1);
    }

  /* Add command name */
  x_argc = 0;
  x_argv[x_argc] = argv[x_argc];
  x_argc++;

  /* Add global config file. */
  read_rc (progname, MU_COMPAT_CONFIG_FILE, capa, &x_argc, &x_argv);

  /* Look for per-user config files in ~/.mailutils/ or in ~/, but
     not both. This allows mailutils' utilities to have their config
     files segregated, if necessary. */

  {
    struct stat s;
    char *rcdirname = mu_tilde_expansion (MU_COMPAT_USER_CONFIG_FILE, "/", NULL);

    if (!rcdirname
	|| (stat(rcdirname, &s) == 0 && S_ISDIR(s.st_mode)))
      rcdir = 1;

    free(rcdirname);
  }

  /* Add per-user config file. */
  if (!rcdir)
    {
      read_rc (progname, MU_COMPAT_USER_CONFIG_FILE, capa, &x_argc, &x_argv);
    }
  else
    {
      char *userrc = NULL;

      userrc = malloc (sizeof (MU_COMPAT_USER_CONFIG_FILE)
		       /* provides an extra slot
			  for null byte as well */
		       + 1 /* slash */
		       + 9 /*mailutils*/); 

      if (!userrc)
	{
	  fprintf (stderr, _("%s: not enough memory\n"), progname);
	  exit (1);
	}
      
      sprintf (userrc, "%s/mailutils", MU_COMPAT_USER_CONFIG_FILE);
      read_rc (progname, userrc, capa, &x_argc, &x_argv);
      
      free (userrc);
    }

  /* Add per-user, per-program config file. */
  {
    char *progrc = NULL;
    int size;
    
    if (rcdir)
      size = sizeof (MU_COMPAT_USER_CONFIG_FILE)
	             + 1
		     + strlen (progname)
		     + 2 /* rc */;
    else
      size = 6 /*~/.mu.*/
	     + strlen (progname)
	     + 3 /* "rc" + null terminator */;

    progrc = malloc (size);

    if (!progrc)
      {
	fprintf (stderr, _("%s: not enough memory\n"), progname);
	exit (1);
      }

    if (rcdir)
      sprintf (progrc, "%s/%src", MU_COMPAT_USER_CONFIG_FILE, progname);
    else
      sprintf (progrc, "~/.mu.%src", progname);

    read_rc (NULL, progrc, capa, &x_argc, &x_argv);
    free (progrc);
  }

  /* Finally, add the command line options */
  x_argv = realloc (x_argv, (x_argc + argc) * sizeof (x_argv[0]));
  for (i = 1; i < argc; i++)
    x_argv[x_argc++] = argv[i];

  x_argv[x_argc] = NULL;

  *p_argc = x_argc;
  *p_argv = x_argv;
}

error_t
mu_argp_parse (const struct argp *myargp, 
	       int *pargc, char **pargv[],  
	       unsigned flags,
	       const char *capa[],
	       int *arg_index,     
	       void *input)
{
  struct argp *argp;
  error_t rc;
  const struct argp argpnull = { 0 };
  int i;
  
  /* Make sure we have program version and bug address initialized */
  mu_argp_init (argp_program_version, argp_program_bug_address);

  mu_set_program_name ((*pargv)[0]);
  mu_libargp_init ();
  for (i = 0; capa[i]; i++)
    {
#ifdef WITH_TLS
      if (strcmp (capa[i], "tls") == 0)
	mu_gocs_register ("tls", mu_tls_module_init);
      else
#endif /* WITH_TLS */
#ifdef WITH_GSASL
      if (strcmp (capa[i], "gsasl") == 0)
	mu_gocs_register ("gsasl", mu_gsasl_module_init);
      else
#endif
      if (strcmp (capa[i], "sieve") == 0)
	mu_gocs_register ("sieve", mu_sieve_module_init);
      else
	mu_gocs_register_std (capa[i]); 
    }
  
  if (!myargp)
    myargp = &argpnull;
  argp = mu_argp_build (myargp, NULL);
  rc = argp_parse (argp, *pargc, *pargv, flags, arg_index, input);
  mu_argp_done (argp);
  if (rc)
    return rc;

  mu_gocs_flush ();

  return 0;
}
