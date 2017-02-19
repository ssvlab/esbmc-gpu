/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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

#include "comsat.h"
#include <mailutils/io.h>
#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
#include <obstack.h>

/* This module implements user-configurable actions for comsat. The
   actions are kept in file .biffrc in the user's home directory and
   are executed in sequence. Possible actions:

   beep              --  Produce audible signal
   echo ARGS...      --  Output ARGS to the user's tty
   exec PROG ARGS... --  Execute given program (absolute pathname
                         required)

   Following metacharacters are accepted in strings:

   $u        --  Expands to username
   $h        --  Expands to hostname
   $H{name}  --  Expands to value of message header `name'
   $B(C,L)   --  Expands to message body. C and L give maximum
                 number of characters and lines in the expansion.
		 When omitted, they default to 400, 5. */

static unsigned
act_getline (FILE *fp, char **sptr, size_t *size)
{
  char buf[256];
  int cont = 1;
  size_t used = 0;
  unsigned lines = 0;
  
  if (feof (fp))
    return 0;
  
  while (cont && fgets (buf, sizeof buf, fp))
    {
      int len = strlen (buf);
      if (buf[len-1] == '\n')
	{
	  lines++;
	  buf[--len] = 0;
	  if (buf[len-1] == '\\')
	    {
	      buf[--len] = 0;
	      cont = 1;
	    }
	  else
	    cont = 0;
	}
      else
	cont = 1;

      if (len + used + 1 > *size)
	{
	  *sptr = realloc (*sptr, len + used + 1);
	  if (!*sptr)
	    return 0;
	  *size = len + used + 1;
	}
      memcpy (*sptr + used, buf, len);
      used += len;
    }

  if (*sptr)
    (*sptr)[used] = 0;

  return lines;
}

static int
expand_escape (char **pp, mu_message_t msg, struct obstack *stk)
{
  char *p = *pp;
  char *start, *sval, *namep;
  int len;
  mu_header_t hdr;
  mu_body_t body;
  mu_stream_t stream;
  int rc = 1;
  size_t size = 0, lncount = 0;

  switch (*++p) /* skip past $ */
    {
    case 'u':
      len = strlen (username);
      obstack_grow (stk, username, len);
      *pp = p;
      rc = 0;
      break;

    case 'h':
      len = strlen (hostname);
      obstack_grow (stk, hostname, len);
      *pp = p;
      rc = 0;
      break;

    case 'H':
      /* Header field */
      if (*++p != '{')
	break;
      start = ++p;
      p = strchr (p, '}');
      if (!p)
	break;
      len = p - start;
      if (len == 0
	  || (namep = malloc (len + 1)) == NULL)
	break;
      memcpy (namep, start, len);
      namep[len] = 0;
      if (mu_message_get_header (msg, &hdr) == 0
	  && mu_header_aget_value (hdr, namep, &sval) == 0)
	{
	  len = strlen (sval);
	  obstack_grow (stk, sval, len);
	}
      free (namep);
      *pp = p;
      rc = 0;
      break;

    case 'B':
      /* Message body */
      if (*++p == '(')
	{
	  size = strtoul (p + 1, &p, 0);
	  if (*p == ',')
	    lncount = strtoul (p + 1, &p, 0);
	  if (*p != ')')
	    break;
	}
      if (size == 0)
	size = 400;
      if (lncount == 0)
	lncount = maxlines;
      if (mu_message_get_body (msg, &body) == 0
	  && mu_body_get_stream (body, &stream) == 0)
	{
	  size_t nread;
	  char *buf = malloc (size+1);

	  if (!buf)
	    break;
 	  if (mu_stream_read (stream, buf, size, 0, &nread) == 0)
	    {
	      char *q;

	      buf[nread] = 0;
	      q = buf;
	      size = 0;
	      while (lncount--)
		{
		  char *s = strchr (q, '\n');
		  if (!s)
		    break;
		  size += s - q + 1;
		  q = s + 1;
		}
	      obstack_grow (stk, buf, size);
	    }
	  free (buf);
	}
      *pp = p;
      rc = 0;
    }
  return rc;
}

static char *
expand_line (const char *str, mu_message_t msg)
{
  const char *p;
  int c = 0;
  struct obstack stk;

  if (!*str)
    return NULL;
  obstack_init (&stk);
  for (p = str; *p; p++)
    {
      switch (*p)
	{
	case '\\':
	  p++;
	  if (*p)
	    {
	      c = mu_argcv_unquote_char (*p);
	      obstack_1grow (&stk, c);
	    }
	  break;

	case '$':
	  if (expand_escape ((char**)&p, msg, &stk) == 0)
	    break;

	  /*FALLTHRU*/
	default:
	  obstack_1grow (&stk, *p);
	}
    }
  obstack_1grow (&stk, 0);
  str = strdup (obstack_finish (&stk));
  obstack_free (&stk, NULL);
  return (char *)str;
}

const char *default_action =
"Mail to \a$u@$h\a\n"
"---\n"
"From: $H{from}\n"
"Subject: $H{Subject}\n"
"---\n"
"$B(,5)\n"
"---\n";

/* Take care to clear eighth bit, so we won't upset some stupid terminals */
#define LB(c) ((c)&0x7f)

static void
action_beep (FILE *tty)
{
  fprintf (tty, "\a\a");
}

static void
echo_string (FILE *tty, const char *cr, char *str)
{
  if (!str)
    return;
  for (; *str; str++)
    {
      if (*str == '\n')
	fprintf (tty, "%s", cr);
      else
	{
	  char c = LB (*str);
	  putc (c, tty);
	}
    }
  fflush (tty);
}

static void
action_echo (FILE *tty, const char *cr, int omit_newline,
	     int argc, char **argv)
{
  int i;

  if (omit_newline)
    {
      argc--;
      argv++;
    }
  
  for (i = 0;;)
    {
      echo_string (tty, cr, argv[i]);
      if (++i < argc)
	echo_string (tty, cr, " ");
      else
	{
	  if (!omit_newline)
	    echo_string (tty, cr, "\n");
	  break;
	}
    }
}

static void
action_exec (FILE *tty, int argc, char **argv)
{
  pid_t pid;
  struct stat stb;

  if (argc == 0)
    {
      mu_diag_output (MU_DIAG_ERROR, _("no arguments for exec"));
      return;
    }

  if (argv[0][0] != '/')
    {
      mu_diag_output (MU_DIAG_ERROR, _("not an absolute pathname: %s"), argv[0]);
      return;
    }

  if (stat (argv[0], &stb))
    {
      mu_diag_funcall (MU_DIAG_ERROR, "stat", argv[0], errno);
      return;
    }

  if (stb.st_mode & (S_ISUID|S_ISGID))
    {
      mu_diag_output (MU_DIAG_ERROR, _("will not execute set[ug]id programs"));
      return;
    }

  pid = fork ();
  if (pid == 0)
    {
      close (1);
      close (2);
      dup2 (fileno (tty), 1);
      dup2 (fileno (tty), 2);
      fclose (tty);
      execv (argv[0], argv);
      mu_diag_output (MU_DIAG_ERROR, _("cannot execute %s: %s"), argv[0], strerror (errno));
      exit (0);
    }
}

static FILE *
open_rc (const char *filename, FILE *tty)
{
  struct stat stb;
  struct passwd *pw = getpwnam (username);

  /* To be on the safe side, we do not allow root to have his .biffrc */
  if (!allow_biffrc || pw->pw_uid == 0)
    return NULL;
  if (stat (filename, &stb) == 0)
    {
      if (stb.st_uid != pw->pw_uid)
	{
	  mu_diag_output (MU_DIAG_NOTICE, _("%s's %s is not owned by %s"),
		  username, filename, username);
	  return NULL;
	}
      if ((stb.st_mode & 0777) != 0600)
	{
	  fprintf (tty, "%s\r\n",
		   _("Warning: your .biffrc has wrong permissions"));
	  mu_diag_output (MU_DIAG_NOTICE, _("%s's %s has wrong permissions"),
		  username, filename);
	  return NULL;
	}
    }
  return fopen (filename, "r");
}

void
run_user_action (FILE *tty, const char *cr, mu_message_t msg)
{
  FILE *fp;
  int nact = 0;
  char *stmt = NULL;
  size_t size = 0;
  
  fp = open_rc (BIFF_RC, tty);
  if (fp)
    {
      unsigned line = 1, n;
      mu_debug_t debug;
      char *cwd = mu_getcwd ();
      char *rcname;

      mu_asprintf (&rcname, "%s/%s", cwd, BIFF_RC);
      free (cwd);
      
      mu_diag_get_debug (&debug);
      
      while ((n = act_getline (fp, &stmt, &size)))
	{
	  int argc;
	  char **argv;

	  if (mu_argcv_get (stmt, "", NULL, &argc, &argv) == 0
	      && argc
	      && argv[0][0] != '#')
	    {
	      mu_debug_set_locus (debug, rcname, line);
	      if (strcmp (argv[0], "beep") == 0)
		{
		  /* FIXME: excess arguments are ignored */
		  action_beep (tty);
		  nact++;
		}
	      else
		{
		  /* Rest of actions require keyword expansion */
		  int i;
		  int n_option = argc > 1 && strcmp (argv[1], "-n") == 0;
		  
		  for (i = 1; i < argc; i++)
		    {
		      char *oldarg = argv[i];
		      argv[i] = expand_line (argv[i], msg);
		      free (oldarg);
		      if (!argv[i])
			break;
		    }
		  
		  if (strcmp (argv[0], "echo") == 0)
		    {
		      action_echo (tty, cr, n_option, argc - 1, argv + 1);
		      nact++;
		    }
		  else if (strcmp (argv[0], "exec") == 0)
		    {
		      action_exec (tty, argc - 1, argv + 1);
		      nact++;
		    }
		  else
		    {
		      fprintf (tty, _(".biffrc:%d: unknown keyword"), line);
		      fprintf (tty, "\r\n");
		      mu_diag_output (MU_DIAG_ERROR, _("unknown keyword %s"),
				      argv[0]);
		      break;
		    }
		} 
	    }
	  mu_argcv_free (argc, argv);
	  line += n;
	}
      fclose (fp);
      mu_debug_set_locus (debug, NULL, 0);
      free (rcname);
    }

  if (nact == 0)
    echo_string (tty, cr, expand_line (default_action, msg));
}
