/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2005, 2006, 2007, 2009, 2010 Free
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

/* Functions for handling escape variables */

#include "mail.h"
#include <sys/stat.h>

static void
dump_headers (FILE *fp, compose_env_t *env)
{
  char buffer[512];
  mu_stream_t stream = NULL;
  size_t off = 0, n;
  
  mu_header_get_stream (env->header, &stream);
  while (mu_stream_read (stream, buffer, sizeof buffer - 1, off, &n) == 0
	 && n != 0)
    {
      buffer[n] = 0;
      fprintf (fp, "%s", buffer);
      off += n;
    }
}

#define STATE_INIT 0
#define STATE_READ 1
#define STATE_BODY 2

static int
parse_headers (FILE *fp, compose_env_t *env)
{
  int status;
  mu_header_t header;
  char *name = NULL;
  char *value = NULL;
  char *buf = NULL;
  int state = STATE_INIT;
  size_t n = 0;
  int errcnt = 0, line = 0;
  
  if ((status = mu_header_create (&header, NULL, 0, NULL)) != 0)
    {
      util_error (_("Cannot create header: %s"), mu_strerror (status));
      return 1;
    }

  while (state != STATE_BODY
	 && errcnt == 0 && getline (&buf, &n, fp) > 0 && n > 0)
    {
      int len = strlen (buf);
      if (len > 0 && buf[len-1] == '\n')
	buf[len-1] = 0;

      line++;
      switch (state)
	{
	case STATE_INIT:
	  if (!buf[0] || mu_isspace (buf[0]))
	    continue;
	  else
	    state = STATE_READ;
	  /*FALLTHRU*/
	  
	case STATE_READ:
	  if (buf[0] == 0)
	    state = STATE_BODY;
	  else if (mu_isspace (buf[0]))
	    {
	      /* A continuation line */
	      if (name)
		{
		  char *p = NULL;
		  asprintf (&p, "%s\n%s", value, buf);
		  free (value);
		  value = p;
		}
	      else
		{
		  util_error (_("%d: not a header line"), line);
		  errcnt++;
		}
	    }
	  else
	    {
	      char *p;
	      
	      if (name)
		{
		  mu_header_set_value (header, name, value[0] ? value : NULL, 0);
		  free (name);
		  free (value);
		  name = value = NULL;
		}
	      p = strchr (buf, ':');
	      if (p)
		{
		  *p++ = 0;
		  while (*p && mu_isspace (*p))
		    p++;
		  value = strdup (p);
		  name = strdup (buf);
		}
	      else
		{
		  util_error (_("%d: not a header line"), line);
		  errcnt++;
		}
	    }
	  break;
	}
    }
  
  free (buf);
  if (name)
    {
      mu_header_set_value (header, name, value, 0);
      free (name);
      free (value);
    }     

  if (errcnt)
    {
      char *p;
      
      mu_header_destroy (&header, NULL);
      p = ml_readline (_("Edit again?"));
      if (mu_true_answer_p (p) == 1)
	return -1;
      else
	return 1;
    }

  mu_header_destroy (&env->header, NULL);
  env->header = header;
  return 0;
}

static void
escape_continue (void)
{
  fprintf (stdout, _("(continue)\n"));
}

static int 
escape_check_args (int argc, char **argv)
{
  if (argc == 1)
    {
      char *escape = "~";
      mailvar_get (&escape, "escape", mailvar_type_string, 0);
      util_error (_("%c%s requires an argument"), escape[0], argv[0]);
      return 1;
    }
  return 0;
}

/* ~![shell-command] */
int
escape_shell (int argc, char **argv, compose_env_t *env)
{
  int status;
  ofile = env->ofile;
  ++*argv;
  status = mail_execute (1, argc, argv);
  ofile = env->file;
  return status;
}

/* ~:[mail-command] */
/* ~-[mail-command] */
int
escape_command (int argc, char **argv, compose_env_t *env)
{
  const struct mail_command_entry *entry;
  int status;

  if (escape_check_args (argc, argv))
    return 1;
  if (argv[1][0] == '#')
    return 0;
  entry = mail_find_command (argv[1]);
  if (!entry)
    {
      util_error (_("Unknown command: %s"), argv[1]);
      return 1;
    }
  if (entry->flags & (EF_FLOW | EF_SEND))
    {
      util_error (_("Command not allowed in an escape sequence\n"));
      return 1;
    }

  ofile = env->ofile;
  status = (*entry->func) (argc - 1, argv + 1);
  ofile = env->file;
  return status;
}

/* ~? */
int
escape_help (int argc, char **argv, compose_env_t *env MU_ARG_UNUSED)
{
  int status;
  if (argc < 2)
    status = mail_escape_help (NULL);
  else
    while (--argc)
      status |= mail_escape_help (*++argv);
  escape_continue ();
  return status;
}

/* ~A */
/* ~a */
int
escape_sign (int argc MU_ARG_UNUSED, char **argv, compose_env_t *env MU_ARG_UNUSED)
{
  char *p;

  if (mailvar_get (&p, mu_isupper (argv[0][0]) ? "Sign" : "sign",
		   mailvar_type_string, 1) == 0)
    {
      fputs ("-- \n", ofile);
      if (mu_isupper (argv[0][0]))
	{
	  char *name = util_fullpath (p);
	  FILE *fp = fopen (name, "r");
	  char *buf = NULL;
	  size_t n = 0;

	  if (!fp)
	    {
	      util_error (_("Cannot open %s: %s"), name, mu_strerror (errno));
	      free (name);
	    }

	  fprintf (stdout, _("Reading %s\n"), name);
	  while (getline (&buf, &n, fp) > 0)
	    fprintf (ofile, "%s", buf);

	  fclose (fp);
	  free (buf);
	  free (name);
	}
      else
	fprintf (ofile, "%s", p);
      escape_continue ();
    }
  return 0;
}

/* ~b[bcc-list] */
int
escape_bcc (int argc, char **argv, compose_env_t *env)
{
  while (--argc)
    compose_header_set (env, MU_HEADER_BCC, *++argv, COMPOSE_SINGLE_LINE);
  return 0;
}

/* ~c[cc-list] */
int
escape_cc (int argc, char **argv, compose_env_t *env)
{
  while (--argc)
    compose_header_set (env, MU_HEADER_CC, *++argv, COMPOSE_SINGLE_LINE);
  return 0;
}

/* ~d */
int
escape_deadletter (int argc MU_ARG_UNUSED, char **argv MU_ARG_UNUSED,
		   compose_env_t *env MU_ARG_UNUSED)
{
  FILE *dead = fopen (getenv ("DEAD"), "r");
  int c;

  if (dead)
    {
      while ((c = fgetc (dead)) != EOF)
	fputc (c, ofile);
      fclose (dead);
    }
  return 0;
}

static int
run_editor (char *ed, char *arg)
{
  char *argv[3];

  argv[0] = ed;
  argv[1] = arg;
  argv[2] = NULL;
  return mail_execute (1, 2, argv);
}

static int
escape_run_editor (char *ed, int argc, char **argv, compose_env_t *env)
{
  if (!mailvar_get (NULL, "editheaders", mailvar_type_boolean, 0))
    {
      char *filename;
      int fd = mu_tempfile (NULL, &filename);
      FILE *fp = fdopen (fd, "w+");
      char buffer[512];
      int rc;
      
      dump_headers (fp, env);

      rewind (env->file);
      while (fgets (buffer, sizeof (buffer), env->file))
	fputs (buffer, fp);

      fclose (env->file);
      
      do
	{
	  fclose (fp);
	  run_editor (ed, filename);
	  fp = fopen (filename, "r");
	}
      while ((rc = parse_headers (fp, env)) < 0);

      if (rc == 0)
	{
	  env->file = fopen (env->filename, "w");
	  while (fgets (buffer, sizeof (buffer), fp))
	    fputs (buffer, env->file);

	  fclose (env->file);
	}
      fclose (fp);
      unlink (filename);
      free (filename);
    }
  else
    {
      fclose (env->file);
      ofile = env->ofile;
      run_editor (ed, env->filename);
    }

  env->file = fopen (env->filename, "a+");
  ofile = env->file;
      
  escape_continue ();
  return 0;
}

/* ~e */
int
escape_editor (int argc, char **argv, compose_env_t *env)
{
  return escape_run_editor (getenv ("EDITOR"), argc, argv, env);
}

/* ~v */
int
escape_visual (int argc, char **argv, compose_env_t *env)
{
  return escape_run_editor (getenv ("VISUAL"), argc, argv, env);
}

/* ~f[mesg-list] */
/* ~F[mesg-list] */
int
escape_print (int argc, char **argv, compose_env_t *env MU_ARG_UNUSED)
{
  return mail_print (argc, argv);
}

void
reread_header (compose_env_t *env, char *hdr, char *prompt)
{
  char *p;
  p = strdup (compose_header_get (env, hdr, ""));
  ml_reread (prompt, &p);
  compose_header_set (env, hdr, p, COMPOSE_REPLACE);
  free (p);
}

/* ~h */
int
escape_headers (int argc, char **argv, compose_env_t *env)
{
  reread_header (env, MU_HEADER_TO, "To: ");
  reread_header (env, MU_HEADER_CC, "Cc: ");
  reread_header (env, MU_HEADER_BCC, "Bcc: ");
  reread_header (env, MU_HEADER_SUBJECT, "Subject: ");
  escape_continue ();
  return 0;
}

/* ~i[var-name] */
int
escape_insert (int argc, char **argv, compose_env_t *send_env MU_ARG_UNUSED)
{
  if (escape_check_args (argc, argv))
    return 1;
  mailvar_variable_format (ofile, mailvar_find_variable (argv[1], 0), NULL);
  return 0;
}

/* ~m[mesg-list] */
/* ~M[mesg-list] */

int
quote0 (msgset_t *mspec, mu_message_t mesg, void *data)
{
  mu_header_t hdr;
  mu_body_t body;
  mu_stream_t stream;
  char buffer[512];
  off_t off = 0;
  size_t n = 0;
  char *prefix = "\t";
  
  fprintf (stdout, _("Interpolating: %lu\n"),
	   (unsigned long) mspec->msg_part[0]);

  mailvar_get (&prefix, "indentprefix", mailvar_type_string, 0);

  if (*(int*)data)
    {
      size_t i, num = 0;
      const char *sptr;

      mu_message_get_header (mesg, &hdr);
      mu_header_get_field_count (hdr, &num);

      for (i = 1; i <= num; i++)
	{
	  mu_header_sget_field_name (hdr, i, &sptr);
	  if (mail_header_is_visible (sptr))
	    {
	      char *value;
	      
	      fprintf (ofile, "%s%s: ", prefix, sptr);
	      if (mu_header_aget_value (hdr, sptr, &value) == 0)
		{
		  int i;
		  char *p, *s;

		  for (i = 0, p = strtok_r (value, "\n", &s); p;
		       p = strtok_r (NULL, "\n", &s), i++)
		    {
		      if (i)
			fprintf (ofile, "%s", prefix);
		      fprintf (ofile, "%s\n", p);
		    }
		  free (value);
		}
	    }
	}
      fprintf (ofile, "%s\n", prefix);
      mu_message_get_body (mesg, &body);
      mu_body_get_stream (body, &stream);
    }
  else
    mu_message_get_stream (mesg, &stream);

  while (mu_stream_readline (stream, buffer, sizeof buffer - 1, off, &n) == 0
	 && n != 0)
    {
      buffer[n] = '\0';
      fprintf (ofile, "%s%s", prefix, buffer);
      off += n;
    }
  return 0;
}

int
escape_quote (int argc, char **argv, compose_env_t *env)
{
  int lower = mu_islower (argv[0][0]);
  util_foreach_msg (argc, argv, MSG_NODELETED|MSG_SILENT, quote0, &lower);
  escape_continue ();
  return 0;
}

/* ~p */
int
escape_type_input (int argc, char **argv, compose_env_t *env)
{
  char buffer[512];

  fprintf (env->ofile, _("Message contains:\n"));

  dump_headers (env->ofile, env);

  rewind (env->file);
  while (fgets (buffer, sizeof (buffer), env->file))
    fputs (buffer, env->ofile);

  escape_continue ();

  return 0;
}

/* ~r[filename] */
int
escape_read (int argc, char **argv, compose_env_t *env MU_ARG_UNUSED)
{
  char *filename;
  FILE *inf;
  size_t size, lines;
  char buf[512];

  if (escape_check_args (argc, argv))
    return 1;
  filename = util_fullpath (argv[1]);
  inf = fopen (filename, "r");
  if (!inf)
    {
      util_error (_("Cannot open %s: %s"), filename, mu_strerror (errno));
      free (filename);
      return 1;
    }

  size = lines = 0;
  while (fgets (buf, sizeof (buf), inf))
    {
      lines++;
      size += strlen (buf);
      fputs (buf, ofile);
    }
  fclose (inf);
  fprintf (stdout, "\"%s\" %lu/%lu\n", filename,
	   (unsigned long) lines, (unsigned long) size);
  free (filename);
  return 0;
}

/* ~s[string] */
int
escape_subj (int argc, char **argv, compose_env_t *env)
{
  if (escape_check_args (argc, argv))
    return 1;
  compose_header_set (env, MU_HEADER_SUBJECT, argv[1], COMPOSE_REPLACE);
  return 0;
}

/* ~t[name-list] */
int
escape_to (int argc, char **argv, compose_env_t *env)
{
  while (--argc)
    compose_header_set (env, MU_HEADER_TO, *++argv, COMPOSE_SINGLE_LINE);
  return 0;
}

/* ~w[filename] */
int
escape_write (int argc, char **argv, compose_env_t *env)
{
  char *filename;
  FILE *fp;
  size_t size, lines;
  char buf[512];

  if (escape_check_args (argc, argv))
    return 1;

  filename = util_fullpath (argv[1]);
  fp = fopen (filename, "w");	/*FIXME: check for the existence first */

  if (!fp)
    {
      util_error (_("Cannot open %s: %s"), filename, mu_strerror (errno));
      free (filename);
      return 1;
    }

  rewind (env->file);
  size = lines = 0;
  while (fgets (buf, sizeof (buf), env->file))
    {
      lines++;
      size += strlen (buf);
      fputs (buf, fp);
    }
  fclose (fp);
  fprintf (stdout, "\"%s\" %lu/%lu\n", filename,
	   (unsigned long) lines, (unsigned long) size);
  free (filename);
  return 0;
}

/* ~|[shell-command] */
int
escape_pipe (int argc, char **argv, compose_env_t *env)
{
  int p[2];
  pid_t pid;
  int fd;

  if (argc == 1)
    {
      /* TRANSLATORS: 'pipe' is a command name. Do not translate it! */
      util_error (_("pipe: no command specified"));
      return 1;
    }

  if (pipe (p))
    {
      util_error ("pipe: %s", mu_strerror (errno));
      return 1;
    }

  fd = mu_tempfile (NULL, NULL);
  if (fd == -1)
    return 1;

  if ((pid = fork ()) < 0)
    {
      close (p[0]);
      close (p[1]);
      close (fd);
      util_error ("fork: %s", mu_strerror (errno));
      return 1;
    }
  else if (pid == 0)
    {
      /* Child */
      int i;
      char **xargv;

      /* Attache the pipes */
      close (0);
      dup (p[0]);
      close (p[0]);
      close (p[1]);

      close (1);
      dup (fd);
      close (fd);

      /* Execute the process */
      xargv = xcalloc (argc, sizeof (xargv[0]));
      for (i = 0; i < argc - 1; i++)
	xargv[i] = argv[i + 1];
      xargv[i] = NULL;
      execvp (xargv[0], xargv);
      util_error (_("Cannot exec process `%s': %s"), xargv[0], mu_strerror (errno));
      exit (1);
    }
  else
    {
      FILE *fp;
      char *buf = NULL;
      size_t n;
      size_t lines, size;
      int rc = 1;
      int status;

      close (p[0]);

      /* Parent */
      fp = fdopen (p[1], "w");

      fclose (env->file);
      env->file = fopen (env->filename, "r");

      lines = size = 0;
      while (getline (&buf, &n, env->file) > 0)
	{
	  lines++;
	  size += n;
	  fputs (buf, fp);
	}
      fclose (env->file);
      fclose (fp);		/* Closes p[1] */

      waitpid (pid, &status, 0);
      if (!WIFEXITED (status))
	{
	  util_error (_("Child terminated abnormally: %d"), WEXITSTATUS (status));
	}
      else
	{
	  struct stat st;
	  if (fstat (fd, &st))
	    {
	      util_error (_("Cannot stat output file: %s"), mu_strerror (errno));
	    }
	  else if (st.st_size > 0)
	    rc = 0;
	}

      fprintf (stdout, "\"|%s\" in: %lu/%lu ", argv[1],
	       (unsigned long) lines, (unsigned long) size);
      if (rc)
	{
	  fprintf (stdout, _("no lines out\n"));
	}
      else
	{
	  /* Ok, replace the old tempfile */
	  fp = fdopen (fd, "r");
	  rewind (fp);

	  env->file = fopen (env->filename, "w+");

	  lines = size = 0;
	  while (getline (&buf, &n, fp) > 0)
	    {
	      lines++;
	      size += n;
	      fputs (buf, env->file);
	    }
	  fclose (env->file);

	  fprintf (stdout, "out: %lu/%lu\n",
		   (unsigned long) lines, (unsigned long) size);
	}

      /* Clean up the things */
      if (buf)
	free (buf);

      env->file = fopen (env->filename, "a+");
      ofile = env->file;
    }

  close (fd);

  return 0;
}
