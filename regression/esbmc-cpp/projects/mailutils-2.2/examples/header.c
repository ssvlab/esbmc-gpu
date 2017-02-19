/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2009, 2010 Free Software Foundation, Inc.

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
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <mailutils/mailutils.h>

char *file;
mu_header_t header;
mu_iterator_t iterator;

char *ps[] = { "> ", ". " };
int interactive;

static void
prompt(int l)
{
  if (interactive)
    {
      printf ("%s", ps[l]);
      fflush (stdout);
    }
}

static int
load_file (const char *name)
{
  struct stat st;
  size_t nread;
  char *buf;
  FILE *fp;
  int status;
  
  if (stat (name, &st))
    {
      mu_error ("cannot stat %s: %s", name, mu_strerror (errno));
      return 1;
    }

  buf = malloc (st.st_size + 2);
  if (!buf)
    {
      mu_error ("not enough memory");
      return 1;
    }

  fp = fopen (name, "r");
  if (!fp)
    {
      mu_error ("cannot open file %s: %s", name, mu_strerror (errno));
      free (buf);
      return 1;
    }
  
  nread = fread (buf, 1, st.st_size, fp);
  fclose (fp);
  if (nread != st.st_size)
    {
      mu_error ("short read on file %s", name);
      free (buf);
      return 1;
    }

  buf[st.st_size] = '\n';
  buf[st.st_size+1] = 0;
  status = mu_header_create (&header, buf, st.st_size + 1, NULL);
  free (buf);
  if (status)
    {
      mu_error ("cannot create header: %s", mu_strerror (status));
      return 1;
    }
  return 0;
}

unsigned line_num = 0;

static int
check_args (char const *cmdname, int argc, int amin, int amax)
{
  if (argc < amin)
    {
      mu_error ("%u: %s: too few arguments",
	       line_num, cmdname);
      return 1;
    }
  if (amax > 0 && argc > amax)
    {
      mu_error ("%u: %s: too many arguments",
		line_num, cmdname);
      return 1;
    }      
  return 0;
}

void
cmd_quit (int argc, char **argv)
{
  exit (0);
}

void
cmd_load (int argc, char **argv)
{
  if (check_args (argv[0], argc, 2, 2))
    return;
  mu_header_destroy (&header, NULL);
  load_file (argv[1]);
}

void
cmd_free (int argc, char **argv)
{
  if (check_args (argv[0], argc, 1, 1))
    return;
  mu_iterator_destroy (&iterator);
  mu_header_destroy (&header, NULL);
}

void
cmd_print (int argc, char **argv)
{
  char *fn;
  int num = 1;
  int status;
  const char *str;
  
  if (check_args (argv[0], argc, 2, 3))
    return;
  fn = argv[1];
  if (argc == 3)
    num = atoi (argv[2]);

  status = mu_header_sget_value_n (header, fn, num, &str);
  if (status == 0)
    printf ("%s: %s\n", fn, str);
  else
    mu_error ("%u: %s", line_num, mu_strerror (status));
}

void
cmd_dump (int argc, char **argv)
{
  mu_off_t off = 0;
  size_t n;
  mu_stream_t stream;
  char buf[512];
  int status;
  
  if (check_args (argv[0], argc, 1, 2))
    return;

  if (argc == 2)
    off = strtoul (argv[1], NULL, 0);

  status = mu_header_get_stream (header, &stream);
  if (status)
    {
      mu_error ("%u: cannot get stream: %s", line_num, mu_strerror (status));
      return;
    }

  status = mu_stream_seek (stream, off, SEEK_SET);
  if (status)
    {
      mu_error ("%u: cannot seek: %s", line_num, mu_strerror (status));
      return;
    }

  while (mu_stream_sequential_read (stream, buf, sizeof buf, &n) == 0
	 && n > 0)
    {
      fwrite (buf, 1, n, stdout);
    }
}  

void
cmd_remove (int argc, char **argv)
{
  char *fn;
  int num = 1;
  int status;
  
  if (check_args (argv[0], argc, 2, 3))
    return;
  fn = argv[1];
  if (argc == 3)
    num = atoi (argv[2]);
  status = mu_header_remove (header, fn, num);
  if (status)
    mu_error ("%u: %s: %s", line_num, argv[0], mu_strerror (status));
}

/* insert header value [ref [num] [before|after] [replace]] */
void
cmd_insert (int argc, char **argv)
{
  int status;
  int flags = 0;
  char *ref = NULL;
  int num = 1;
  int n;
  
  if (check_args (argv[0], argc, 3, 7))
    return;

  if (argc >= 4)
    {
      ref = argv[3];
      n = 4;
      if (n < argc)
	{
	  char *p;
	  int tmp;
	  
	  tmp = strtoul(argv[4], &p, 0);
	  if (*p == 0)
	    {
	      num = tmp;
	      n++;
	    }

	  for (; n < argc; n++)
	    {
	      if (strcmp(argv[n], "before") == 0)
		flags |= MU_HEADER_BEFORE;
	      else if (strcmp(argv[n], "after") == 0)
		;
	      else if (strcmp(argv[n], "replace") == 0)
		flags |= MU_HEADER_REPLACE;
	      else
		{
		  mu_error("%u: %s: unknown option", line_num, argv[4]);
		  return;
		}
	    }
	}
    }
  status = mu_header_insert (header, argv[1], argv[2],
			     ref, num, flags);
  if (status)
    mu_error ("%u: %s: %s", line_num, argv[0], mu_strerror (status));
}

void
cmd_write (int argc, char **argv)
{
  char buf[512];
  mu_stream_t str;
  int status;
  
  if (check_args (argv[0], argc, 1, 1))
    return;

  status = mu_header_get_stream (header, &str);
  if (status)
    {
      mu_error ("%u: cannot get stream: %s", line_num, mu_strerror (status));
      return;
    }
  printf("[reading headers; end with an empty line]\n");
  mu_stream_seek (str, 0, SEEK_SET);
  while (prompt (1), fgets(buf, sizeof buf, stdin))
    {
      mu_stream_sequential_write (str, buf, strlen (buf));
      if (buf[0] == '\n')
	break;
    }
}

void
cmd_iterate (int argc, char **argv)
{
  if (check_args (argv[0], argc, 1, 2))
    return;
  if (argc == 1)
    {
      mu_iterator_t itr;
      MU_ASSERT (mu_header_get_iterator (header, &itr));
      for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
	   mu_iterator_next (itr))
	{
	  const char *hdr, *val;
	  MU_ASSERT (mu_iterator_current_kv (itr,
					     (const void**)&hdr,
					     (void**)&val));
	  printf ("%s: %s\n", hdr, val);
	}
      mu_iterator_destroy (&itr);
    }
  else
    {
      const char *hdr, *val;

      if (!iterator)
	MU_ASSERT (mu_header_get_iterator (header, &iterator));

      if (strcmp (argv[1], "first") == 0 || strcmp (argv[1], "1") == 0)
	mu_iterator_first (iterator);
      else if (strcmp (argv[1], "next") == 0 || strcmp (argv[1], "n") == 0)
	{
	  mu_iterator_next (iterator);
	  if (mu_iterator_is_done (iterator))
	    {
	      printf ("Past end of headers. Use `itr first'.\n");
	      return;
	    }
	}

      MU_ASSERT (mu_iterator_current_kv (iterator,
					 (const void **)&hdr,
					 (void**)&val));
      printf ("%s: %s\n", hdr, val);
    }
}

void
cmd_readline (int argc, char **argv)
{
  char *buf;
  size_t size;
  mu_stream_t stream;
  size_t nbytes;
  
  if (check_args (argv[0], argc, 1, 2))
    return;
  size = atoi (argv[1]);
  buf = malloc (size);
  if (!buf)
    abort ();
  mu_header_get_stream (header, &stream);
  mu_stream_readline (stream, buf, size, 0, &nbytes);
  printf ("\"%*.*s\"", (int) nbytes, (int) nbytes, buf);
  free (buf);
}
  

struct cmdtab
{
  char *name;
  void (*fun) (int argc, char **argv);
  char *args;
  char *help;
};

static void cmd_help (int argc, char **argv);

static struct cmdtab cmdtab[] = {
  { "quit", cmd_quit, NULL, "quit the program" },
  { "load", cmd_load, "FILE", "read headers from the specified FILE" },
  { "free", cmd_free, NULL, "discard all headers" },
  { "print", cmd_print, "NAME [N]",
    "find and print the Nth (by default, 1st) instance of header named NAME" },
  { "dump", cmd_dump, NULL, "dump all headers on screen" },
  { "itr", cmd_iterate, "[first|1|next|n]", "iterate over headers" },
  { "readline", cmd_readline, "[SIZE]", "read line" },
  { "remove", cmd_remove, "NAME [N]",
    "remove the Nth (by default, 1st) instance of header named NAME" },
  { "insert", cmd_insert, "NAME VALUE [REF [NUM] [before|after] [replace]]",
    "insert new header" },
  { "write", cmd_write, NULL, "accept headers from raw stream" },
  { "help", cmd_help, "[COMMAND]", "print short usage message" },
  { NULL }
};

static struct cmdtab *
find_cmd (const char *name)
{
  struct cmdtab *p;
  for (p = cmdtab; p->name; p++)
    if (strcmp (p->name, name) == 0)
      return p;
  return NULL;
}

static void
format_help_str (int col, char *p)
{
  if (col > 31)
    col = 80;
  while (*p)
    {
      int len;
      char *q;

      if (*p == ' ' || *p == '\t')
	{
	  p++;
	  continue;
	}
      
      q = strchr (p, ' ');
      if (!q)
	len = strlen (p);
      else
	len = q - p;
      
      if (col + len > 80)
	{
	  fputc ('\n', stdout);
	  for (col = 0; col < 30; col++)
	    fputc (' ', stdout);
	}
      for (; len > 0; len--, p++, col++)
	fputc (*p, stdout);

      if (q)
	{
	  if (col < 80)
	    {
	      fputc (' ', stdout);
	      col++;
	    }
	  p++;
	}
    }
  fputc ('\n', stdout);
}
	  
      

void
cmd_help (int argc, char **argv)
{
  struct cmdtab *p;
  
  if (check_args (argv[0], argc, 1, 2))
    return;

  for (p = cmdtab; p->name; p++)
    {
      int col;
      
      col = printf ("%s", p->name);
      for (; col < 10; col++)
	fputc (' ', stdout);
      if (p->args)
	col += printf ("%s", p->args);
      for (; col < 30; col++)
	fputc (' ', stdout);
      format_help_str (col, p->help);
    }
}

int
docmd (int argc, char **argv)
{
  struct cmdtab *cmd = find_cmd (argv[0]);
  if (!cmd)
    {
      mu_error ("%u: unknown command %s", line_num, argv[0]);
      return 1;
    }
  else
    cmd->fun (argc, argv);
  return 0;
}

int
main (int argc, char **argv)
{
  int c;
  char buf[512];
  char **prevv;
  int prevc = 0;
  
  interactive = isatty (0);
  while ((c = getopt (argc, argv, "f:h")) != EOF)
    {
      switch (c)
	{
	case 'f':
	  file = optarg;
	  break;

	case 'h':
	  printf ("usage: header [-f file]\n");
	  exit (0);

	default:
	  exit (1);
	}
    }

  if (file)
    {
      if (load_file (file))
	exit (1);
    }
  else
    {
      int status = mu_header_create (&header, NULL, 0, NULL);
      if (status)
	{
	  mu_error ("cannot create header: %s", mu_strerror (status));
	  exit (1);
	}
    }
  
  while (prompt(0), fgets(buf, sizeof buf, stdin))
    {
      int c;
      char **v;
      int status;

      line_num++;
      status = mu_argcv_get (buf, NULL, "#", &c, &v);
      if (status)
	{
	  mu_error ("%u: cannot parse: %s",
		   line_num, mu_strerror (status));
	  continue;
	}

      if (c == 0)
	{
	  if (prevc)
	    docmd (prevc, prevv);
	  else
	    mu_argcv_free (c, v);
	}
      else
	{
	  docmd (c, v);
	  mu_argcv_free (prevc, prevv);
	  prevc = c;
	  prevv = v;
	}
    }
  exit (0);
}

