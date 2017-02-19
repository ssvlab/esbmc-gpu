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

#include "mail.h"
#include <sys/stat.h>
#include <dirent.h>
#include <mailutils/folder.h>

#ifdef WITH_READLINE
static char **ml_command_completion (char *cmd, int start, int end);
static char *ml_command_generator (const char *text, int state);
#endif

static volatile int _interrupted;

static RETSIGTYPE
sig_handler (int signo)
{
  switch (signo)
    {
    case SIGINT:
      if (mailvar_get (NULL, "quit", mailvar_type_boolean, 0) == 0)
	exit (0);
      _interrupted++;
      break;
#if defined (SIGWINCH)
    case SIGWINCH:
      util_do_command ("set screen=%d", util_getlines());
      util_do_command ("set columns=%d", util_getcols());
      page_invalidate (1);
      break;
#endif
    }
#ifndef HAVE_SIGACTION
  signal (signo, sig_handler);
#endif
}

void
ml_clear_interrupt ()
{
  _interrupted = 0;
}

int
ml_got_interrupt ()
{
  int rc = _interrupted;
  _interrupted = 0;
  return rc;
}

static int
ml_getc (FILE *stream)
{
  unsigned char c;

  while (1)
    {
      if (read (fileno (stream), &c, 1) == 1)
	return c;
      if (errno == EINTR)
	{
	  if (_interrupted)
	    break;
	  /* keep going if we handled the signal */
	}
      else
	break;
    }
  return EOF;
}

void
ml_readline_init ()
{
  if (!interactive)
    return;

#ifdef WITH_READLINE
  rl_readline_name = "mail";
  rl_attempted_completion_function = (CPPFunction*)ml_command_completion;
  rl_getc_function = ml_getc;
#endif
#ifdef HAVE_SIGACTION
  {
    struct sigaction act;
    act.sa_handler = sig_handler;
    sigemptyset (&act.sa_mask);
    act.sa_flags = 0;
    sigaction (SIGINT, &act, NULL);
#if defined(SIGWINCH)
    sigaction (SIGWINCH, &act, NULL);
#endif
  }
#else
  signal (SIGINT, sig_handler);
#if defined(SIGWINCH)
  signal (SIGWINCH, sig_handler);
#endif
#endif
}

char *
ml_readline_internal ()
{
  char *line;
  char *p;
  size_t alloclen, linelen;

  p = line = xcalloc (1, 255);
  alloclen = 255;
  linelen = 0;
  for (;;)
    {
      size_t n;

      p = fgets (p, alloclen - linelen, stdin);

      if (p)
	n = strlen(p);
      else if (_interrupted)
	{
	  free (line);
	  return NULL;
	}
      else
	n = 0;

      linelen += n;

      /* Error.  */
      if (linelen == 0)
	{
	  free (line);
	  return NULL;
	}

      /* Ok.  */
      if (line[linelen - 1] == '\n')
	{
	  line[linelen - 1] = '\0';
	  return line;
	}
      else
        {
	  char *tmp;
	  alloclen *= 2;
	  tmp = realloc (line, alloclen);
	  if (tmp == NULL)
	    {
	      free (line);
	      return NULL;
	    }
	  line = tmp;
	  p = line + linelen;
	}
    }
}

char *
ml_readline (const char *prompt)
{
  if (interactive)
    return readline (prompt);
  return ml_readline_internal ();
}

char *
ml_readline_with_intr (const char *prompt)
{
  char *str = ml_readline (prompt);
  if (_interrupted)
    printf ("\n");
  return str;
}

#ifdef WITH_READLINE

static char *insert_text;

static int
ml_insert_hook (void)
{
  if (insert_text)
    rl_insert_text (insert_text);
  return 0;
}

int
ml_reread (const char *prompt, char **text)
{
  char *s;

  ml_clear_interrupt ();
  insert_text = *text;
  rl_startup_hook = ml_insert_hook;
  s = readline ((char *)prompt);
  if (!ml_got_interrupt ())
    {
      if (*text)
	free (*text);
      *text = s;
    }
  else
    {
      putc('\n', stdout);
    }
  rl_startup_hook = NULL;
  return 0;
}

/*
 * readline tab completion
 */
char **
ml_command_completion (char *cmd, int start, int end)
{
  int argc;
  char **argv;
  char **ret;
  char *p;

  for (p = rl_line_buffer; p < rl_line_buffer + start && mu_isblank (*p); p++)
    ;
  
  if (mu_argcv_get_n (p, end, NULL, NULL, &argc, &argv))
    return NULL;
  rl_completion_append_character = ' ';
  
  if (argc == 0 || (argc == 1 && strlen (argv[0]) <= end - start))
    {
      ret = rl_completion_matches (cmd, ml_command_generator);
      rl_attempted_completion_over = 1;
    }
  else
    {
      const struct mail_command_entry *entry = mail_find_command (argv[0]);
      if (entry && entry->command_completion)
	ret = entry->command_completion (argc, argv, start == end);
      else
	ret = NULL;
    }
  mu_argcv_free (argc, argv);
  return ret;
}

/*
 * more readline
 */
char *
ml_command_generator (const char *text, int state)
{
  static int i, len;
  const char *name;
  const struct mail_command *cp;
  
  if (!state)
    {
      i = 0;
      len = strlen (text);
    }

  while ((cp = mail_command_name (i)))
    {
      name = cp->longname;
      if (strlen (cp->shortname) > strlen (name))
	name = cp->shortname;
      i++;
      if (strncmp (name, text, len) == 0)
	return strdup (name);
    }

  return NULL;
}

void
ml_set_completion_append_character (int c)
{
  rl_completion_append_character = c;
}

void
ml_attempted_completion_over ()
{
  rl_attempted_completion_over = 1;
}


/* Completion functions */
char **
no_compl (int argc MU_ARG_UNUSED, char **argv MU_ARG_UNUSED, int ws MU_ARG_UNUSED)
{
  ml_attempted_completion_over ();
  return NULL;
}

char **
msglist_compl (int argc, char **argv, int ws)
{
  /* FIXME */
  ml_attempted_completion_over ();
  return NULL;
}

char **
msglist_file_compl (int argc, char **argv, int ws MU_ARG_UNUSED)
{
  if (argc == 1)
    ml_attempted_completion_over ();
  return NULL;
}

char **
command_compl (int argc, char **argv, int ws)
{
  ml_set_completion_append_character (0);
  if (ws)
    return NULL;
  return rl_completion_matches (argv[argc-1], ml_command_generator);
}

/* Generate file list based on reference prefix TEXT, relative to PATH.
   Remove PATHLEN leading characters from the returned names. Replace
   them with REPL unless it is 0.

   Select only those files that match given FLAGS (MU_FOLDER_ATTRIBUTE_*
   constants).
   
   STATE is 0 for the first call, 1 otherwise.
 */
static char *
file_generator (const char *text, int state,
		char *path, size_t pathlen,
		char repl, 
		int flags)
{
  static mu_list_t list;
  static mu_iterator_t itr;
  
  if (!state)
    {
      char *wcard;
      mu_folder_t folder;
      size_t count;
      
      wcard = xmalloc (strlen (text) + 2);
      strcat (strcpy (wcard, text), "*");

      if (mu_folder_create (&folder, path))
	{
	  free (wcard);
	  return NULL;
	}
      
      mu_folder_list (folder, path, wcard, 1, &list);
      free (wcard);
      mu_folder_destroy (&folder);

      if (mu_list_count (list, &count) || count == 0)
	{
	  mu_list_destroy (&list);
	  return NULL;
	}
      else if (count == 1)
	ml_set_completion_append_character (0);
      
      if (mu_list_get_iterator (list, &itr))
	{
	  mu_list_destroy (&list);
	  return NULL;
	}
      mu_iterator_first (itr);
    }

  while (!mu_iterator_is_done (itr))
    {
      struct mu_list_response *resp;
      mu_iterator_current (itr, (void**)&resp);
      mu_iterator_next (itr);
      if (resp->type & flags)
	{
	  char *ret;
	  if (pathlen)
	    {
	      size_t len = strlen (resp->name + pathlen);
	      char *ptr;
	      
	      ret = xmalloc (len + (repl ? 1 : 0) + 1);
	      ptr = ret;
	      if (repl)
		*ptr++ = repl;
	      memcpy (ptr, resp->name + pathlen, len);
	      ptr[len] = 0;
	    }
	  else
	    ret = xstrdup (resp->name);
	  return ret;
	}
    }
  mu_iterator_destroy (&itr);
  mu_list_destroy (&list);
  return NULL;
}

static char *
folder_generator (const char *text, int state)
{
  char *ret;
  static size_t pathlen;
  
  if (!state)
    {
      char *path = util_folder_path ("");
      if (!path)
	return NULL;
      
      pathlen = strlen (path);
      ret = file_generator (text, state, path, pathlen, '+',
			    MU_FOLDER_ATTRIBUTE_ALL);
      free (path);
    }
  else
    ret = file_generator (text, state, NULL, pathlen, '+',
			  MU_FOLDER_ATTRIBUTE_ALL);
  return ret;
}

char **
file_compl (int argc, char **argv, int ws)
{
  char *text;

  if (ws)
    {
      ml_set_completion_append_character (0);
      ml_attempted_completion_over ();
      return NULL;
    }
  
  text = argv[argc-1];
  switch (text[0])
    {
    case '+':
      text++;
      break;

    case '#':
    case '&':
      ml_attempted_completion_over ();
      return NULL;
      
    default:
      return NULL; /* Will be expanded by readline itself */
    }
  
  return rl_completion_matches (text, folder_generator);
}

static char *
dir_generator (const char *text, int state)
{
  char *ret;
  static size_t pathlen;
  static int repl;

  if (!state)
    {
      char *path;
      switch (text[0])
	{
	case '+':
	  text++;
	  repl = '+';
	  path = util_folder_path (text);
	  pathlen = strlen (path) - strlen (text);
	  break;

	case '~':
	  repl = '~';
	  if (text[1] == '/')
	    {
	      path = mu_get_homedir ();
	      text += 2;
	      pathlen = strlen (path);
	      break;
	    }
	  /* else FIXME! */

	case '/':
	  path = strdup (text);
	  pathlen = 0;
	  repl = 0;
	  break;
	  
	default:
	  path = strdup ("./");
	  pathlen = 2;
	  repl = 0;
	}
      
      ret = file_generator (text, state, path, pathlen, repl,
			    MU_FOLDER_ATTRIBUTE_DIRECTORY);
      free (path);
    }
  else
    ret = file_generator (text, state, NULL, pathlen, repl,
			  MU_FOLDER_ATTRIBUTE_DIRECTORY);
  return ret;
}

char **
dir_compl (int argc, char **argv, int ws)
{
  ml_attempted_completion_over ();
  if (ws)
    {
      ml_set_completion_append_character (0);
      return NULL;
    }
  return rl_completion_matches (argv[argc-1], dir_generator);
}

static char *
alias_generator (const char *text, int state)
{
  static alias_iterator_t itr;
  const char *p;
  
  if (!state)
    p = alias_iterate_first (text, &itr);
  else
    p = alias_iterate_next (itr);

  if (!p)
    {
      alias_iterate_end (&itr);
      return NULL;
    }
  return strdup (p);
}

char **
alias_compl (int argc, char **argv, int ws)
{
  ml_attempted_completion_over ();
  return rl_completion_matches (ws ? "" : argv[argc-1], alias_generator);
}

static char *
mkfilename (const char *dir, const char *file)
{
  size_t len = strlen (dir) + 1 + strlen (file) + 1;
  char *p = malloc (len);
  if (p)
    {
      strcpy (p, dir);
      strcat (p, "/");
      strcat (p, file);
    }
  return p;
}

static char *
exec_generator (const char *text, int state)
{
  static int prefix_len;
  static char *var;
  static char *dir;
  static size_t dsize;
  static DIR *dp;
  struct dirent *ent;
  
  if (!state)
    {
      var = getenv ("PATH");
      if (!var)
	return NULL;
      prefix_len = strlen (text);
      dsize = 0;
      dir = NULL;
      dp = NULL;
    }

  while (1)
    {
      if (!dp)
	{
	  char *p;
	  size_t len;

	  if (*var == 0)
	    break;
	  else
	    var++;
	  
	  p = strchr (var, ':');
	  if (!p)
	    len = strlen (var) + 1;
	  else
	    len = p - var + 1;
	  
	  if (dsize == 0)
	    {
	      dir = malloc (len);
	      dsize = len;
	    }
	  else if (len > dsize)
	    {
	      dir = realloc (dir, len);
	      dsize = len;
	    }
	  
	  if (!dir)
	    return NULL;
	  memcpy (dir, var, len - 1);
	  dir[len - 1] = 0;
	  var += len - 1;
	  
	  dp = opendir (dir);
	  if (!dp)
	    continue;
	}

      while ((ent = readdir (dp)))
	{
	  char *name = mkfilename (dir, ent->d_name);
	  if (name)
	    {
	      int rc = access (name, X_OK); 
	      if (rc == 0)
		{
		  struct stat st;
		  rc = !(stat (name, &st) == 0 && S_ISREG (st.st_mode));
		}
		    
	      free (name);
	      if (rc == 0
		  && strlen (ent->d_name) >= prefix_len
		  && strncmp (ent->d_name, text, prefix_len) == 0)
		return strdup (ent->d_name);
	    }
	}

      closedir (dp);
      dp = NULL;
    }

  free (dir);
  return NULL;
}

char **
exec_compl (int argc, char **argv, int ws)
{
  return rl_completion_matches (ws ? "" : argv[argc-1], exec_generator);
}

#else

#include <sys/ioctl.h>

#define STDOUT 1
#ifndef TIOCSTI
static int ch_erase;
static int ch_kill;
#endif

#if defined(TIOCSTI)
#elif defined(HAVE_TERMIOS_H)
# include <termios.h>

static struct termios term_settings;

int
set_tty ()
{
  struct termios new_settings;

  if (tcgetattr(STDOUT, &term_settings) == -1)
    return 1;

  ch_erase = term_settings.c_cc[VERASE];
  ch_kill = term_settings.c_cc[VKILL];

  new_settings = term_settings;
  new_settings.c_lflag &= ~(ICANON|ECHO);
#if defined(TAB3)
  new_settings.c_oflag &= ~(TAB3);
#elif defined(OXTABS)
  new_settings.c_oflag &= ~(OXTABS);
#endif
  new_settings.c_cc[VMIN] = 1;
  new_settings.c_cc[VTIME] = 0;
  tcsetattr(STDOUT, TCSADRAIN, &new_settings);
  return 0;
}

void
restore_tty ()
{
  tcsetattr (STDOUT, TCSADRAIN, &term_settings);
}

#elif defined(HAVE_TERMIO_H)
# include <termio.h>

static struct termio term_settings;

int
set_tty ()
{
  struct termio new_settings;

  if (ioctl(STDOUT, TCGETA, &term_settings) < 0)
    return -1;

  ch_erase = term_settings.c_cc[VERASE];
  ch_kill = term_settings.c_cc[VKILL];

  new_settings = term_settings;
  new_settings.c_lflag &= ~(ICANON | ECHO);
  new_settings.c_oflag &= ~(TAB3);
  new_settings.c_cc[VMIN] = 1;
  new_settings.c_cc[VTIME] = 0;
  ioctl(STDOUT, TCSETA, &new_settings);
  return 0;
}

void
restore_tty ()
{
  ioctl(STDOUT, TCSETA, &term_settings);
}

#elif defined(HAVE_SGTTY_H)
# include <sgtty.h>

static struct sgttyb term_settings;

int
set_tty ()
{
  struct sgttyb new_settings;

  if (ioctl(STDOUT, TIOCGETP, &term_settings) < 0)
    return 1;

  ch_erase = term_settings.sg_erase;
  ch_kill = term_settings.sg_kill;

  new_settings = term_settings;
  new_settings.sg_flags |= CBREAK;
  new_settings.sg_flags &= ~(ECHO | XTABS);
  ioctl(STDOUT, TIOCSETP, &new_settings);
  return 0;
}

void
restore_tty ()
{
  ioctl(STDOUT, TIOCSETP, &term_settings);
}

#else
# define DUMB_MODE
#endif


#define LINE_INC 80

int
ml_reread (const char *prompt, char **text)
{
  int ch;
  char *line;
  int line_size;
  int pos;
  char *p;

  if (*text)
    {
      line = strdup (*text);
      if (line)
	{
	  pos = strlen(line);
	  line_size = pos + 1;
	}
    }
  else
    {
      line_size = LINE_INC;
      line = malloc (line_size);
      pos = 0;
    }

  if (!line)
    {
      util_error (_("Not enough memory to edit the line"));
      return -1;
    }

  line[pos] = 0;

  if (prompt)
    {
      fputs (prompt, stdout);
      fflush (stdout);
    }

#ifdef TIOCSTI

  for (p = line; *p; p++)
    {
      ioctl(0, TIOCSTI, p);
    }

  pos = 0;

  while ((ch = ml_getc (stdin)) != EOF && ch != '\n')
    {
      if (pos >= line_size)
	{
	  if ((p = realloc (line, line_size + LINE_INC)) == NULL)
	    {
	      fputs ("\n", stdout);
	      util_error (_("Not enough memory to edit the line"));
	      break;
	    }
	  else
	    {
	      line_size += LINE_INC;
	      line = p;
	    }
	}
      line[pos++] = ch;
    }

#else

  fputs (line, stdout);
  fflush (stdout);

# ifndef DUMB_MODE
  set_tty ();

  while ((ch = ml_getc (stdin)) != EOF)
    {
      if (ch == ch_erase)
	{
	  /* kill last char */
	  if (pos > 0)
	    line[--pos] = 0;
	  putc('\b', stdout);
	}
      else if (ch == ch_kill)
	{
	  /* kill the entire line */
	  pos = 0;
	  line[pos] = 0;
	  putc ('\r', stdout);
	  if (prompt)
	    fputs (prompt, stdout);
	}
      else if (ch == '\n' || ch == EOF)
	break;
      else
	{
	  if (pos >= line_size)
	    {
	      if ((p = realloc (line, line_size + LINE_INC)) == NULL)
		{
		  fputs ("\n", stdout);
		  util_error (_("Not enough memory to edit the line"));
		  break;
		}
	      else
		{
		  line_size += LINE_INC;
		  line = p;
		}
	    }
	  line[pos++] = ch;
	  putc(ch, stdout);
	}
      fflush (stdout);
    }

  putc ('\n', stdout);
  restore_tty ();
# else
  /* Dumb mode: the last resort */

  putc ('\n', stdout);
  if (prompt)
    {
      fputs (prompt, stdout);
      fflush (stdout);
    }

  pos = 0;

  while ((ch = ml_getc (stdin)) != EOF && ch != '\n')
    {
      if (pos >= line_size)
	{
	  if ((p = realloc (line, line_size + LINE_INC)) == NULL)
	    {
	      fputs ("\n", stdout);
	      util_error (_("Not enough memory to edit the line"));
	      break;
	    }
	  else
	    {
	      line_size += LINE_INC;
	      line = p;
	    }
	}
      line[pos++] = ch;
    }
# endif
#endif

  line[pos] = 0;

  if (ml_got_interrupt ())
    free (line);
  else
    {
      if (*text)
	free (*text);
      *text = line;
    }
  return 0;
}

char *
readline (char *prompt)
{
  if (prompt)
    {
      fprintf (ofile, "%s", prompt);
      fflush (ofile);
    }

  return ml_readline_internal ();
}

void
ml_set_completion_append_character (int c MU_ARG_UNUSED)
{
}

void
ml_attempted_completion_over ()
{
}

#endif

