/* nntpclient.c -- An application which demonstrates how to use the
   GNU Mailutils nntp functions.  This application interactively allows users
   to contact a nntp server.

   Copyright (C) 2003, 2004, 2005, 2007, 2009, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>  
#endif 
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <termios.h>
#include <signal.h>
#include <time.h>

#ifdef WITH_READLINE
# include <readline/readline.h>
# include <readline/history.h>
#endif

#include <mailutils/nntp.h>
#include <mailutils/iterator.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/mutil.h>
#include <mailutils/cctype.h>
#include <mailutils/cstr.h>

/* A structure which contains information on the commands this program
   can understand. */

typedef struct
{
  const char *name;		/* User printable name of the function. */
  int (*func) (char *);		/* Function to call to do the job. */
  const char *doc;		/* Documentation for this function.  */
}
COMMAND;

/* The names of functions that actually do the manipulation. */
int com_article (char *);
int com_body (char *);
int com_connect (char *);
int com_date (char *);
int com_disconnect (char *);
int com_exit (char *);
int com_group (char *);
int com_head (char *);
int com_help (char *);
int com_ihave (char *);
int com_info (char *);
int com_last (char *);
int com_list (char *);
int com_list_active (char *);
int com_list_active_times (char *);
int com_list_distributions (char *);
int com_list_distrib_pats (char *);
int com_list_extensions (char *);
int com_list_newsgroups (char *);
int com_mode (char *);
int com_mode_reader (char *);
int com_newgroups (char *);
int com_newnews (char *);
int com_next (char *);
int com_post (char *);
int com_quit (char *);
int com_stat (char *);
int com_verbose (char *);

void initialize_readline (void);
COMMAND *find_command (char *);
char *dupstr (const char *);
int execute_line (char *);
int valid_argument (const char *, char *);

void sig_int (int);

COMMAND commands[] = {
  {"article", com_article, "Retrieve an article: ARTICLE [message_id|number]"},
  {"body", com_body, "Retrieve the body of an article: BODY [message_id|number]"},
  {"connect", com_connect, "Open connection: connect hostname [port]"},
  {"date", com_date, "Server date: DATE"},
  {"disconnect", com_disconnect, "Close connection: disconnect"},
  {"exit", com_exit, "exit program"},
  {"group", com_group, "Select a group: GROUP group"},
  {"head", com_head, "Retrieve the head of an article: HEAD [message_id|number]"},
  {"help", com_help, "Ask the server for info: HELP"},
  {"ihave", com_ihave, "Transfer an article to the server : IHAVE <<enter the article, finish with a '.' or ctrl-D>>"},
  {"last", com_last, "Set current to the previous article: LAST"},
  {"list", com_list, "List : LIST [ACTIVE|ACTIVE.TIMES|DISTRIB.PATS|DISTRIBUTIONS|EXTENSIONS|NEWSGROUPS]"},
  {"mode", com_mode, "Set mode reader: MODE READER"},
  {"newgroups", com_newgroups, "Ask for new groups : NEWGROUPS [yyyymmdd hhmmss [GMT]]"},
  {"newnews", com_newnews, "Ask for new news : NEWNEWS wildmat [yyyymmdd hhmmss]"},
  {"next", com_next, "Set current to the next article: NEXT"},
  {"post", com_post, "Post an article to the server : POST <<enter the article, finish with a '.' or ctrl-D>>"},
  {"quit", com_quit, "Terminate the session: QUIT"},
  {"stat", com_stat, "Check the status of an article : STAT [message_id|number]"},
  {"verbose", com_verbose, "Enable Protocol tracing: verbose {on|off}"},
  {"?", com_info, "Dysplay this help"},
  {NULL, NULL, NULL}
};

/* The name of this program, as taken from argv[0]. */
char *progname;

/* Global handle for nntp.  */
mu_nntp_t nntp;

/* Flag if verbosity is needed.  */
int verbose;

/* When non-zero, this global means the user is done using this program. */
int done;

char *
dupstr (const char *s)
{
  char *r;

  r = malloc (strlen (s) + 1);
  if (!r)
    {
      fprintf (stderr, "Memory exhausted\n");
      exit (1);
    }
  strcpy (r, s);
  return r;
}


#ifdef WITH_READLINE
/* Interface to Readline Completion */

char *command_generator (const char *, int);
char **nntp_completion (char *, int, int);

/* Tell the GNU Readline library how to complete.  We want to try to complete
   on command names if this is the first word in the line, or on filenames
   if not. */
void
initialize_readline ()
{
  /* Allow conditional parsing of the ~/.inputrc file. */
  rl_readline_name = (char *) "nntp";

  /* Tell the completer that we want a crack first. */
  rl_attempted_completion_function = (CPPFunction *) nntp_completion;
}

/* Attempt to complete on the contents of TEXT.  START and END bound the
   region of rl_line_buffer that contains the word to complete.  TEXT is
   the word to complete.  We can use the entire contents of rl_line_buffer
   in case we want to do some simple parsing.  Return the array of matches,
   or NULL if there aren't any. */
char **
nntp_completion (char *text, int start, int end)
{
  char **matches;

  (void) end;
  matches = (char **) NULL;

  /* If this word is at the start of the line, then it is a command
     to complete.  Otherwise it is the name of a file in the current
     directory. */
  if (start == 0)
    matches = rl_completion_matches (text, command_generator);

  return (matches);
}

/* Generator function for command completion.  STATE lets us know whether
   to start from scratch; without any state (i.e. STATE == 0), then we
   start at the top of the list. */
char *
command_generator (const char *text, int state)
{
  static int list_index, len;
  const char *name;

  /* If this is a new word to complete, initialize now.  This includes
     saving the length of TEXT for efficiency, and initializing the index
     variable to 0. */
  if (!state)
    {
      list_index = 0;
      len = strlen (text);
    }

  /* Return the next name which partially matches from the command list. */
  while ((name = commands[list_index].name))
    {
      list_index++;

      if (strncmp (name, text, len) == 0)
	return (dupstr (name));
    }

  /* If no names matched, then return NULL. */
  return ((char *) NULL);
}

#else
void
initialize_readline ()
{
}

char *
readline (char *prompt)
{
  char buf[255];

  if (prompt)
    {
      printf ("%s", prompt);
      fflush (stdout);
    }

  if (!fgets (buf, sizeof (buf), stdin))
    return NULL;
  return strdup (buf);
}

void
add_history (const char *s MU_ARG_UNUSED)
{
}
#endif


int
main (int argc MU_ARG_UNUSED, char **argv)
{
  char *line, *s;

  progname = strrchr (argv[0], '/');
  if (progname)
    progname++;
  else
    progname = argv[0];

  initialize_readline ();	/* Bind our completer. */

  /* Loop reading and executing lines until the user quits. */
  while (!done)
    {

      line = readline ((char *) "nntp> ");

      if (!line)
	break;

      /* Remove leading and trailing whitespace from the line.
         Then, if there is anything left, add it to the history list
         and execute it. */
      s = mu_str_stripws (line);

      if (*s)
	{
	  int status;
	  add_history (s);
	  status = execute_line (s);
	  if (status != 0)
	    fprintf (stderr, "Error: %s\n", mu_strerror (status));
	}

      free (line);
    }
  exit (0);
}

/* Parse and execute a command line. */
int
execute_line (char *line)
{
  COMMAND *command;
  char *word, *arg;

  /* Isolate the command word. */
  word = mu_str_skip_class (line, MU_CTYPE_SPACE);
  arg = mu_str_skip_class_comp (word, MU_CTYPE_SPACE);
  if (*arg)
    {
      *arg++ = 0;
      arg = mu_str_skip_class (arg, MU_CTYPE_SPACE);
    }
      
  command = find_command (word);

  if (!command)
    {
      mu_error ("%s: No such command.", word);
      return 0;
    }

  /* Call the function. */
  return ((*(command->func)) (arg));
}

/* Look up NAME as the name of a command, and return a pointer to that
   command.  Return a NULL pointer if NAME isn't a command name. */
COMMAND *
find_command (char *name)
{
  register int i;

  for (i = 0; commands[i].name; i++)
    if (strcmp (name, commands[i].name) == 0)
      return (&commands[i]);

  return ((COMMAND *) NULL);
}

int
com_verbose (char *arg)
{
  int status = 0;
  if (!valid_argument ("verbose", arg))
    return EINVAL;

  verbose = (strcmp (arg, "on") == 0);
  if (nntp != NULL)
    {
      if (verbose)
	{
	  mu_debug_t debug;
	  mu_debug_create (&debug, NULL);
	  mu_debug_set_level (debug, MU_DEBUG_LEVEL_UPTO (MU_DEBUG_PROT));
	  status = mu_nntp_set_debug (nntp, debug);
	}
      else
	{
	  status = mu_nntp_set_debug (nntp, NULL);
	}
    }
  return status;
}

int com_mode (char *arg)
{
  if (!valid_argument("mode", arg))
    return EINVAL;
  if (mu_c_strncasecmp (arg, "READER", 6) == 0)
    return com_mode_reader (arg);
  return EINVAL;
}

int
com_mode_reader (char *arg MU_ARG_UNUSED)
{
  return mu_nntp_mode_reader (nntp);
}

int
com_head (char *arg)
{
  mu_stream_t stream = NULL;
  int status;

  /* No space allowed. */
  if (arg != NULL && strchr (arg, ' ') != NULL)
    return EINVAL;

  if ((arg == NULL || *arg == '\0') || (arg != NULL && *arg == '<'))
    status = mu_nntp_head_id (nntp, arg, NULL, NULL, &stream);
  else
    {
      unsigned long number = strtoul (arg, NULL, 10);
      status = mu_nntp_head (nntp, number, NULL, NULL, &stream);
    }

   if (status == 0 && stream != NULL)
    {
      size_t n = 0;
      char buf[128];
      while ((mu_stream_readline (stream, buf, sizeof buf, 0, &n) == 0) && n)
        printf ("%s", buf);
      mu_stream_destroy (&stream, NULL);
    }
   return status;
}

int
com_body (char *arg)
{
  mu_stream_t stream = NULL;
  int status;

  /* No space allowed. */
  if (arg != NULL && strchr (arg, ' ') != NULL)
    return EINVAL;

  if ((arg == NULL || *arg == '\0') || (arg != NULL && *arg == '<'))
    status = mu_nntp_body_id (nntp, arg, NULL, NULL, &stream);
  else
    {
      unsigned long number = strtoul (arg, NULL, 10);
      status = mu_nntp_body (nntp, number, NULL, NULL, &stream);
    }

   if (status == 0 && stream != NULL)
    {
      size_t n = 0;
      char buf[128];
      while ((mu_stream_readline (stream, buf, sizeof buf, 0, &n) == 0) && n)
        printf ("%s", buf);
      mu_stream_destroy (&stream, NULL);
    }
   return status;
}

int
com_article (char *arg)
{
  mu_stream_t stream = NULL;
  int status;

  /* No space allowed. */
  if (arg != NULL && strchr (arg, ' ') != NULL)
    return EINVAL;

  if ((arg == NULL || *arg == '\0') || (arg != NULL && *arg == '<'))
    status = mu_nntp_article_id (nntp, arg, NULL, NULL, &stream);
  else
    {
      unsigned long number = strtoul (arg, NULL, 10);
      status = mu_nntp_article (nntp, number, NULL, NULL, &stream);
    }

   if (status == 0 && stream != NULL)
    {
      size_t n = 0;
      char buf[128];
      while ((mu_stream_readline (stream, buf, sizeof buf, 0, &n) == 0) && n)
        printf ("%s", buf);
      mu_stream_destroy (&stream, NULL);
    }
   return status;
}

int
com_group (char *arg)
{
  int status;
  unsigned long total, low, high;
  char *name = NULL;
  if (!valid_argument ("group", arg))
    return EINVAL;
  status = mu_nntp_group (nntp, arg, &total, &low, &high, &name);
  if (status == 0)
    {
      printf ("%s: low[%ld] high[%ld] total[%ld]\n", (name == NULL) ? "" : name, low, high, total);
      free (name);
    }
  return status;
}

int com_list (char *arg)
{
  int status = EINVAL;
  char *keyword = NULL;

  if (arg != NULL)
    {
      char *p = strchr (arg, ' ');
      if (p)
	{
	  *p++ = '\0';
	  keyword = arg;
	  arg = p;
	}
      else
	{
	  keyword = arg;
	}
    }
  else
    keyword = arg;

  if (keyword == NULL || *keyword == '\0')
    {
      status = com_list_active (arg);
   }
  else if (mu_c_strncasecmp (keyword, "ACTIVE.TIMES", 12) == 0)
    {
      status = com_list_active_times (arg);
    }
  else if (mu_c_strncasecmp (keyword, "ACTIVE", 6) == 0)
    {
      status = com_list_active (arg);
    }
  else if (mu_c_strncasecmp (keyword, "EXTENSIONS", 10) == 0)
    {
      status = com_list_extensions (arg);
    }
  else if (mu_c_strncasecmp (keyword, "DISTRIBUTIONS", 13) == 0)
    {
      status = com_list_distributions (arg);
    }
  else if (mu_c_strncasecmp (keyword, "DISTRIB.PATS", 12) == 0)
    {
      status = com_list_distrib_pats (arg);
    }
  else if (mu_c_strncasecmp (keyword, "NEWSGROUPS", 10) == 0)
    {
      status = com_list_newsgroups (arg);
    }
  return status;
}

int
com_list_extensions (char *arg MU_ARG_UNUSED)
{
  mu_iterator_t iterator = NULL;
  int status = mu_nntp_list_extensions (nntp, &iterator);

  if (status == 0)
    {
      printf ("List Extension:\n");
      for (mu_iterator_first (iterator);
	   !mu_iterator_is_done (iterator); mu_iterator_next (iterator))
	{
	  char *extension = NULL;
	  mu_iterator_current (iterator, (void **) &extension);
	  printf (" %s\n", extension);
	}
      mu_iterator_destroy (&iterator);
    }
  return status;
}

int
com_list_active (char *arg)
{
  mu_iterator_t iterator = NULL;
  int status = mu_nntp_list_active (nntp, arg, &iterator);

  if (status == 0)
    {
      printf ("List Active:\n");
      for (mu_iterator_first (iterator);
	   !mu_iterator_is_done (iterator); mu_iterator_next (iterator))
	{
	  char *buffer = NULL;
	  char *group = NULL;
	  unsigned long high;
	  unsigned long low;
	  char stat;
	  mu_iterator_current (iterator, (void **) &buffer);
	  mu_nntp_parse_list_active (buffer, &group, &high, &low, &stat);
	  if (group)
	    {
	      printf (" group(%s)", group);
	      free (group);
	    }
	  printf (" high(%ld) low(%ld) status(%c)\n", high, low, stat);
	}
      mu_iterator_destroy (&iterator);
    }
  return status;
}

int
com_list_active_times (char *arg)
{
  mu_iterator_t iterator = NULL;
  int status = mu_nntp_list_active_times (nntp, arg, &iterator);

  if (status == 0)
    {
      printf ("List Active.Times:\n");
      for (mu_iterator_first (iterator);
	   !mu_iterator_is_done (iterator); mu_iterator_next (iterator))
	{
	  char *buffer = NULL;
	  char *group = NULL;
	  char *creator = NULL;
	  unsigned long time = 0;
	  mu_iterator_current (iterator, (void **) &buffer);
	  mu_nntp_parse_list_active_times (buffer, &group, &time, &creator);
	  if (group)
	    {
	      printf (" group(%s)", group);
	      free (group);
	    }
	  if (time)
	    {
	      char *p = ctime((time_t *)&time);
	      char *buf = strdup (p);
	      p = strchr (buf, '\n');
	      if (p)
		{
		  buf[p - buf] = '\0';
		}
	      printf (" times(%s)", buf);
	      free (buf);
	    }
	  if (creator)
	    {
	      printf (" creator(%s)", creator);
	      free (creator);
	    }
	  printf ("\n");
	}
      mu_iterator_destroy (&iterator);
    }
  return status;
}

int
com_list_distributions (char *arg MU_ARG_UNUSED)
{
  mu_iterator_t iterator = NULL;
  int status = mu_nntp_list_distributions (nntp, arg, &iterator);

  if (status == 0)
    {
      printf ("List Distributions:\n");
      for (mu_iterator_first (iterator);
	   !mu_iterator_is_done (iterator); mu_iterator_next (iterator))
	{
	  char *buffer = NULL;
	  char *key = NULL;
	  char *value = NULL;
	  mu_iterator_current (iterator, (void **) &buffer);
	  mu_nntp_parse_list_distributions (buffer, &key, &value);
	  if (key)
	    {
	      printf (" %s", key);
	      free (key);
	    }
	  if (value)
	    {
	      printf (": %s", value);
	      free (value);
	    }
	  printf ("\n");
	}
      mu_iterator_destroy (&iterator);
    }
  return status;
}

int
com_list_distrib_pats (char *arg MU_ARG_UNUSED)
{
  mu_iterator_t iterator = NULL;
  int status = mu_nntp_list_distrib_pats (nntp, &iterator);

  if (status == 0)
    {
      printf ("List Distrib Pats:\n");
      for (mu_iterator_first (iterator);
	   !mu_iterator_is_done (iterator); mu_iterator_next (iterator))
	{
	  char *buffer = NULL;
	  unsigned long weight = 0;
	  char *wildmat = NULL;
	  char *distrib = NULL;
	  mu_iterator_current (iterator, (void **) &buffer);
	  mu_nntp_parse_list_distrib_pats (buffer, &weight, &wildmat, &distrib);
	  printf (" weight(%ld)", weight);
	  if (wildmat)
	    {
	      printf (":%s", wildmat);
	      free (wildmat);
	    }
	  if (distrib)
	    {
	      printf (":%s", distrib);
	      free (distrib);
	    }
	  printf ("\n");
	}
      mu_iterator_destroy (&iterator);
    }
  return status;
}

int
com_list_newsgroups (char *arg)
{
  mu_iterator_t iterator = NULL;
  int status = mu_nntp_list_newsgroups (nntp, arg, &iterator);

  if (status == 0)
    {
      printf ("Newsgroups:\n");
      for (mu_iterator_first (iterator);
	   !mu_iterator_is_done (iterator); mu_iterator_next (iterator))
	{
	  char *buffer = NULL;
	  char *group = NULL;
	  char *desc = NULL;
	  mu_iterator_current (iterator, (void **) &buffer);
	  mu_nntp_parse_list_newsgroups (buffer, &group, &desc);
	  if (group)
	    {
	      printf (" %s", group);
	      free (group);
	    }
	  if (desc)
	    {
	      printf (":%s", desc);
	      free (desc);
	    }
	  printf ("\n");
	}
      mu_iterator_destroy (&iterator);
    }
  return status;
}

int
com_last (char *arg MU_ARG_UNUSED)
{
  char *mid = NULL;
  unsigned long number = 0;
  int status;
  status = mu_nntp_last (nntp, &number, &mid);
  if (status == 0)
    {
      fprintf (stdout, "%ld %s\n", number, (mid == NULL) ? "" : mid);
      free (mid);
    }
  return status;
}

int
com_next (char *arg MU_ARG_UNUSED)
{
  char *mid = NULL;
  unsigned long number = 0;
  int status;
  status = mu_nntp_next (nntp, &number, &mid);
  if (status == 0)
    {
      fprintf (stdout, "%ld %s\n", number, (mid == NULL) ? "" : mid);
      free (mid);
    }
  return status;
}

int
com_newgroups (char *arg)
{
  mu_iterator_t iterator = NULL;
  int status;
  int year, month, day, hour, min, sec, is_gmt;
  year = month = day = hour = min = sec = is_gmt = 0;
  
  if (arg != NULL && *arg != '\0')
    {
      char gmt[4];
      memset (gmt, 0, 4);
      sscanf (arg, "%4d%2d%2d %2d%2d%2d %3s", &year, &month, &day, &hour, &min, &sec, gmt);
      is_gmt = mu_c_strncasecmp ("GMT", gmt, 3) == 0;
    }

  /* If nothing defined take the current time.  */
  if (year == 0)
    {
      time_t now = time (NULL);
      struct tm *stime = localtime (&now);
      sec = stime->tm_sec;         /* seconds */
      min = stime->tm_min;         /* minutes */
      hour = stime->tm_hour;       /* hours */
      day = stime->tm_mday;        /* day of the month */
      month = stime->tm_mon;        /* month */
      year = stime->tm_year + 1900;       /* year */
    }

  status = mu_nntp_newgroups (nntp, year, month, day, hour, min, sec, is_gmt, &iterator);
  if (status == 0)
    {
      printf ("New Groups:\n");
      for (mu_iterator_first (iterator);
	   !mu_iterator_is_done (iterator); mu_iterator_next (iterator))
	{
	  char *buffer = NULL;
	  char *group = NULL;
	  unsigned long high = 0;
	  unsigned long low = 0;
	  char stat = 0;
	  mu_iterator_current (iterator, (void **) &buffer);
	  mu_nntp_parse_newgroups (buffer, &group, &high, &low, &stat);
	  if (group)
	    {
	      printf (" group(%s)", group);
	      free (group);
	    }
	  printf (" hig(%lu) low(%lu) status(%c)\n", high, low, stat);
	}
      mu_iterator_destroy (&iterator);
    }
  return status;
}

int
com_newnews (char *arg)
{
  mu_iterator_t iterator = NULL;
  char *wildmat;
  char gmt[4];
  int year, month, day, hour, min, sec, is_gmt;
  int status;
  
  if (!valid_argument ("newnews", arg))
    return EINVAL;

  year = month = day = hour = min = sec = is_gmt = 0;
  memset (gmt, 0, 4);

  wildmat = calloc (1, 512);
  sscanf (arg, "%511s %4d%2d%2d %2d%2d%2d %3s", wildmat, &year, &month, &day, &hour, &min, &sec, gmt);
  is_gmt = mu_c_strncasecmp ("GMT", gmt, 3) == 0;

  if (year == 0)
    {
      time_t now = time (NULL);
      struct tm *stime = localtime (&now);
      sec = stime->tm_sec;         /* seconds */
      min = stime->tm_min;         /* minutes */
      hour = stime->tm_hour;       /* hours */
      day = stime->tm_mday;        /* day of the month */
      month = stime->tm_mon;        /* month */
      year = stime->tm_year + 1900;       /* year */
    }

  status = mu_nntp_newnews (nntp, wildmat, year, month, day, hour, min, sec, is_gmt, &iterator);
  if (status == 0)
    {
      printf ("New News:\n");
      for (mu_iterator_first (iterator);
	   !mu_iterator_is_done (iterator); mu_iterator_next (iterator))
	{
	  char *mid = NULL;
	  mu_iterator_current (iterator, (void **) &mid);
	  printf (" %s\n", mid);
	}
      mu_iterator_destroy (&iterator);
    }
  return status;
}

int
com_stat (char *arg)
{
  char *mid = NULL;
  unsigned long number = 0;
  int status = 0;

  /* No space allowed. */
  if (arg != NULL && strchr (arg, ' ') != NULL)
    return EINVAL;

  if ((arg == NULL || *arg == '\0') || (arg != NULL && *arg == '<'))
    status = mu_nntp_stat_id (nntp, arg, &number, &mid);
  else
    {
      unsigned long number = strtoul (arg, NULL, 10);
      status = mu_nntp_stat (nntp, number, &number, &mid);
    }
  if (status == 0)
    {
      fprintf (stdout, "status: %ld %s\n", number, (mid == NULL) ? "" : mid);
      free (mid);
    }
  return status;
}

int
com_date (char *arg MU_ARG_UNUSED)
{
  unsigned int year, month, day, hour, min, sec;
  int status;
  year = month = day = hour = min = sec = 0;
  status = mu_nntp_date (nntp, &year, &month, &day, &hour, &min, &sec);
  if (status == 0)
    {
      fprintf (stdout, "date: year(%d) month(%d) day(%d) hour(%d) min(%d) sec(%d)\n", year, month, day, hour, min, sec);
    }
  return status;
}

int
com_post (char *arg)
{
  fprintf (stderr, "Not implemented\n");
  return 0;
}

int
com_ihave (char *arg)
{
  fprintf (stderr, "Not implemented\n");
  return 0;
}

int
com_help (char *arg MU_ARG_UNUSED)
{
  mu_stream_t stream = NULL;
  int status;

  status = mu_nntp_help (nntp, &stream);
  if (status == 0 && stream != NULL)
    {
      size_t n = 0;
      char buf[128];
      while ((mu_stream_readline (stream, buf, sizeof buf, 0, &n) == 0) && n)
        printf ("%s", buf);
      mu_stream_destroy (&stream, NULL);
    }
  return status;
}

/* Print out help for ARG, or for all of the commands if ARG is
   not present. */
int
com_info (char *arg)
{
  int i;
  int printed = 0;

  for (i = 0; commands[i].name; i++)
    {
      if (!*arg || (strcmp (arg, commands[i].name) == 0))
	{
	  printf ("%s\t\t%s.\n", commands[i].name, commands[i].doc);
	  printed++;
	}
    }

  if (!printed)
    {
      printf ("No commands match `%s'.  Possibilties are:\n", arg);

      for (i = 0; commands[i].name; i++)
	{
	  /* Print in six columns. */
	  if (printed == 6)
	    {
	      printed = 0;
	      printf ("\n");
	    }

	  printf ("%s\t", commands[i].name);
	  printed++;
	}

      if (printed)
	printf ("\n");
    }
  return 0;
}

int
com_connect (char *arg)
{
  char host[256];
  int port = 119;
  int status;

  *host = '\0';

  /* Try with the environment.  */
  if (arg == NULL || *arg == '\0')
    arg = getenv ("NNTPSERVER");
  if (!valid_argument ("connect", arg))
    return EINVAL;
  sscanf (arg, "%256s %d", host, &port);
  if (!valid_argument ("connect", host))
    return EINVAL;
  if (nntp)
    com_disconnect (NULL);
  status = mu_nntp_create (&nntp);
  if (status == 0)
    {
      mu_stream_t tcp;

      if (verbose)
	com_verbose ("on");
      status =
	mu_tcp_stream_create (&tcp, host, port,
			   MU_STREAM_READ | MU_STREAM_NO_CHECK);
      if (status == 0)
	{
	  mu_nntp_set_carrier (nntp, tcp);
	  status = mu_nntp_connect (nntp);
	}
      else
	{
	  mu_nntp_destroy (&nntp);
	  nntp = NULL;
	}
    }

  if (status != 0)
    fprintf (stderr, "Failed to create nntp: %s\n", mu_strerror (status));
  return status;
}

int
com_disconnect (char *arg MU_ARG_UNUSED)
{
  (void) arg;
  if (nntp)
    {
      mu_nntp_disconnect (nntp);
      mu_nntp_destroy (&nntp);
      nntp = NULL;
    }
  return 0;
}

int
com_quit (char *arg MU_ARG_UNUSED)
{
  int status = 0;
  if (nntp)
    {
      if (mu_nntp_quit (nntp) == 0)
	{
	  status = com_disconnect (arg);
	}
      else
	{
	  fprintf (stdout, "Try 'exit' to leave %s\n", progname);
	}
    }
  else
    fprintf (stdout, "Try 'exit' to leave %s\n", progname);
  return status;
}

int
com_exit (char *arg MU_ARG_UNUSED)
{
  if (nntp)
    {
      mu_nntp_disconnect (nntp);
      mu_nntp_destroy (&nntp);
    }
  done = 1;
  return 0;
}

/* Return non-zero if ARG is a valid argument for CALLER, else print
   an error message and return zero. */
int
valid_argument (const char *caller, char *arg)
{
  if (arg == NULL || *arg == '\0')
    {
      fprintf (stderr, "%s: Argument required.\n", caller);
      return 0;
    }

  return 1;
}
