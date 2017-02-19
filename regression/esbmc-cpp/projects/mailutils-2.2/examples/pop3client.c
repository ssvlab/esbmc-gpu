/* pop3client.c -- An application which demonstrates how to use the
   GNU Mailutils pop3 functions.  This application interactively allows users
   to contact a pop3 server.

   Copyright (C) 2003, 2004, 2005, 2007, 2008, 2009, 2010 Free Software
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
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <termios.h>
#include <signal.h>

#ifdef WITH_READLINE
# include <readline/readline.h>
# include <readline/history.h>
#endif

#include <mailutils/pop3.h>
#include <mailutils/iterator.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/vartab.h>
#include <mailutils/argcv.h>
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
int com_apop (char *);
int com_capa (char *);
int com_disconnect (char *);
int com_dele (char *);
int com_exit (char *);
int com_help (char *);
int com_list (char *);
int com_noop (char *);
int com_connect (char *);
int com_pass (char *);
int com_quit (char *);
int com_retr (char *);
int com_rset (char *);
int com_stat (char *);
int com_top (char *);
int com_uidl (char *);
int com_user (char *);
int com_verbose (char *);
int com_prompt (char *);

void initialize_readline (void);
COMMAND *find_command (char *);
char *dupstr (const char *);
int execute_line (char *);
int valid_argument (const char *, char *);

void sig_int (int);

COMMAND commands[] = {
  { "apop", com_apop, "Authenticate with APOP: APOP user secret" },
  { "capa", com_capa, "List capabilities: capa" },
  { "disconnect", com_disconnect, "Close connection: disconnect" },
  { "dele", com_dele, "Mark message: DELE msgno" },
  { "exit", com_exit, "exit program" },
  { "help", com_help, "Display this text" },
  { "?", com_help, "Synonym for `help'" },
  { "list", com_list, "List messages: LIST [msgno]" },
  { "noop", com_noop, "Send no operation: NOOP" },
  { "pass", com_pass, "Send passwd: PASS [passwd]" },
  { "prompt", com_prompt, "Set command prompt" },
  { "connect", com_connect, "Open connection: connect hostname [port]" },
  { "quit", com_quit, "Go to Update state : QUIT" },
  { "retr", com_retr, "Dowload message: RETR msgno" },
  { "rset", com_rset, "Unmark all messages: RSET" },
  { "stat", com_stat, "Get the size and count of mailbox : STAT [msgno]" },
  { "top", com_top, "Get the header of message: TOP msgno [lines]" },
  { "uidl", com_uidl, "Get the unique id of message: UIDL [msgno]" },
  { "user", com_user, "send login: USER user" },
  { "verbose", com_verbose, "Enable Protocol tracing: verbose [on|off]" },
  { NULL, NULL, NULL }
};

/* Global handle for pop3.  */
mu_pop3_t pop3;

/* Flag if verbosity is needed.  */
int verbose;

/* When non-zero, this global means the user is done using this program. */
int done;

enum pop_session_status
  {
    pop_session_disconnected,
    pop_session_connected,
    pop_session_logged_in
  };

enum pop_session_status pop_session_status;

int connect_argc;
char **connect_argv;

/* Host we are connected to. */
#define host connect_argv[0]
int port = 110;
char *username;

/* Command line prompt */
#define DEFAULT_PROMPT "pop3> "
char *prompt;

const char *
pop_session_str (enum pop_session_status stat)
{
  switch (stat)
    {
    case pop_session_disconnected:
      return "disconnected";
      
    case pop_session_connected:
      return "connected";
      
    case pop_session_logged_in:
      return "logged in";
    }
  return "unknown";
}

char *
expand_prompt ()
{
  mu_vartab_t vtab;
  char *str;
  
  if (mu_vartab_create (&vtab))
    return strdup (prompt);
  mu_vartab_define (vtab, "user",
		    (pop_session_status == pop_session_logged_in) ?
		      username : "not logged in", 1);
  mu_vartab_define (vtab, "host",
		    (pop_session_status != pop_session_disconnected) ?
		      host : "not connected", 1);
  mu_vartab_define (vtab, "program-name", mu_program_name, 1);
  mu_vartab_define (vtab, "canonical-program-name", "pop3client", 1);
  mu_vartab_define (vtab, "package", PACKAGE, 1);
  mu_vartab_define (vtab, "version", PACKAGE_VERSION, 1);
  mu_vartab_define (vtab, "status", pop_session_str (pop_session_status), 1);
  
  if (mu_vartab_expand (vtab, prompt, &str))
    str = strdup (prompt);
  mu_vartab_destroy (&vtab);
  return str;
}

char *
dupstr (const char *s)
{
  char *r;

  r = malloc (strlen (s) + 1);
  if (!r)
    {
      mu_error ("Memory exhausted");
      exit (1);
    }
  strcpy (r, s);
  return r;
}


#ifdef WITH_READLINE
/* Interface to Readline Completion */

char *command_generator (const char *, int);
char **pop_completion (char *, int, int);

/* Tell the GNU Readline library how to complete.  We want to try to complete
   on command names if this is the first word in the line, or on filenames
   if not. */
void
initialize_readline ()
{
  /* Allow conditional parsing of the ~/.inputrc file. */
  rl_readline_name = (char *) "pop3";

  /* Tell the completer that we want a crack first. */
  rl_attempted_completion_function = (CPPFunction *) pop_completion;
}

/* Attempt to complete on the contents of TEXT.  START and END bound the
   region of rl_line_buffer that contains the word to complete.  TEXT is
   the word to complete.  We can use the entire contents of rl_line_buffer
   in case we want to do some simple parsing.  Return the array of matches,
   or NULL if there aren't any. */
char **
pop_completion (char *text, int start, int end MU_ARG_UNUSED)
{
  char **matches;

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
  return NULL;
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

  mu_set_program_name (argv[0]);
  prompt = strdup (DEFAULT_PROMPT);
  initialize_readline ();	/* Bind our completer. */

  /* Loop reading and executing lines until the user quits. */
  while (!done)
    {
      char *p = expand_prompt ();
      line = readline (p);
      free (p);
      
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
	    mu_error ("Error: %s", mu_strerror (status));
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
  if (pop3 != NULL)
    {
      if (verbose == 1)
	{
	  mu_debug_t debug;
	  mu_debug_create (&debug, NULL);
	  mu_debug_set_level (debug, MU_DEBUG_LEVEL_UPTO (MU_DEBUG_PROT));
	  status = mu_pop3_set_debug (pop3, debug);
	}
      else
	{
	  status = mu_pop3_set_debug (pop3, NULL);
	}
    }
  return status;
}

int
com_user (char *arg)
{
  int status;
  
  if (!valid_argument ("user", arg))
    return EINVAL;
  status = mu_pop3_user (pop3, arg);
  if (status == 0)
    username = strdup (arg);
  return status;
}

int
com_apop (char *arg)
{
  int status;
  char *user, *digest;

  if (!valid_argument ("apop", arg))
    return EINVAL;
  user = strtok (arg, " ");
  digest = strtok (NULL, " ");
  if (!valid_argument ("apop", user) || !valid_argument ("apop", digest))
    return EINVAL;
  status = mu_pop3_apop (pop3, user, digest);
  if (status == 0)
    {
      username = strdup (user);
      pop_session_status = pop_session_logged_in;
    }
  return status;
}

int
com_capa (char *arg MU_ARG_UNUSED)
{
  mu_iterator_t iterator = NULL;
  int status = mu_pop3_capa (pop3, &iterator);

  if (status == 0)
    {
      for (mu_iterator_first (iterator);
	   !mu_iterator_is_done (iterator); mu_iterator_next (iterator))
	{
	  char *capa = NULL;
	  mu_iterator_current (iterator, (void **) &capa);
	  printf ("Capa: %s\n", (capa) ? capa : "");
	}
      mu_iterator_destroy (&iterator);
    }
  return status;
}

int
com_uidl (char *arg)
{
  int status = 0;
  if (arg == NULL || *arg == '\0')
    {
      mu_iterator_t uidl_iterator = NULL;
      status = mu_pop3_uidl_all (pop3, &uidl_iterator);
      if (status == 0)
	{
	  for (mu_iterator_first (uidl_iterator);
	       !mu_iterator_is_done (uidl_iterator);
	       mu_iterator_next (uidl_iterator))
	    {
	      char *uidl = NULL;
	      mu_iterator_current (uidl_iterator, (void **) &uidl);
	      printf ("UIDL: %s\n", (uidl) ? uidl : "");
	    }
	  mu_iterator_destroy (&uidl_iterator);
	}
    }
  else
    {
      char *uidl = NULL;
      unsigned int msgno = strtoul (arg, NULL, 10);
      status = mu_pop3_uidl (pop3, msgno, &uidl);
      if (status == 0)
	printf ("Msg: %d UIDL: %s\n", msgno, (uidl) ? uidl : "");
      free (uidl);
    }
  return status;
}

int
com_list (char *arg)
{
  int status = 0;
  if (arg == NULL || *arg == '\0')
    {
      mu_iterator_t list_iterator;
      status = mu_pop3_list_all (pop3, &list_iterator);
      if (status == 0)
	{
	  for (mu_iterator_first (list_iterator);
	       !mu_iterator_is_done (list_iterator);
	       mu_iterator_next (list_iterator))
	    {
	      char *list = NULL;
	      mu_iterator_current (list_iterator, (void **) &list);
	      printf ("LIST: %s\n", (list) ? list : "");
	    }
	  mu_iterator_destroy (&list_iterator);
	}
    }
  else
    {
      size_t size = 0;
      unsigned int msgno = strtoul (arg, NULL, 10);
      status = mu_pop3_list (pop3, msgno, &size);
      if (status == 0)
	printf ("Msg: %u Size: %lu\n", msgno, (unsigned long) size);
    }
  return status;
}

int
com_noop (char *arg MU_ARG_UNUSED)
{
  return mu_pop3_noop (pop3);
}

static void
echo_off (struct termios *stored_settings)
{
  struct termios new_settings;
  tcgetattr (0, stored_settings);
  new_settings = *stored_settings;
  new_settings.c_lflag &= (~ECHO);
  tcsetattr (0, TCSANOW, &new_settings);
}

static void
echo_on (struct termios *stored_settings)
{
  tcsetattr (0, TCSANOW, stored_settings);
}

int
com_prompt (char *arg)
{
  int quote;
  size_t size;
  
  if (!valid_argument ("prompt", arg))
    return EINVAL;

  free (prompt);
  size = mu_argcv_quoted_length (arg, &quote);
  prompt = malloc (size + 1);
  if (!prompt)
    {
      mu_error ("Memory exhausted");
      exit (1);
    }
  mu_argcv_unquote_copy (prompt, arg, size);
  return 0;
}

int
com_pass (char *arg)
{
  int status;
  char pass[256];
  
  if (!arg || *arg == '\0')
    {
      struct termios stored_settings;

      printf ("passwd:");
      fflush (stdout);
      echo_off (&stored_settings);
      fgets (pass, sizeof pass, stdin);
      echo_on (&stored_settings);
      putchar ('\n');
      fflush (stdout);
      pass[strlen (pass) - 1] = '\0';	/* nuke the trailing line.  */
      arg = pass;
    }
  status = mu_pop3_pass (pop3, arg);
  if (status == 0)
    pop_session_status = pop_session_logged_in;
  return status;
}

int
com_stat (char *arg MU_ARG_UNUSED)
{
  unsigned count = 0;
  size_t size = 0;
  int status = 0;

  status = mu_pop3_stat (pop3, &count, &size);
  printf ("Mesgs: %lu Size %lu\n",
	  (unsigned long) count, (unsigned long) size);
  return status;
}

int
com_dele (char *arg)
{
  unsigned msgno;
  if (!valid_argument ("dele", arg))
    return EINVAL;
  msgno = strtoul (arg, NULL, 10);
  return mu_pop3_dele (pop3, msgno);
}

/* Print out help for ARG, or for all of the commands if ARG is
   not present. */
int
com_help (char *arg)
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
com_rset (char *arg MU_ARG_UNUSED)
{
  return mu_pop3_rset (pop3);
}

int
com_top (char *arg)
{
  mu_stream_t stream;
  unsigned int msgno;
  unsigned int lines;
  char *space;
  int status;

  if (!valid_argument ("top", arg))
    return EINVAL;

  space = strchr (arg, ' ');
  if (space)
    {
      *space++ = '\0';
      lines = strtoul (space, NULL, 10);
    }
  else
    lines = 0;
  msgno = strtoul (arg, NULL, 10);

  status = mu_pop3_top (pop3, msgno, lines, &stream);

  if (status == 0)
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
com_retr (char *arg)
{
  mu_stream_t stream;
  unsigned int msgno;
  int status;

  if (!valid_argument ("retr", arg))
    return EINVAL;

  msgno = strtoul (arg, NULL, 10);
  status = mu_pop3_retr (pop3, msgno, &stream);

  if (status == 0)
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
get_port (const char *port_str, int *pn)
{
  short port_num;
  long num;
  char *p;
  
  num = port_num = strtol (port_str, &p, 0);
  if (*p == 0)
    {
      if (num != port_num)
	{
	  mu_error ("bad port number: %s", port_str);
	  return 1;
	}
    }
  else
    {
      struct servent *sp = getservbyname (port_str, "tcp");
      if (!sp)
	{
	  mu_error ("unknown port name");
	  return 1;
	}
      port_num = ntohs (sp->s_port);
    }
  *pn = port_num;
  return 0;
}

int
com_connect (char *arg)
{
  int status;
  int n = 110;
  int argc;
  char **argv;
  
  if (!valid_argument ("connect", arg))
    return 1;
  
  if (mu_argcv_get (arg, NULL, NULL, &argc, &argv))
    {
      mu_error ("Cannot parse arguments");
      return 0;
    }
  
  if (!valid_argument ("connect", argv[0]))
    {
      mu_argcv_free (argc, argv);
      return EINVAL;
    }
  
  if (argc > 2)
    {
      mu_error ("Too many arguments");
      mu_argcv_free (argc, argv);
      return 0;
    }

  if (argc == 2 && get_port (argv[1], &n))
    {
      mu_argcv_free (argc, argv);
      return 0;
    }
  
  if (pop_session_status != pop_session_disconnected)
    com_disconnect (NULL);
  
  status = mu_pop3_create (&pop3);
  if (status == 0)
    {
      mu_stream_t tcp;

      if (verbose)
	com_verbose ("on");
      status =
	mu_tcp_stream_create (&tcp, argv[0], n,
			      MU_STREAM_READ | MU_STREAM_NO_CHECK);
      if (status == 0)
	{
	  mu_pop3_set_carrier (pop3, tcp);
	  status = mu_pop3_connect (pop3);
	}
      else
	{
	  mu_pop3_destroy (&pop3);
	  pop3 = NULL;
	}
    }

  if (status)
    {
      mu_error ("Failed to create pop3: %s", mu_strerror (status));
      mu_argcv_free (argc, argv);
    }
  else
    {
      connect_argc = argc;
      connect_argv = argv;
      port = n;
      pop_session_status = pop_session_connected;
    }
  
  return status;
}

int
com_disconnect (char *arg MU_ARG_UNUSED)
{
  if (pop3)
    {
      mu_pop3_disconnect (pop3);
      mu_pop3_destroy (&pop3);
      pop3 = NULL;
      
      mu_argcv_free (connect_argc, connect_argv);
      connect_argc = 0;
      connect_argv = NULL;
      pop_session_status = pop_session_disconnected;
    }
  return 0;
}

int
com_quit (char *arg MU_ARG_UNUSED)
{
  int status = 0;
  if (pop3)
    {
      if (mu_pop3_quit (pop3) == 0)
	{
	  status = com_disconnect (arg);
	}
      else
	{
	  printf ("Try 'exit' to leave %s\n", mu_program_name);
	}
    }
  else
    printf ("Try 'exit' to leave %s\n", mu_program_name);
  return status;
}

int
com_exit (char *arg MU_ARG_UNUSED)
{
  if (pop3)
    {
      mu_pop3_disconnect (pop3);
      mu_pop3_destroy (&pop3);
    }
  done = 1;
  return 0;
}

/* Return non-zero if ARG is a valid argument for CALLER, else print
   an error message and return zero. */
int
valid_argument (const char *caller, char *arg)
{
  if (!arg || !*arg)
    {
      mu_error ("%s: Argument required", caller);
      return 0;
    }

  return 1;
}
