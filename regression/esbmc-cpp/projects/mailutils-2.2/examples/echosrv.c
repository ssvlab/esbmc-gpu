/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2008, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; If not, see
   <http://www.gnu.org/licenses/>.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <sys/wait.h>

#include <mailutils/mailutils.h>
#include <mailutils/server.h>

mu_server_t server;

int
echo_conn (int fd, struct sockaddr *s, int len,
	   void *server_data, void *call_data,
	   mu_ip_server_t srv)
{
  struct sockaddr_in srv_addr, *s_in = (struct sockaddr_in *)s;
  int addrlen = sizeof srv_addr;
  pid_t pid;
  char buf[512];
  FILE *in, *out;
  
  mu_ip_server_get_sockaddr (srv, (struct sockaddr *)&srv_addr, &addrlen);

  pid = fork ();
  if (pid == -1)
    {
      mu_error ("fork failed: %s", mu_strerror (errno));
      return 0;
    }

  if (pid)
    {
      mu_diag_output (MU_DIAG_INFO, "%lu: opened connection %s:%d => %s:%d",
		      (unsigned long) pid,
		      inet_ntoa (srv_addr.sin_addr), ntohs (srv_addr.sin_port),
		      inet_ntoa (s_in->sin_addr), ntohs (s_in->sin_port));
      return 0;
    }

  mu_ip_server_shutdown (srv);

  in = fdopen (fd, "r");
  out = fdopen (fd, "w");
  setvbuf (in, NULL, _IOLBF, 0);
  setvbuf (out, NULL, _IOLBF, 0);

  pid = getpid ();
  while (fgets (buf, sizeof (buf), in) > 0)
    {
      int len = strlen (buf);
      if (len > 0 && buf[len-1] == '\n')
	{
	  buf[--len] = 0;
	  if (buf[len-1] == '\r')
	    buf[--len] = 0;
	}
      fprintf (out, "%lu: you said: \"%s\"\r\n", (unsigned long) pid, buf);
    }
  exit (0);
}

int
tcp_conn_handler (int fd, void *conn_data, void *server_data)
{
  mu_ip_server_t tcpsrv = (mu_ip_server_t) conn_data;
  int rc = mu_ip_server_accept (tcpsrv, server_data);
  if (rc && rc != EINTR)
    {
      mu_ip_server_shutdown (tcpsrv);
      return MU_SERVER_CLOSE_CONN;
    }
  return MU_SERVER_SUCCESS;
}

void
tcp_conn_free (void *conn_data, void *server_data)
{
  mu_ip_server_t tcpsrv = (mu_ip_server_t) conn_data;
  mu_ip_server_destroy (&tcpsrv);
}

void
create_server (char *arg)
{
  char *p, *q;
  struct sockaddr_in s;
  mu_ip_server_t tcpsrv;
  unsigned n;
  
  p = strchr (arg, ':');
  if (!*p)
    {
      mu_error ("invalid specification: %s\n", arg);
      exit (1);
    }
  *p++ = 0;
  s.sin_family = AF_INET;
  if (inet_aton (arg, &s.sin_addr) == 0)
    {
      mu_error ("invalid IP address: %s\n", arg);
      exit (1);
    }
  n = strtoul (p, &q, 0);
  if (*q)
    {
      mu_error ("invalid port number: %s\n", p);
      exit (1);
    }      
  s.sin_port = htons (n);

  MU_ASSERT (mu_ip_server_create (&tcpsrv, (struct sockaddr*) &s, sizeof s,
				  MU_IP_TCP));
  MU_ASSERT (mu_ip_server_open (tcpsrv));
  MU_ASSERT (mu_ip_server_set_conn (tcpsrv, echo_conn));
  MU_ASSERT (mu_server_add_connection (server,
				       mu_ip_server_get_fd (tcpsrv),
				       tcpsrv,
				       tcp_conn_handler, tcp_conn_free));
}

static int cleanup_needed;

RETSIGTYPE
sig_child (int sig)
{
  cleanup_needed = 1;
  signal (sig, sig_child);
}

int
server_idle (void *server_data)
{
  if (cleanup_needed)
    {
      int status;
      pid_t pid;

      cleanup_needed = 0;
      while ((pid = waitpid (-1, &status, WNOHANG)) > 0)
	{
	  if (WIFEXITED (status))
	    mu_diag_output (MU_DIAG_INFO, "%lu: finished with code %d",
			    (unsigned long) pid,
			    WEXITSTATUS (status));
	  else if (WIFSIGNALED (status))
	    mu_diag_output (MU_DIAG_ERR, "%lu: terminated on signal %d",
			    (unsigned long) pid,
			    WTERMSIG (status));
	  else
	    mu_diag_output (MU_DIAG_ERR, "%lu: terminated (cause unknown)",
			    (unsigned long) pid);
	}
    }
  return 0;
}

int
run ()
{
  int rc;
  signal (SIGCHLD, sig_child);
  rc = mu_server_run (server);
  if (rc)
    mu_error ("%s", mu_strerror (rc));
  mu_server_destroy (&server);
  return rc ? 1 : 0;
}

int
main (int argc, char **argv)
{
  int rc;
  
  mu_set_program_name (argv[0]);
  while ((rc = getopt (argc, argv, "Dd:")) != EOF)
    {
      switch (rc)
	{
	case 'D':
	  mu_debug_line_info = 1;
	  break;
	  
	case 'd':
	  mu_global_debug_from_string (optarg, "command line");
	  break;

	default:
	  exit (1);
	}
    }

  argc -= optind;
  argv += optind;

  MU_ASSERT (mu_server_create (&server));
  mu_server_set_idle (server, server_idle);
  while (argc--)
    create_server (*argv++);
  return run ();
}

