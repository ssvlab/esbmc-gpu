/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2010 Free Software
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
#include <mailutils/mailutils.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/un.h>
#include <arpa/inet.h>

#include <stdlib.h>
#include <string.h>

struct sockaddr *target_sa;
int target_salen;
mu_acl_t acl;

struct sockaddr *
parse_address (int *psalen, char *str)
{
  struct sockaddr_in in;
  struct sockaddr *sa;
  
  in.sin_family = AF_INET;
  if (inet_aton (str, &in.sin_addr) == 0)
    {
      mu_error ("Invalid IPv4: %s", str);
      exit (1);
    }
  in.sin_port = 0;
  *psalen = sizeof (in);
  sa = malloc (*psalen);
  if (!sa)
    {
      mu_error ("%s", mu_strerror (errno));
      exit (1);
    }
  
  memcpy (sa, &in, sizeof (in));
  return sa;
}

void
read_rules (FILE *fp)
{
  char buf[512];
  int line = 0;
  int argc = 0;
  char **argv;
  int rc;
  
  rc = mu_acl_create (&acl);
  if (rc)
    {
      mu_error ("cannot create acl: %s", mu_strerror (rc));
      exit (1);
    }
  
  while (fgets (buf, sizeof buf, fp))
    {
      unsigned long netmask;
      int salen;
      struct sockaddr *sa;
      mu_acl_action_t action;
      void *data = NULL;
      char *p;
      
      int len = strlen (buf);
      if (len == 0)
	continue;
      if (buf[len-1] != '\n')
	{
	  mu_error ("%d: line too long", line);
	  continue;
	}
      buf[len-1] = 0;
      line++;
      if (buf[0] == '#')
	continue;

      if (argc)
	mu_argcv_free (argc, argv);

      mu_argcv_get (buf, " \t", "#", &argc, &argv);
      if (argc < 2)
	{
 	  mu_error ("%d: invalid input", line);
	  continue;
	}

      p = strchr (argv[1], '/');
      if (p)
	{
	  char *q;
	  unsigned netlen;
	  
	  *p++ = 0;
	  netlen = strtoul (p, &q, 10);
	  if (*q == 0)
	    {
	      if (netlen == 0)
		netmask = 0;
	      else
		{
		  netmask = 0xfffffffful >> (32 - netlen);
		  netmask <<= (32 - netlen);
		  netmask = htonl (netmask);
		}
	    }
	  else if (*q == '.')
	    {
	      struct in_addr addr;
	      
	      if (inet_aton (p, &addr) == 0)
		{
		  mu_error ("%d: invalid netmask", line);
		  continue;
		}
	      netmask = addr.s_addr;
	    }
	  else
	    {
	      mu_error ("%d: invalid netmask", line);
	      continue;
	    }
	}
      else
	netmask = 0xfffffffful;
      
      sa = parse_address (&salen, argv[1]);
      
      /* accept addr
	 deny addr
	 log addr [rest ...]
	 exec addr [rest ...]
	 execif addr rest ....]
      */
      if (mu_acl_string_to_action (argv[0], &action))
	{
	  mu_error ("%d: invalid command", line);
	  continue;
	}

      switch (action)
	{
	case mu_acl_accept:
	case mu_acl_deny:
	  break;

	case mu_acl_log:
	case mu_acl_exec:
	case mu_acl_ifexec:
	  data = strdup (argv[2]);
	}

      rc = mu_acl_append (acl, action, data, sa, salen, netmask);
      if (rc)
	mu_error ("%d: cannot append acl entry: %s", line,
		  mu_strerror (rc));
    }
}

int
main (int argc, char **argv)
{
  int rc;
  FILE *file = NULL;
  mu_acl_result_t result;
  
  mu_set_program_name (argv[0]);
  while ((rc = getopt (argc, argv, "Dd:a:f:")) != EOF)
    {
      switch (rc)
	{
	case 'D':
	  mu_debug_line_info = 1;
	  break;
	  
	case 'd':
	  mu_global_debug_from_string (optarg, "command line");
	  break;

	case 'a':
	  target_sa = parse_address (&target_salen, optarg);
	  break;

	case 'f':
	  file = fopen (optarg, "r");
	  if (file == 0)
	    {
	      mu_error ("cannot open file %s: %s", optarg, 
			mu_strerror (errno));
	      exit (1);
	    }
	  break;
	  
	default:
	  exit (1);
	}
    }

  argv += optind;
  argc -= optind;

  read_rules (file ? file : stdin);
  rc = mu_acl_check_sockaddr (acl, target_sa, target_salen, &result);
  if (rc)
    {
      mu_error ("mu_acl_check_sockaddr failed: %s", mu_strerror (rc));
      exit (1);
    }
  switch (result)
    {
    case mu_acl_result_undefined:
      puts ("undefined");
      break;
      
    case mu_acl_result_accept:
      puts ("accept");
      break;

    case mu_acl_result_deny:
      puts ("deny");
      break;
    }
  exit (0);
}
