/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2006, 2007, 2010 Free Software Foundation, Inc.

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
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include <mailutils/mailutils.h>

/* Replace all octal escapes in BUF with the corresponding characters. */
static void
decode_octal (char *buf)
{
  char *p;
  unsigned i, n;
  
  for (p = buf; *p;)
    {
      if (*buf == '\\')
	{
	  buf++;
	  switch (*buf)
	    {
	    case 'a':
	      *p++ = '\a';
	      buf++;
	      break;
	      
	    case 'b':
	      *p++ = '\b';
	      buf++;
	      break;
	      
	    case 'f':
	      *p++ = '\f';
	      buf++;
	      break;
	      
	    case 'n':
	      *p++ = '\n';
	      buf++;
	      break;
	      
	    case 'r':
	      *p++ = '\r';
	      buf++;
	      break;
	      
	    case 't':
	      *p++ = '\t';
	      buf++;
	      break;

	    case '0': case '1': case '2': case '3':
	    case '4': case '5': case '6': case '7':
	      n = 0;
	      for (i = 0; i < 3; i++, buf++)
		{
		  unsigned x = *(unsigned char*)buf - '0';
		  if (x > 7)
		    break;
		  n <<= 3;
		  n += x;
		}
	      if (i != 3)
		{
		  buf -= i;
		  *p++ = '\\';
		}
	      else
		*p++ = n;
	      break;

	    default:
	      *p++ = '\\';
	      *p++ = *buf++;
	      break;
	    }
	}
      else
	*p++ = *buf++;
    }
  *p = 0;
}

int
main (int argc, char *argv[])
{
  int c;
  char buf[256];
  char vbuf[256];
  char *charset = strdup ("iso-8859-1");
  char *encoding = strdup ("quoted-printable");
  int octal = 0;
  
  while ((c = getopt (argc, argv, "c:e:hot")) != EOF)
    switch (c)
      {
      case 'c':
	free (charset);
	charset = strdup (optarg);
	break;
	
      case 'e':
	free (encoding);
	encoding = strdup (optarg);
	break;

      case 'o':
	octal = 1;
	break;

      case 't':
	octal = 0;
	break;
	
      case 'h':
	printf ("usage: %s [-c charset] [-e encoding] [-ot]\n", argv[0]);
	exit (0);
	
      default:
	exit (1);
      }

  while (fgets (buf, sizeof (buf), stdin))
    {
      int len;
      char *p = NULL;
      char *cmd;
      int rc;
	
      len = strlen (buf);
      if (len > 0 && buf[len - 1] == '\n')
	buf[len - 1] = 0;
      strncpy(vbuf, buf, sizeof vbuf);
      cmd = vbuf;
      if (cmd[0] == '\\')
	{
	  if (cmd[1] == 0)
	    {
	      fprintf (stderr, "Unfinished command\n");
	      continue;
	    }
	  
	  for (p = cmd + 2; *p && *p == ' '; p++)
	    ;
	  switch (cmd[1])
	    {
	    case 'c':
	      free (charset);
	      charset = strdup (p);
	      continue;
	      
	    case 'e':
	      free (encoding);
	      encoding = strdup (p);
	      continue;
	      
	    case 'o':
	      octal = 1;
	      continue;
	      
	    case 't':
	      octal = 0;
	      continue;

	    case '\\':
	      cmd++;
	      break;
	      
	    default:
	      fprintf (stderr, "Unknown command\n");
	      continue;
	    }
	}

      if (octal)
	decode_octal (cmd);
	  
      rc = mu_rfc2047_encode (charset, encoding, cmd, &p);
      printf ("%s=> %s\n", buf, mu_strerror (rc));
      if (p)
	printf ("%s\n", p);
      free (p);
    }
    return 0;
}
