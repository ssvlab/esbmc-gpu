/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2007, 2010 Free Software Foundation,
   Inc.

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

/*
 * ec[ho] string ...
 */

static int echo (char *s);

int
mail_echo (int argc, char **argv)
{
  int i = 0;
  if (argc > 1)
    {
      for (i = 1; i < argc - 1; i++)
	{
	  echo (argv[i]);
	  fputc (' ', ofile);
	}
      /* Last argument.  */
      if (echo(argv[argc - 1]) == 0)
	fputc ('\n', ofile);
    }
  return 0;
}

/* Cumbersome switch for checking escape char '\'
   if present replace with appropriately.
   Return of 1 means to not print newline.  */
static int
echo (char *s)
{
  int process_escape = 0;
  int c;

  if (s == NULL)
    return 0;

  for (; (c = *s) != 0; s++)
    {
      if (process_escape)
	{
	  switch (c)
	    {
	      /* \a Bell.  */
	    case 'a':
	      c = '\a';
	      break;

	      /* \b Backspace.  */
	    case 'b':
	      c = '\b';
	      break;

	      /* \c means not to print ending newline.  */
	      /* Bail out and tell the caller not to print newline.  */
	    case 'c':
	      return 1;
	      break;

	      /* \f Formfeed.  */
	    case 'f':
	      c = '\f';
	      break;

	      /* \n Newline.  */
	    case 'n':
	      c = '\n';
	      break;

	      /* \r Carriage return.  */
	    case 'r':
	      c = '\r';
	      break;

	      /* \t Tab. */
	    case 't':
	      c = '\t';
	      break;

	      /* \v Vertical Tab.  */
	    case 'v':
	      c = '\v';
	      break;

	      /* Escape sequence.  */
	    case '\\':
	      c = '\\';
	      break;

	      /* \0x99 for example, let strtol() handle it.  */
	      /* WARNING: Side effects because of strtol().  */
	    case '0':
	      {
		long number = strtol (s, &s, 0);
		switch (number)
		  {
		  case LONG_MIN:
		  case LONG_MAX:
		    /* if (errno == ERANGE) */
		    /*  fputc (c, ofile); */
		    break;

		  default:
		    fprintf (ofile, "%ld", number);
		    s--;
		    continue;
		  }
	      }
	      break;

	      /* Can not be here.  */
	    case '\0':
	      return 0;
	      break;

	      /* \\ means \ It was not an escape char.  */
	    default:
	      fputc ('\\', ofile);

	    }
	  process_escape =0;
	}
      else if (c == '\\') /* Find the escape char, go back and process.  */
	{
	  process_escape = 1;
	  continue;
	}
      fputc (c, ofile);
    }
  return 0;
}
