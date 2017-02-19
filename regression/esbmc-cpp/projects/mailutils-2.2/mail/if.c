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

#define COND_STK_SIZE 64
#define COND_STK_INCR 16
static int *_cond_stack;     /* Stack of conditions */
static int _cond_stack_size; /* Number of elements allocated this far */
static int _cond_level;      /* Number of nested `if' blocks */

static void _cond_push(int val);
static int _cond_pop(void);

int
if_cond()
{
  if (_cond_level == 0)
    return 1;
  return _cond_stack[_cond_level-1];
}

void
_cond_push(int val)
{
  if (!_cond_stack)
    {
      _cond_stack = calloc(COND_STK_SIZE, sizeof(_cond_stack[0]));
      _cond_stack_size = COND_STK_SIZE;
      _cond_level = 0;
    }
  else if (_cond_level >= _cond_stack_size)
    {
      _cond_stack_size += COND_STK_INCR;
      _cond_stack = realloc(_cond_stack,
			    sizeof(_cond_stack[0])*_cond_stack_size);
    }

  if (!_cond_stack)
    {
      util_error(_("Not enough memory"));
      exit (EXIT_FAILURE);
    }
  _cond_stack[_cond_level++] = val;
}

int
_cond_pop()
{
  if (_cond_level == 0)
    {
      util_error(_("Internal error: condition stack underflow"));
      abort();
    }
  return _cond_stack[--_cond_level];
}

/*
 * i[f] s|r|t
 * mail-commands
 * el[se]
 * mail-commands
 * en[dif]
 */

int
mail_if (int argc, char **argv)
{
  char *mode;
  int cond;

  if (argc != 2)
    {
      /* TRANSLATORS: 'if' is the function name. Do not translate it */
      util_error(_("if requires an argument: s | r | t"));
      return 1;
    }

  if (argv[1][1] != 0)
    {
      util_error(_("Valid if arguments are: s | r | t"));
      return 1;
    }

  if (mailvar_get (&mode, "mode", mailvar_type_string, 1))
    exit (EXIT_FAILURE);

  if (if_cond() == 0)
    /* Propagate negative condition */
    cond = 0;
  else
    {
      switch (argv[1][0])
	{
	case 's': /* Send mode */
	  cond = strcmp(mode, "send") == 0;
	  break;
	case 'r': /* Read mode */
	  cond = strcmp(mode, "read") == 0;
	  break;
	case 't': /* Stdout is a terminal device? */
	  cond = isatty (fileno (stdout));
	  break;
	default:
	  util_error(_("Valid if arguments are: s | r | t"));
	  return 1;
	}
    }
  _cond_push(cond);
  return 0;
}


int
mail_else (int argc MU_ARG_UNUSED, char **argv MU_ARG_UNUSED)
{
  int cond;

  if (_cond_level == 0)
    {
      /* TRANSLATORS: 'else' and 'if' are function names. Do not translate them */
      util_error(_("else without matching if"));
      return 1;
    }
  cond = _cond_pop();
  if (if_cond ())
    cond = !cond;
  _cond_push(cond);
  return 0;
}

int
mail_endif (int argc MU_ARG_UNUSED, char **argv MU_ARG_UNUSED)
{
  if (_cond_level == 0)
    {
      /* TRANSLATORS: 'endif' and 'if' are function names. Do not translate them */
      util_error(_("endif without matching if"));
      return 1;
    }
  _cond_pop();
  return 1;
}

