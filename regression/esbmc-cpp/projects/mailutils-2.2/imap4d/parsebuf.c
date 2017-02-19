/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

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

#include "imap4d.h"

static void
parsebuf_free (struct imap4d_parsebuf *p)
{
  if (p->peek_ptr)
    free (p->peek_ptr);
}
  
void
imap4d_parsebuf_exit (struct imap4d_parsebuf *p, char *text)
{
  p->err_text = text;
  longjmp (p->errjmp, 1);
}

static char *
pbcopy (const char *str, size_t len)
{
  char *p = malloc (len + 1);
  if (!p)
    imap4d_bye (ERR_NO_MEM);
  memcpy (p, str, len);
  p[len] = 0;
  return p;
}

static char *
pbcopy_char (int c)
{
  char ch = c;
  return pbcopy (&ch, 1);
}

char *
imap4d_parsebuf_peek (struct imap4d_parsebuf *p)
{
  if (!p->peek_ptr)
    {
      if (!p->tokptr || p->save_char == 0)
 	{
	  char *token = imap4d_tokbuf_getarg (p->tok, p->arg);
	  if (!token)
	    return NULL;
	  if (p->delim)
	    {
	      if (strchr (p->delim, token[0]))
		p->peek_ptr = pbcopy_char (token[0]);
	      else
		p->peek_ptr = pbcopy (token, strcspn (token, p->delim));
	    }
	  else
	    p->peek_ptr = pbcopy (token, strlen (token));
	}
      else
	{
	  char *token = p->token + p->tokoff;

	  if (strchr (p->delim, p->save_char))
	    p->peek_ptr = pbcopy_char (p->save_char);
	  else
	    {
	      size_t off = strcspn (token + 1, p->delim);
	      p->peek_ptr = pbcopy (token, off + 1);
	      p->peek_ptr[0] = p->save_char;
	    }
	}
    }
  return p->peek_ptr;
}

char *
imap4d_parsebuf_next (struct imap4d_parsebuf *p, int req)
{
  if (p->peek_ptr)
    {
      free (p->peek_ptr);
      p->peek_ptr = NULL;
    }

  if (!p->tokptr || p->save_char == 0)
    {
      p->tokptr = imap4d_tokbuf_getarg (p->tok, p->arg++);
      p->token = p->tokptr;
      p->tokoff = 0;
      if (!p->token)
	{
	  if (req)
	    imap4d_parsebuf_exit (p, "Too few arguments");
	  return NULL;
	}
      if (p->delim)
	{
	  if (strchr (p->delim, p->token[0]))
	    {
	      p->save_char = p->token[1];
	      p->tokoff = 1;
	      p->token[1] = 0;
	    }
	  else
	    {
	      p->tokoff = strcspn (p->token, p->delim);
	      if ((p->save_char = p->token[p->tokoff]))
		p->token[p->tokoff] = 0;
	    }
	}
      else
	p->save_char = 0;
    }
  else
    {
      p->token[p->tokoff] = p->save_char;
      p->token += p->tokoff;
      if (strchr (p->delim, p->save_char))
	{
	  p->save_char = p->token[1];
	  p->token[1] = 0;
	  p->tokoff = 1;
	}
      else
	{
	  p->tokoff = strcspn (p->token, p->delim);
	  if ((p->save_char = p->token[p->tokoff]))
	    p->token[p->tokoff] = 0;
	}
    }
  
  return p->token;
}

int
imap4d_with_parsebuf (imap4d_tokbuf_t tok, int arg, const char *delim,
		      int (*thunk) (imap4d_parsebuf_t), void *data,
		      char **err_text)
{
  struct imap4d_parsebuf pbuf;
  int rc;
  
  memset (&pbuf, 0, sizeof pbuf);
  pbuf.tok = tok;
  pbuf.arg = arg;
  pbuf.delim = delim;
  pbuf.data = data;
  pbuf.err_text = "Syntax error";
  pbuf.data = data;
  
  if (setjmp (pbuf.errjmp))
    {
      *err_text = pbuf.err_text;
      parsebuf_free (&pbuf);
      return RESP_BAD;
    }

  rc = thunk (&pbuf);
  parsebuf_free (&pbuf);
  return rc;
}
