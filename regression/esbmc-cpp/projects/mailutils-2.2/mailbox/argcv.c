/* argcv.c - simple functions for parsing input based on whitespace
   Copyright (C) 1999, 2000, 2001, 2003, 2004, 2005, 2006, 2010 Free
   Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <ctype.h>
#include <errno.h>
#include <mailutils/argcv.h>

/* Keep mailutils namespace clean */
#define argcv_get            mu_argcv_get 
#define argcv_get_n          mu_argcv_get_n 
#define argcv_get_np         mu_argcv_get_np 
#define argcv_string         mu_argcv_string
#define argcv_free           mu_argcv_free
#define argv_free            mu_argv_free 
#define argcv_unquote_char   mu_argcv_unquote_char
#define argcv_quote_char     mu_argcv_quote_char  
#define argcv_quoted_length  mu_argcv_quoted_length
#define argcv_unquote_copy   mu_argcv_unquote_copy 
#define argcv_quote_copy     mu_argcv_quote_copy     

/*
 * takes a string and splits it into several strings, breaking at ' '
 * command is the string to split
 * the number of strings is placed into argc
 * the split strings are put into argv
 * returns 0 on success, nonzero on failure
 */

#define isws(c) ((c)==' '||(c)=='\t'||(c)=='\n')
#define isdelim(c,delim) (strchr(delim,(c))!=NULL)

struct argcv_info
{
  int len;
  const char *command;
  const char *delim;
  const char *comment;
  int flags;
  
  int start;
  int end;
  int save;
  int finish_pos;
};

static void
init_argcv_info (struct argcv_info *ap, int flags,
		 int len, const char *command, const char *delim,
		 const char *comment)
{
  memset (ap, 0, sizeof *ap);
  ap->len = len;
  ap->command = command;
  ap->delim = delim;
  ap->comment = comment;
  ap->flags = flags;
}

static int
argcv_scan (struct argcv_info *ap)
{
  int i = 0;
  int len = ap->len;
  const char *command = ap->command;
  const char *delim = ap->delim;
  const char *comment = ap->comment;
  
  for (;;)
    {
      i = ap->save;

      if (i >= len)
	return i + 1;

      /* Skip initial whitespace */
      while (i < len && isws (command[i]))
	i++;
      ap->start = i;

      if (!isdelim (command[i], delim))
	{
	  while (i < len)
	    {
	      if (command[i] == '\\')
		{
		  if (++i == len)
		    break;
		  i++;
		  continue;
		}
	      
	      if (command[i] == '\'' || command[i] == '"')
		{
		  int j;
		  for (j = i + 1; j < len && command[j] != command[i]; j++)
		    if (command[j] == '\\')
		      j++;
		  if (j < len)
		    i = j + 1;
		  else
		    i++;
		}
	      else if (isws (command[i]) || isdelim (command[i], delim))
		break;
	      else
		i++; /* skip the escaped character */
	    }
	  i--;
	}
      else if (!(ap->flags & MU_ARGCV_RETURN_DELIMS))
	{
	  while (i < len && isdelim (command[i], delim))
	    i++;
	  ap->save = i;
	  continue;
	}
      

      ap->end = i;
      ap->save = ap->finish_pos = i + 1;

      /* If we have a token, and it starts with a comment character, skip
         to the newline and restart the token search. */
      if (ap->save <= len)
	{
	  if (strchr (comment, command[ap->start]) != NULL)
	    {
	      ap->finish_pos = ap->start;
	      i = ap->save;
	      while (i < len && command[i] != '\n')
		i++;

	      ap->save = i;
	      continue;
	    }
	}
      break;
    }
  return ap->save;
}

static char quote_transtab[] = "\\\\\"\"a\ab\bf\fn\nr\rt\tv\v";

int
argcv_unquote_char (int c)
{
  char *p;

  for (p = quote_transtab; *p; p += 2)
    {
      if (*p == c)
	return p[1];
    }
  return c;
}

int
argcv_quote_char (int c)
{
  char *p;
  
  for (p = quote_transtab + sizeof(quote_transtab) - 2;
       p > quote_transtab; p -= 2)
    {
      if (*p == c)
	return p[-1];
    }
  return -1;
}
  
#define to_num(c) \
  (isdigit(c) ? c - '0' : (isxdigit(c) ? toupper(c) - 'A' + 10 : 255 ))

static int
xtonum (int *pval, const char *src, int base, int cnt)
{
  int i, val;
  
  for (i = 0, val = 0; i < cnt; i++, src++)
    {
      int n = *(unsigned char*)src;
      if (n > 127 || (n = to_num(n)) >= base)
	break;
      val = val*base + n;
    }
  *pval = val;
  return i;
}

size_t
argcv_quoted_length (const char *str, int *quote)
{
  size_t len = 0;

  *quote = 0;
  for (; *str; str++)
    {
      if (*str == ' ')
	{
	  len++;
	  *quote = 1;
	}
      else if (*str == '"')
	{
	  len += 2;
	  *quote = 1;
	}
      else if (*str != '\t' && *str != '\\' && isprint (*str))
	len++;
      else if (argcv_quote_char (*str) != -1)
	len += 2;
      else
	len += 4;
    }
  return len;
}

void
argcv_unquote_copy (char *dst, const char *src, size_t n)
{
  int i = 0;
  int c;
  int expect_delim = 0; 
    
  while (i < n)
    {
      switch (src[i])
	{
	case '\'':
	case '"':
	  if (!expect_delim)
	    {
	      const char *p;
	      
	      for (p = src+i+1; *p && *p != src[i]; p++)
		if (*p == '\\')
		  p++;
	      if (*p)
		expect_delim = src[i++];
	      else
		*dst++ = src[i++];
	    }
	  else if (expect_delim == src[i])
	    ++i;
	  else
	    *dst++ = src[i++];
	  break;
	  
	case '\\':
	  ++i;
	  if (src[i] == 'x' || src[i] == 'X')
	    {
	      if (n - i < 2)
		{
		  *dst++ = '\\';
		  *dst++ = src[i++];
		}
	      else 
		{
		  int off = xtonum(&c, src + i + 1, 16, 2);
		  if (off == 0)
		    {
		      *dst++ = '\\';
		      *dst++ = src[i++];
		    }
		  else
		    {
		      *dst++ = c;
		      i += off + 1;
		    }
		}
	    }
	  else if ((unsigned char)src[i] < 128 && isdigit (src[i]))
	    {
	      if (n - i < 1)
		{
		  *dst++ = '\\';
		  *dst++ = src[i++];
		}
	      else
		{
		  int off = xtonum (&c, src+i, 8, 3);
		  if (off == 0)
		    {
		      *dst++ = '\\';
		      *dst++ = src[i++];
		    }
		  else
		    {
		      *dst++ = c;
		      i += off;
		    }
		}
	    }
	  else
	    *dst++ = argcv_unquote_char (src[i++]);
	  break;
	  
	default:
	  *dst++ = src[i++];
	}
    }
  *dst = 0;
}

void
argcv_quote_copy (char *dst, const char *src)
{
  for (; *src; src++)
    {
      if (*src == '"')
	{
	  *dst++ = '\\';
	  *dst++ = *src;
	}
      else if (*src != '\t' && *src != '\\' && isprint(*src))
	*dst++ = *src;      
      else
	{
	  int c = argcv_quote_char (*src);
	  *dst++ = '\\';
	  if (c != -1)
	    *dst++ = c;
	  else
	    {
	      char tmp[4];
	      snprintf (tmp, sizeof tmp, "%03o", *(unsigned char*)src);
	      memcpy (dst, tmp, 3);
	      dst += 3;
	    }
	}
    }
}

int
argcv_get_np (const char *command, int len,
	      const char *delim, const char *cmnt,
	      int flags,
	      int *pargc, char ***pargv, char **endp)
{
  int i = 0;
  struct argcv_info info;
  int argc;
  char **argv;
  
  if (!delim)
    delim = "";
  if (!cmnt)
    cmnt = "";

  init_argcv_info (&info, flags, len, command, delim, cmnt);

  /* Count number of arguments */
  argc = 0;
  while (argcv_scan (&info) <= len)
    argc++;

  argv = calloc ((argc + 1), sizeof (char *));
  if (argv == NULL)
    return ENOMEM;
  
  i = 0;
  info.save = 0;
  for (i = 0; i < argc; i++)
    {
      int n;
      int unquote;
      
      argcv_scan (&info);

      if ((command[info.start] == '"' || command[info.end] == '\'')
	  && command[info.end] == command[info.start])
	{
	  if (info.start < info.end)
	    {
	      info.start++;
	      info.end--;
	    }
	  unquote = 0;
	}
      else
	unquote = 1;
      
      n = info.end - info.start + 1;
      argv[i] = calloc (n + 1,  sizeof (char));
      if (argv[i] == NULL)
	{
	  argcv_free (i, argv);
	  return ENOMEM;
	}
      if (unquote)
	argcv_unquote_copy (argv[i], &command[info.start], n);
      else
	memcpy (argv[i], &command[info.start], n);
      argv[i][n] = 0;
    }
  argv[i] = NULL;

  *pargc = argc;
  *pargv = argv;
  if (endp)
    *endp = (char*) (command + info.finish_pos);
  return 0;
}

int
argcv_get_n (const char *command, int len, const char *delim, const char *cmnt,
	     int *pargc, char ***pargv)
{
  return argcv_get_np (command, len, delim, cmnt, MU_ARGCV_RETURN_DELIMS,
		       pargc, pargv, NULL);
}

int
argcv_get (const char *command, const char *delim, const char *cmnt,
	   int *argc, char ***argv)
{
  return argcv_get_n (command, strlen (command), delim, cmnt, argc, argv);
}


/*
 * frees all elements of an argv array
 * argc is the number of elements
 * argv is the array
 */
void
argcv_free (int argc, char **argv)
{
  while (--argc >= 0)
    if (argv[argc])
      free (argv[argc]);
  free (argv);
}

void
argv_free (char **argv)
{
  int i;

  for (i = 0; argv[i]; i++)
    free (argv[i]);
  free (argv);
}

/* Make a argv an make string separated by ' '.  */

int
argcv_string (int argc, char **argv, char **pstring)
{
  size_t i, j, len;
  char *buffer;

  /* No need.  */
  if (pstring == NULL)
    return EINVAL;

  buffer = malloc (1);
  if (buffer == NULL)
    return ENOMEM;
  *buffer = '\0';

  for (len = i = j = 0; i < argc; i++)
    {
      int quote;
      int toklen;

      toklen = argcv_quoted_length (argv[i], &quote);
      
      len += toklen + 2;
      if (quote)
	len += 2;
      
      buffer = realloc (buffer, len);
      if (buffer == NULL)
        return ENOMEM;

      if (i != 0)
	buffer[j++] = ' ';
      if (quote)
	buffer[j++] = '"';
      argcv_quote_copy (buffer + j, argv[i]);
      j += toklen;
      if (quote)
	buffer[j++] = '"';
    }

  for (; j > 0 && isspace (buffer[j-1]); j--)
    ;
  buffer[j] = 0;
  if (pstring)
    *pstring = buffer;
  return 0;
}

void
mu_argcv_remove (int *pargc, char ***pargv,
		 int (*sel) (const char *, void *), void *data)
{
  int i, j;
  int argc = *pargc;
  char **argv = *pargv;
  int cnt = 0;
  
  for (i = j = 0; i < argc; i++)
    {
      if (sel (argv[i], data))
	{
	  free (argv[i]);
	  cnt++;
	}
      else
	{
	  if (i != j)
	    argv[j] = argv[i];
	  j++;
	}
    }
  if (i != j)
    argv[j] = NULL;
  argc -= cnt;

  *pargc = argc;
  *pargv = argv;
}
      
 
