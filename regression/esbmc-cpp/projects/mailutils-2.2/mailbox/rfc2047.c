/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2004, 2005, 2006, 2007, 2009, 2010 Free Software
   Foundation, Inc.

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

#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <ctype.h>
#include <mailutils/stream.h>
#include <mailutils/filter.h>
#include <mailutils/errno.h>
#include <mailutils/mutil.h>

static int
realloc_buffer (char **bufp, size_t *bufsizep, size_t incr)
{
  size_t newsize = *bufsizep + incr;
  char *newp = realloc (*bufp, newsize);
  if (newp == NULL)
    return 1;
  *bufp = newp;
  *bufsizep = newsize;
  return 0;
}

int
getword (char **pret, const char **pstr, int delim)
{
  size_t len;
  char *ret;
  const char *start = *pstr;
  const char *end = strchr (start, delim);

  free (*pret);
  *pret = NULL;
  if (!end)
    return MU_ERR_BAD_2047_INPUT;
  len = end - start;
  ret = malloc (len + 1);
  if (!ret)
    return ENOMEM;
  memcpy (ret, start, len);
  ret[len] = 0;
  *pstr = end + 1;
  *pret = ret;
  return 0;
}
    
int
mu_rfc2047_decode (const char *tocode, const char *input, char **ptostr)
{
  int status = 0;
  const char *fromstr;
  char *buffer;
  size_t bufsize;
  size_t bufpos;
  size_t run_count = 0;
  char *fromcode = NULL;
  char *encoding_type = NULL;
  char *encoded_text = NULL;

#define BUFINC 128  
#define CHKBUF(count) do {                       \
  if (bufpos+count >= bufsize)                   \
    {                                            \
      size_t s = bufpos + count - bufsize;       \
      if (s < BUFINC)                            \
        s = BUFINC;                              \
      if (realloc_buffer (&buffer, &bufsize, s)) \
	{                                        \
	  free (buffer);                         \
          free (fromcode);                       \
          free (encoding_type);                  \
          free (encoded_text);                   \
	  return ENOMEM;                         \
	}                                        \
     }                                           \
 } while (0) 
  
  if (!input)
    return EINVAL;
  if (!ptostr)
    return MU_ERR_OUT_PTR_NULL;

  fromstr = input;

  /* Allocate the buffer. It is assumed that encoded string is always
     longer than it's decoded variant, so it's safe to use its length
     as the first estimate */
  bufsize = strlen (fromstr) + 1;
  buffer = malloc (bufsize);
  if (buffer == NULL)
    return ENOMEM;
  bufpos = 0;
  
  while (*fromstr)
    {
      if (strncmp (fromstr, "=?", 2) == 0)
	{
	  mu_stream_t filter = NULL;
	  mu_stream_t in_stream = NULL;
	  const char *filter_type = NULL;
	  size_t nbytes = 0, size;
	  const char *sp = fromstr + 2;
	  char tmp[128];
	  
	  status = getword (&fromcode, &sp, '?');
	  if (status)
	    break;
	  status = getword (&encoding_type, &sp, '?');
	  if (status)
	    break;
	  status = getword (&encoded_text, &sp, '?');
	  if (status)
	    break;
	  if (sp == NULL || sp[0] != '=')
	    {
	      status = MU_ERR_BAD_2047_INPUT;
	      break;
	    }
      
	  size = strlen (encoded_text);

	  switch (encoding_type[0])
	    {
            case 'b':
	    case 'B':
	      filter_type = "base64";
	      break;
	     
            case 'q': 
	    case 'Q':
	      filter_type = "Q";
	      break;

	    default:
	      status = MU_ERR_BAD_2047_INPUT;
	      break;
	    }
	  
	  if (status != 0)
	    break;

	  mu_memory_stream_create (&in_stream, 0, 0);
	  mu_stream_write (in_stream, encoded_text, size, 0, NULL);
	  status = mu_decode_filter (&filter, in_stream, filter_type, fromcode,
				     tocode);
	  if (status != 0)
	    break;

	  while (mu_stream_sequential_read (filter, tmp, sizeof (tmp),
					    &nbytes) == 0
		 && nbytes)
	    {
	      CHKBUF (nbytes);
	      memcpy (buffer + bufpos, tmp, nbytes);
	      bufpos += nbytes;
	    }

	  mu_stream_close (filter);
	  mu_stream_destroy (&filter, mu_stream_get_owner (filter));
	  
	  fromstr = sp + 1;
	  run_count = 1;
	}
      else if (run_count)
	{
	  if (*fromstr == ' ' || *fromstr == '\t')
	    {
	      run_count++;
	      fromstr++;
	      continue;
	    }
	  else
	    {
	      if (--run_count)
		{
		  CHKBUF (run_count);
		  memcpy (buffer + bufpos, fromstr - run_count, run_count);
		  bufpos += run_count;
		  run_count = 0;
		}
	      CHKBUF (1);
	      buffer[bufpos++] = *fromstr++;
	    }
	}
      else
	{
	  CHKBUF (1);
	  buffer[bufpos++] = *fromstr++;
	}
    }
  
  if (*fromstr)
    {
      size_t len = strlen (fromstr);
      CHKBUF (len);
      memcpy (buffer + bufpos, fromstr, len);
      bufpos += len;
    }

  CHKBUF (1);
  buffer[bufpos++] = 0;
  
  free (fromcode);
  free (encoding_type);
  free (encoded_text);

  if (status)
    free (buffer);
  else
    *ptostr = realloc (buffer, bufpos);
  return status;
}


/**
   Encode a header according to RFC 2047
   
   @param charset
     Charset of the text to encode
   @param encoding
     Requested encoding (must be "base64" or "quoted-printable")
   @param text
     Actual text to encode
   @param result [OUT]
     Encoded string

   @return 0 on success
*/
int
mu_rfc2047_encode (const char *charset, const char *encoding,
		   const char *text, char **result)
{
  mu_stream_t input_stream;
  mu_stream_t output_stream;
  int rc;
  
  if (charset == NULL || encoding == NULL || text == NULL)
    return EINVAL;

  if (strcmp (encoding, "base64") == 0)
    encoding = "B";
  else if (strcmp (encoding, "quoted-printable") == 0)
    encoding = "Q";
  else if (encoding[1] || !strchr ("BQ", encoding[0]))
    return MU_ERR_BAD_2047_ENCODING;

  rc = mu_memory_stream_create (&input_stream, 0, 0);
  if (rc)
    return rc;
  
  mu_stream_sequential_write (input_stream, text, strlen (text));

  rc = mu_filter_create (&output_stream, input_stream,
			 encoding, MU_FILTER_ENCODE, MU_STREAM_READ);
  if (rc == 0)
    {
      /* Assume strlen(qp_encoded_text) <= strlen(text) * 3 */
      /* malloced length is composed of:
	 "=?"  
	 charset 
	 "?"
	 B or Q
	 "?" 
	 encoded_text
	 "?="
	 zero terminator */
      
      *result = malloc (2 + strlen (charset) + 3 + strlen (text) * 3 + 3);
      if (*result)
	{
	  char *p = *result;
	  size_t s;
	  
	  p += sprintf (p, "=?%s?%s?", charset, encoding);
	  
	  rc = mu_stream_sequential_read (output_stream,
				       p,
				       strlen (text) * 3, &s);

	  strcpy (p + s, "?=");
	}
      else
	rc = ENOMEM;
      mu_stream_destroy (&output_stream, NULL);
    }
  else
    mu_stream_destroy (&input_stream, NULL);

  return rc;
}
