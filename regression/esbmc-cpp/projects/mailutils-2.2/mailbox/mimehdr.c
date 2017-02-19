/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2007, 2009, 2010 Free
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
   Public License along with this library.  If not,
   see <http://www.gnu.org/licenses/>. */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <mailutils/cctype.h>
#include <mailutils/cstr.h>
#include <mailutils/errno.h>
#include <mailutils/message.h>
#include <mailutils/header.h>
#include <mailutils/stream.h>
#include <mailutils/url.h> /* FIXME: for mu_url_decode, which should
			      be renamed! */
#include <mailutils/mime.h>
#include <mailutils/filter.h>
#include <mailutils/mutil.h>

/* See RFC 2045, 5.1.  Syntax of the Content-Type Header Field */
#define _ISSPECIAL(c) !!strchr ("()<>@,;:\\\"/[]?=", c)

/* _header_get_param - an auxiliary function to extract values from
   Content-Type, Content-Disposition and similar headers.

   Arguments:
   
   FIELD_BODY    Header value, complying to RFCs 2045, 2183, 2231.3;
   DISP          Disposition.  Unless it is NULL, the disposition part
                 of FIELD_BODY is compared with it.  If they differ,
		 the function returns MU_ERR_NOENT.
   PARAM         Name of the parameter to extract from FIELD_BODY;
   BUF           Where to extract the value to;
   BUFSZ         Size of BUF;
   PRET          Pointer to the memory location for the return buffer (see
                 below).
   PLEN          Pointer to the return size.
   PFLAGS        On return, flags describing the parameter are stored there.
                 The MU_MIMEHDR_MULTILINE bit is set if the parameter value
		 was multiline (RFC 2231.3).  The MU_MIMEHDR_CSINFO bit is set
		 if the parameter value includes charset/language
		 information (RFC 2231.4).

   The function parses FIELD_BODY and extracts the value of the parameter
   PARAM.

   If BUF is not NULL and BUFSZ is not 0, the extracted value is stored into
   BUF.  At most BUFSZ-1 bytes are copied.

   Otherwise, if PRET is not NULL, the function allocates enough memory to
   hold the extracted value, copies there the result, and stores the
   pointer to the allocated memory into the location pointed to by PRET.

   If PLEN is not NULL, the size of the extracted value (without terminating
   NUL character) is stored there.

   If BUF==NULL *and* PRET==NULL, no memory is allocated, but PLEN is
   honored anyway, i.e. unless it is NULL it receives size of the result.
   This can be used to estimate the needed buffer size.

   Return values:
     0             on success.
     MU_ERR_NOENT, requested parameter not found, or disposition does
                   not match DISP.
     MU_ERR_PARSE, if FIELD_BODY does not comply to any of the abovemntioned
                   RFCs.
     ENOMEM      , if unable to allocate memory.
*/

/* Internal flag used by _header_get_param to delay increasing
   estimated continuation index. */
#define _MU_MIMEHDR_INCR_CIND 0x8000

int
_header_get_param (const char *field_body,
		   const char *disp,
		   const char *param,
		   char *buf, size_t bufsz,
		   char **pret, size_t *plen,
		   int *pflags)
{
  int res = MU_ERR_NOENT;            /* Return value, pessimistic default */
  size_t param_len = strlen (param);
  char *p;
  size_t size;
  char *mem = NULL;                  /* Allocated memory storage */
  size_t retlen = 0;                 /* Total number of bytes copied */
  unsigned long cind = 0;            /* Expected continued parameter index.
					See RFC 2231, Section 3,
					"Parameter Value Continuations" */
  int flags = 0;
  
  if (field_body == NULL)
    return EINVAL;

  if (bufsz == 0) /* Make sure buf value is meaningful */
    buf = NULL;
  
  p = strchr (field_body, ';');
  if (!p)
    return MU_ERR_NOENT;
  /* Allow for possible whitespace before the semicolon */
  for (size = p - field_body;
       size > 0 && mu_isblank (field_body[size-1]); size--)
    ;
  /* Remove surrounding quotes.
     FIXME: unescape the quoted contents. */
  if (field_body[0] == '"' && field_body[size-1] == '"')
    {
      field_body++;
      size -= 2;
    }
  if (disp && mu_c_strncasecmp (field_body, disp, size))
    return MU_ERR_NOENT;
      
  while (p && *p)
    {
      char *v, *e, *ep, *cp;
      size_t len, escaped_chars = 0;
      
      if (*p != ';')
	{
	  res = MU_ERR_PARSE;
	  break;
	}
      
      /* walk upto start of param */      
      p = mu_str_skip_class (p + 1, MU_CTYPE_SPACE);

      /* Reportedly, some MUAs insert several semicolons */
      if (*p == ';')
	continue;

      /* Ignore stray characters */
      if (_ISSPECIAL (*p))
	{
	  p = strchr (p, ';');
	  continue;
	}
	
      if ((ep = strchr (p, '=')) == NULL)
	break;
      /* Allow for optional whitespace after '=' */
      v = mu_str_skip_class (ep + 1, MU_CTYPE_SPACE);
      /* Find end of the parameter */
      if (*v == '"')
	{
	  /* Quoted string */
	  for (e = ++v; *e != '"'; e++)
	    {
	      if (*e == 0) /* Malformed header */
		{
		  res = MU_ERR_PARSE;
		  break;
		}
	      if (*e == '\\')
		{
		  if (*++e == 0)
		    {
		      res = MU_ERR_PARSE;
		      break;
		    }
		  escaped_chars++;
		}
	    }
	  if (res == MU_ERR_PARSE)
	    break;
	  len = e - v;
	  e++;
	}
      else
	{
	  for (e = v + 1; *e && !(*e == ';' || mu_isspace (*e)); e++)
	    ;
	  len = e - v;
	}

      /* Is it our parameter? */
      if (mu_c_strncasecmp (p, param, param_len))
	{			/* nope, jump to next */
	  p = strchr (e, ';');
	  continue;
	}

      cp = p + param_len;

      if (*cp == '*')
	{
	  cp++;
	  /* It is a parameter value continuation (RFC 2231, Section 3)
	     or parameter value character set and language information
	     (ibid., Section 4). */
	  if (mu_isdigit (*cp))
	    {
	      /* See if the index is OK */
	  
	      char *end;
	      unsigned long n = strtoul (cp, &end, 10);

	      if (*end == '*')
		{
		  flags |= MU_MIMEHDR_CSINFO;
		  end++;
		}
	      if (n != cind)
		{
		  res = MU_ERR_PARSE;
		  break;
		}
	      /* Everything OK, mark this as a multiline (continued)
		 parameter. We also need to increment the estimation,
		 but it cannot be done right now because its value is
		 used below to decide whether to do flag cleanup
		 on error. So we set _MU_MIMEHDR_INCR_CIND flag instead
		 and increment cind later. */
	      flags |= (MU_MIMEHDR_MULTILINE|_MU_MIMEHDR_INCR_CIND);
	      /* And point cp to the last character: there are more
		 checks ahead. */
	      cp = end;
	    }
	  else
	    flags |= MU_MIMEHDR_CSINFO;
	}
      /* Allow for optional whitespace before '=' */
      cp = mu_str_skip_class (cp, MU_CTYPE_SPACE);
      /* cp must now point to the equals sign */
      if (cp != ep)
	{
	  /* Clean up everything, unless we're in the middle of a
	     parameter continuation. */
	  if (cind == 0)
	    flags = 0;

	  /* Try next parameter */
	  p = strchr (e, ';');
	  continue;
	}

      if (flags & _MU_MIMEHDR_INCR_CIND)
	{
	  /* Increase the estimation. */
	  flags &= ~_MU_MIMEHDR_INCR_CIND;
	  cind++;
	}
      
      res = 0; /* Indicate success */
      
      /* Prepare P for the next iteration */
      p = e;

      /* Escape characters that appear in quoted-pairs are
	 semantically "invisible" (RFC 2822, Section 3.2.2,
	 "Quoted characters") */
      len -= escaped_chars;

      /* Adjust len if nearing end of the buffer */
      if (bufsz && len >= bufsz)
	len = bufsz - 1;

      if (pret)
	{
	  /* The caller wants us to allocate the memory */
	  if (!buf && !mem)
	    {
	      mem = malloc (len + 1);
	      if (!mem)
		{
		  res = ENOMEM;
		  break;
		}
	      buf = mem;
	    }
	  else if (mem)
	    {
	      /* If we got here, it means we are iterating over
		 a parameter value continuation, and cind=0 has
		 already been passed.  Reallocate the memory to
		 accomodate next chunk of data. */
	      char *newmem = realloc (mem, retlen + len + 1);
	      if (!newmem)
		{
		  res = ENOMEM;
		  break;
		}
	      buf = mem = newmem;
	    }
	}

      if (buf)
	{
	  /* Actually copy the data.  Buf is not NULL either because
	     the user passed it as an argument, or because we allocated
	     memory for it. */
	  if (escaped_chars)
	    {
	      int i;
	      for (i = 0; i < len; i++)
		{
		  if (*v == '\\')
		    ++v;
		  buf[retlen + i] = *v++;
		}
	    }
	  else
	    memcpy (buf + retlen, v, len);
	}
      /* Adjust total result size ... */
      retlen += len;
      /* ... and remaining buffer size, if necessary */
      if (bufsz)
	{
	  bufsz -= len;
	  if (bufsz == 0)
	    break;
	}
    }

  if (res == 0)
    {
      /* Everything OK, prepare the returned data. */
      if (buf)
	buf[retlen] = 0;
      if (plen)
	*plen = retlen;
      if (pret)
	*pret = mem;
      if (pflags)
	*pflags = flags;
    }
  else if (mem)
    free (mem);
  return res;
}

static size_t
disp_segment_len (const char *str)
{
  char *p = strchr (str, ';');
  size_t size;
  
  if (!p)
    size = strlen (str);
  else
    size = p - str;
  while (size > 0 && mu_isblank (str[size-1]))
    size--;
  return size;
}

/* STR is a value of a structured MIME header, e.g. Content-Type.
   This function returns the `disposition part' of it.  In other
   words, it returns disposition, if STR is a Content-Disposition
   value, and `type/subtype' part, if it is a Content-Type value.
*/
int
mu_mimehdr_get_disp (const char *str, char *buf, size_t bufsz, size_t *retsz)
{
  size_t size;

  str = mu_str_skip_class (str, MU_CTYPE_BLANK);
  size = disp_segment_len (str);
  if (size > 2 && str[0] == '"' && str[size-1] == '"')
    {
      str++;
      size -= 2;
    }
  if (buf)
    size = mu_cpystr (buf, str, size);
  if (retsz)
    *retsz = size;
  return 0;
}

/* Same as mu_mimehdr_get_disp, but allocates memory */
int
mu_mimehdr_aget_disp (const char *str, char **pvalue)
{
  char *p;
  size_t size;
  
  str = mu_str_skip_class (str, MU_CTYPE_BLANK);
  size = disp_segment_len (str);
  if (size > 2 && str[0] == '"' && str[size-1] == '"')
    {
      str++;
      size -= 2;
    }

  p = malloc (size + 1);
  if (!p)
    return ENOMEM;
  memcpy (p, str, size);
  p[size] = 0;
  *pvalue = p;
  return 0;
}

/* Get the value of the parameter PARAM from STR, which must be
   a value of a structured MIME header.
   At most BUFSZ-1 of data are stored in BUF.  A terminating NUL
   character is appended to it.
   
   Unless NULL, RETSZ is filled with the actual length of the
   returned data (not including the NUL terminator).

   Unless PFLAGS is null it will contain, on return, the flags describing
   the parameter.  The MU_MIMEHDR_MULTILINE bit is set if the parameter value
   was multiline (RFC 2231.3).  The MU_MIMEHDR_CSINFO bit is set if the
   parameter value includes charset/language information (RFC 2231.4).
   
   BUF may be NULL, in which case the function will only fill
   RETSZ and PFLAGS, as described above. */
int
mu_mimehdr_get_param (const char *str, const char *param,
		     char *buf, size_t bufsz, size_t *retsz,
		     int *pflags)
{
  return _header_get_param (str, NULL, param, buf, bufsz, NULL, retsz,
			    pflags);
}

/* Same as mu_mimehdr_get_param, but allocates memory. */
int
mu_mimehdr_aget_param (const char *str, const char *param,
		      char **pval, int *pflags)
{
  return _header_get_param (str, NULL, param, NULL, 0, pval, NULL, pflags);
}

/* Decode a parameter value.  Arguments:

   Input:
   VALUE        Parameter value.
   FLAGS        Flags obtained from a previous call to one of the functions
                above.
   CHARSET      Output charset.

   Output:
   PVAL         A pointer to the decoded value is stored there.
                The memory is allocated using malloc.
   PLANG        If language information was present in VALUE, its
                malloc'ed copy is stored in the memory location pointed
		to by this variable.  If there was no language information,
		*PLANG is set to NULL. 

   Both PVAL and PLANG may be NULL if that particular piece of information
   is not needed. */
int
mu_mimehdr_decode_param (const char *value, int flags,
			 const char *charset, char **pval, char **plang)
{
  char *decoded;
  int rc;
  char *lang = NULL;
  char *data;

  if (flags == 0)
    {
      rc = mu_rfc2047_decode (charset, value, &decoded);
      if (rc)
	return rc;
    }
  else
    {
      decoded = mu_url_decode (value);
      if (!decoded)
	return ENOMEM;
  
      if ((flags & MU_MIMEHDR_CSINFO)
	  && (lang = strchr (decoded, '\''))
	  && (data = strchr (lang + 1, '\'')))
	{
	  char *source_cs = decoded;

	  *lang++ = 0;
	  *data++ = 0;

	  lang = lang[0] ? strdup (lang) : NULL;
      
	  if (source_cs[0] && charset && mu_c_strcasecmp (source_cs, charset))
	    {
	      char *outval = NULL;
	      mu_stream_t instr = NULL;
	      mu_stream_t outstr = NULL;
	      mu_stream_t cvt = NULL;
	      char iobuf[512];
	  
	      do
		{
		  size_t total = 0, pos;
		  size_t nbytes;

		  rc = mu_memory_stream_create (&instr, 0, 0);
		  if (rc)
		    break;
		  rc = mu_stream_write (instr, data, strlen (data), 0, NULL);
		  if (rc)
		    break;

		  rc = mu_memory_stream_create (&outstr, 0, 0);
		  if (rc)
		    break;
		  
		  rc = mu_filter_iconv_create (&cvt, instr, source_cs, charset,
					       MU_STREAM_NO_CLOSE,
					       mu_default_fallback_mode);
		  if (rc)
		    break;
		  
		  rc = mu_stream_open (cvt);
		  if (rc)
		    break;

		  while (mu_stream_sequential_read (cvt, iobuf, sizeof (iobuf),
						    &nbytes) == 0
			 && nbytes)
		    {
		      rc = mu_stream_sequential_write (outstr, iobuf, nbytes);
		      if (rc)
			break;
		      total += nbytes;
		    }
		  
		  if (rc)
		    break;
		  
		  outval = malloc (total + 1);
		  if (!outval)
		    {
		      rc = ENOMEM;
		      break;
		    }

		  mu_stream_seek (outstr, 0, SEEK_SET);
		  pos = 0;
		  while (mu_stream_sequential_read (outstr, outval + pos,
						    total - pos, &nbytes) == 0
			 && nbytes)
		    pos += nbytes;
		  outval[pos] = 0;
		}
	      while (0);
	      
	      mu_stream_close (cvt);
	      mu_stream_destroy (&cvt, mu_stream_get_owner (cvt));
	      mu_stream_close (instr);
	      mu_stream_destroy (&instr, mu_stream_get_owner (instr));
	      mu_stream_close (outstr);
	      mu_stream_destroy (&outstr, mu_stream_get_owner (outstr));
	      
	      free (decoded);
	  
	      if (rc)
		{
		  /* Cleanup after an error. */
		  free (lang);
		  free (outval);
		  return rc;
		}
	      decoded = outval;
	    }
	  else
	    memmove (decoded, data, strlen (data) + 1);
	}
    }
  
  if (pval)
    *pval = decoded;
  else
    free (decoded);
  
  if (plang)
    *plang = lang;
  return 0;
}

/* Similar to mu_mimehdr_aget_param, but the returned value is decoded
   according to the CHARSET.  Unless PLANG is NULL, it receives malloc'ed
   language name from STR.  If there was no language name, *PLANG is set
   to NULL. 
*/
int
mu_mimehdr_aget_decoded_param (const char *str, const char *param,
			       const char *charset, 
			       char **pval, char **plang)
{
  char *value;
  int rc;
  int flags;
  
  rc = mu_mimehdr_aget_param (str, param, &value, &flags);
  if (rc == 0)
    {
      rc = mu_mimehdr_decode_param (value, flags, charset, pval, plang);
      free (value);
    }
  return rc;
}
  
/* Get the attachment name from MSG.  See _header_get_param, for a
   description of the rest of arguments. */
static int
_get_attachment_name (mu_message_t msg, char *buf, size_t bufsz,
		      char **pbuf, size_t *sz, int *pflags)
{
  int ret = EINVAL;
  mu_header_t hdr;
  char *value = NULL;

  if (!msg)
    return ret;

  if ((ret = mu_message_get_header (msg, &hdr)) != 0)
    return ret;

  ret = mu_header_aget_value_unfold (hdr, "Content-Disposition", &value);

  /* If the header wasn't there, we'll fall back to Content-Type, but
     other errors are fatal. */
  if (ret != 0 && ret != MU_ERR_NOENT)
    return ret;

  if (ret == 0 && value != NULL)
    {
      ret = _header_get_param (value, "attachment",
			       "filename", buf, bufsz, pbuf, sz, pflags);
      free (value);
      value = NULL;
      if (ret == 0 || ret != MU_ERR_NOENT)
	return ret;
    }

  /* If we didn't get the name, we fall back on the Content-Type name
     parameter. */

  free (value);
  ret = mu_header_aget_value_unfold (hdr, "Content-Type", &value);
  if (ret == 0)
    ret = _header_get_param (value, NULL, "name", buf, bufsz, pbuf, sz,
			     pflags);
  free (value);

  return ret;
}

int
mu_message_aget_attachment_name (mu_message_t msg, char **name, int *pflags)
{
  if (name == NULL)
    return MU_ERR_OUT_PTR_NULL;
  return _get_attachment_name (msg, NULL, 0, name, NULL, pflags);
}

int
mu_message_aget_decoded_attachment_name (mu_message_t msg,
					 const char *charset,
					 char **pval,
					 char **plang)
{
  char *value;
  int flags;
  int rc = mu_message_aget_attachment_name (msg, &value, &flags);
  if (rc == 0)
    {
      rc = mu_mimehdr_decode_param (value, flags, charset, pval, plang);
      free (value);
    }
  return rc;
}

int
mu_message_get_attachment_name (mu_message_t msg, char *buf, size_t bufsz,
				size_t *sz, int *pflags)
{
  return _get_attachment_name (msg, buf, bufsz, NULL, sz, pflags);
}

