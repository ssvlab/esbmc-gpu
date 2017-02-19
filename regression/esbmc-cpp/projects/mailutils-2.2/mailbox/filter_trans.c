/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2007, 2010 Free Software
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

/* Notes:
First Draft: Dave Inglis.
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mailutils/stream.h>
#include <mailutils/errno.h>

#include <filter0.h>

#define MU_TRANS_DECODE		1
#define MU_TRANS_ENCODE		2
#define MU_TRANS_BSIZE		2048


/* General-purpose implementation */
struct _trans_stream
{
  int t_offset; /* Orignal stream offset.  */

  size_t min_size;
  int s_offset;
  char *s_buf;     /* Used when read if not big enough to handle min_size
		      for decoder/encoder */

  int offset;      /* Current stream offset */
  int line_len;

  int w_rhd;       /* Working buffer read ahead  */
  int w_whd;       /* Working buffer write ahead */
  char w_buf[MU_TRANS_BSIZE]; /* working source/dest buffer */

  int (*transcoder) (const char *iptr, size_t isize, char *optr,
		     size_t osize, size_t *nbytes, int *line_len);
};

static void
trans_destroy (mu_filter_t filter)
{
  struct _trans_stream *ts = filter->data;
  if (ts->s_buf)
    free (ts->s_buf);
  free (ts);
}

static int
trans_read (mu_filter_t filter, char *optr, size_t osize, mu_off_t offset,
	    size_t *n_bytes)
{
  struct _trans_stream *ts = filter->data;
  size_t obytes, wbytes = 0, tbytes = 0;
  int ret = 0, i;
  size_t bytes, *nbytes = &bytes;

  if (optr == NULL)
    return MU_ERR_OUT_NULL;
  if (osize == 0)
    return EINVAL;

  if (n_bytes)
    nbytes = n_bytes;
  *nbytes = 0;

  if (offset && ts->t_offset != offset)
    return ESPIPE;

  if (offset == 0)
    ts->s_offset = ts->t_offset = ts->w_whd = ts->w_rhd =
      ts->offset = ts->line_len = 0;

  while (*nbytes < osize)
    {
      if ((ts->w_rhd + (int)ts->min_size) >= ts->w_whd)
	{
	  memmove (ts->w_buf, ts->w_buf + ts->w_rhd, ts->w_whd - ts->w_rhd);
	  ts->w_whd = ts->w_whd - ts->w_rhd;
	  ts->w_rhd = 0;
	  ret = mu_stream_read (filter->stream, ts->w_buf + ts->w_whd,
			     MU_TRANS_BSIZE - ts->w_whd, ts->offset,
			     &wbytes);
	  if (ret != 0)
	    break;
	  ts->offset += wbytes;
	  ts->w_whd += wbytes;
	}
      if ((osize - *nbytes) >= ts->min_size
	  && ts->s_offset == 0
	  && ts->w_whd - ts->w_rhd)
	{
	  tbytes = ts->transcoder (ts->w_buf + ts->w_rhd,
				   ts->w_whd - ts->w_rhd,
				   optr + *nbytes, osize - *nbytes,
				   &obytes, &ts->line_len);
	  ts->w_rhd += tbytes;
	  if (ts->w_rhd > ts->w_whd) /* over shot due to padding */
	    ts->w_rhd = ts->w_whd;
	  *nbytes += obytes;
	  ts->t_offset += obytes;
	}
      else
	{
	  if (ts->s_offset == 0 && ts->w_whd - ts->w_rhd)
	    {
	      tbytes = ts->transcoder (ts->w_buf + ts->w_rhd,
					   ts->w_whd - ts->w_rhd, ts->s_buf,
					   ts->min_size, &obytes,
					   &ts->line_len);
	      ts->w_rhd += tbytes;
	      if (ts->w_rhd > ts->w_whd) /* over shot due to padding */
		ts->w_rhd = ts->w_whd;
	      ts->s_offset = obytes;
	    }
	  for (i = ts->s_offset; i > 0; i--)
	    {
	      optr[(*nbytes)++] = ts->s_buf[ts->s_offset - i];
	      ts->t_offset++;
	      if (*nbytes >= osize)
		{
		  i--;
		  memmove (ts->s_buf, &ts->s_buf[ts->s_offset - i], i);
		  break;
		}
	    }
	  ts->s_offset = i;
	}
      /* FIXME: Should return error code if tbytes == 0? */
      if (wbytes == 0 && (tbytes == 0 || ts->w_whd - ts->w_rhd == 0))
	break;
    }
  return ret;
}


/*------------------------------------------------------
 * quoted-printable decoder/encoder
 *------------------------------------------------------*/
static const char _hexdigits[] = "0123456789ABCDEF";

#define QP_LINE_MAX	76
#define ISWS(c) ((c)==' ' || (c)=='\t')

static int
qp_decode (const char *iptr, size_t isize, char *optr, size_t osize,
	   size_t *nbytes, int *line_len MU_ARG_UNUSED)
{
  char c;
  int last_char = 0;
  size_t consumed = 0;
  size_t wscount = 0;
  
  *nbytes = 0;
  while (consumed < isize && *nbytes < osize)
    {
      c = *iptr++;

      if (ISWS (c))
	{
	  wscount++;
	  consumed++;
	}
      else
	{
	  /* Octets with values of 9 and 32 MAY be
	     represented as US-ASCII TAB (HT) and SPACE characters,
	     respectively, but MUST NOT be so represented at the end
	     of an encoded line.  Any TAB (HT) or SPACE characters
	     on an encoded line MUST thus be followed on that line
	     by a printable character. */
	  
	  if (wscount)
	    {
	      if (c != '\r' && c != '\n')
		{
		  size_t sz;
		  
		  if (consumed >= isize)
		    break;

		  if (*nbytes + wscount > osize)
		    sz = osize - *nbytes;
		  else
		    sz = wscount;
		  memcpy (optr, iptr - wscount - 1, sz);
		  optr += sz;
		  (*nbytes) += sz;
		  if (wscount > sz)
		    {
		      wscount -= sz;
		      break;
		    }
		}
	      wscount = 0;
	      if (*nbytes == osize)
		break;
	    }
		
	  if (c == '=')
	    {
	      /* There must be 2 more characters before I consume this.  */
	      if (consumed + 2 >= isize)
		break;
	      else
		{
		  /* you get =XX where XX are hex characters.  */
		  char 	chr[3];
		  int 	new_c;

		  chr[2] = 0;
		  chr[0] = *iptr++;
		  /* Ignore LF.  */
		  if (chr[0] != '\n')
		    {
		      chr[1] = *iptr++;
		      new_c = strtoul (chr, NULL, 16);
		      *optr++ = new_c;
		      (*nbytes)++;
		      consumed += 3;
		    }
		  else
		    consumed += 2;
		}
	    }
	  /* CR character.  */
	  else if (c == '\r')
	    {
	      /* There must be at least 1 more character before
		 I consume this.  */
	      if (consumed + 1 >= isize)
		break;
	      else
		{
		  iptr++; /* Skip the CR character.  */
		  *optr++ = '\n';
		  (*nbytes)++;
		  consumed += 2;
		}
	    }
	  else
	    {
	      *optr++ = c;
	      (*nbytes)++;
	      consumed++;
	    }
	}	  
      last_char = c;
    }
  return consumed - wscount;
}

static int
qp_encode (const char *iptr, size_t isize, char *optr, size_t osize,
	   size_t *nbytes, int *line_len)
{
  unsigned int c;
  size_t consumed = 0;

  *nbytes = 0;

  /* Strategy: check if we have enough room in the output buffer only
     once the required size has been computed. If there is not enough,
     return and hope that the caller will free up the output buffer a
     bit. */

  while (consumed < isize)
    {
      int simple_char;
      
      /* candidate byte to convert */
      c = *(unsigned char*) iptr;
      simple_char = (c >= 32 && c <= 60)
     	             || (c >= 62 && c <= 126)
	             || c == '\t'
	             || c == '\n';

      if (*line_len == QP_LINE_MAX
	  || (c == '\n' && consumed && ISWS (optr[-1]))
	  || (!simple_char && *line_len >= (QP_LINE_MAX - 3)))
	{
	  /* to cut a qp line requires two bytes */
	  if (*nbytes + 2 > osize) 
	    break;

	  *optr++ = '=';
	  *optr++ = '\n';
	  (*nbytes) += 2;
	  *line_len = 0;
	}
	  
      if (simple_char)
	{
	  /* a non-quoted character uses up one byte */
	  if (*nbytes + 1 > osize) 
	    break;

	  *optr++ = c;
	  (*nbytes)++;
	  (*line_len)++;

	  iptr++;
	  consumed++;

	  if (c == '\n')
	    *line_len = 0;
	}
      else
	{
	  /* a quoted character uses up three bytes */
	  if ((*nbytes) + 3 > osize) 
	    break;

	  *optr++ = '=';
	  *optr++ = _hexdigits[(c >> 4) & 0xf];
	  *optr++ = _hexdigits[c & 0xf];

	  (*nbytes) += 3;
	  (*line_len) += 3;

	  /* we've actuall used up one byte of input */
	  iptr++;
	  consumed++;
	}
    }
  return consumed;
}

static int
qp_init (mu_filter_t filter)
{
  struct _trans_stream *ts;
  ts = calloc (sizeof (*ts), 1);
  if (ts == NULL)
    return ENOMEM;

  ts->min_size = QP_LINE_MAX;
  ts->s_buf = calloc (ts->min_size, 1);
  if (ts->s_buf == NULL)
    {
      free (ts);
      return ENOMEM;
    }
  ts->transcoder = (filter->type == MU_FILTER_DECODE) ? qp_decode : qp_encode;

  filter->_read = trans_read;
  filter->_destroy = trans_destroy;
  filter->data = ts;
  return 0;
}

static struct _mu_filter_record _qp_filter =
{
  "quoted-printable",
  qp_init,
  NULL,
  NULL,
  NULL
};


/*------------------------------------------------------
 * base64 encode/decode
 *----------------------------------------------------*/

static int
b64_input (char c)
{
  const char table[64] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  int i;

  for (i = 0; i < 64; i++)
    {
      if (table[i] == c)
	return i;
    }
  return -1;
}

static int
base64_decode (const char *iptr, size_t isize, char *optr, size_t osize,
	       size_t *nbytes, int *line_len MU_ARG_UNUSED)
{
  int i = 0, tmp = 0, pad = 0;
  size_t consumed = 0;
  unsigned char data[4];

  *nbytes = 0;
  while (consumed < isize && (*nbytes)+3 < osize)
    {
      while (( i < 4 ) && (consumed < isize))
	{
	  tmp = b64_input (*iptr++);
	  consumed++;
	  if (tmp != -1)
	    data[i++] = tmp;
	  else if (*(iptr-1) == '=')
	    {
	      data[i++] = '\0';
	      pad++;
	    }
	}

      /* I have a entire block of data 32 bits get the output data.  */
      if (i == 4)
	{
	  *optr++ = (data[0] << 2) | ((data[1] & 0x30) >> 4);
	  *optr++ = ((data[1] & 0xf) << 4) | ((data[2] & 0x3c) >> 2);
	  *optr++ = ((data[2] & 0x3) << 6) | data[3];
	  (*nbytes) += 3 - pad;
	}
      else
	{
	  /* I did not get all the data.  */
	  consumed -= i;
	  return consumed;
	}
      i = 0;
    }
  return consumed;
}

static int
base64_encode_internal (const char *iptr, size_t isize,
			char *optr, size_t osize,
			size_t *nbytes, int *line_len, int line_max)
{
  size_t consumed = 0;
  int pad = 0;
  const char *b64 =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const unsigned char* ptr = (const unsigned char*) iptr;
	
  *nbytes = 0;
  if (isize <= 3)
    pad = 1;
  while (((consumed + 3) <= isize && (*nbytes + 4) <= osize) || pad)
    {
      if (line_max && *line_len == line_max)
	{
	  *optr++ = '\n';
	  (*nbytes)++;
	  (*line_len) = 0;
	  if ((*nbytes + 4) > osize)
	    return consumed;
	}
      *optr++ = b64[ptr[0] >> 2];
      *optr++ = b64[((ptr[0] << 4) + (--isize ? (ptr[1] >> 4): 0)) & 0x3f];
      *optr++ = isize ? b64[((ptr[1] << 2) + (--isize ? (ptr[2] >> 6) : 0 )) & 0x3f] : '=';
      *optr++ = isize ? b64[ptr[2] & 0x3f] : '=';
      ptr += 3;
      consumed += 3;
      (*nbytes) += 4;
      (*line_len) +=4;
      pad = 0;
    }
  return consumed;
}

static int
base64_encode (const char *iptr, size_t isize,
	       char *optr, size_t osize,
	       size_t *nbytes, int *line_len)
{
  return  base64_encode_internal (iptr, isize, optr, osize,
				  nbytes, line_len, 76);
}
    
static int
base64_init (mu_filter_t filter)
{
  struct _trans_stream *ts;
  ts = calloc (sizeof (*ts), 1);
  if (ts == NULL)
    return ENOMEM;

  ts->min_size = 4;
  ts->s_buf = calloc (ts->min_size, 1);
  if (ts->s_buf == NULL)
    {
      free (ts);
      return ENOMEM;
    }
  ts->transcoder = (filter->type == MU_FILTER_DECODE) ? base64_decode : base64_encode;

  filter->_read = trans_read;
  filter->_destroy = trans_destroy;
  filter->data = ts;
  return 0;
}

static struct _mu_filter_record _base64_filter =
{
  "base64",
  base64_init,
  NULL,
  NULL,
  NULL
};

static int
B_encode (const char *iptr, size_t isize,
	       char *optr, size_t osize,
	       size_t *nbytes, int *line_len)
{
  return  base64_encode_internal (iptr, isize, optr, osize,
				  nbytes, line_len, 0);
}

static int
B_init (mu_filter_t filter)
{
  struct _trans_stream *ts;
  ts = calloc (sizeof (*ts), 1);
  if (ts == NULL)
    return ENOMEM;

  ts->min_size = 4;
  ts->s_buf = calloc (ts->min_size, 1);
  if (ts->s_buf == NULL)
    {
      free (ts);
      return ENOMEM;
    }
  ts->transcoder = (filter->type == MU_FILTER_DECODE) ? base64_decode : B_encode;

  filter->_read = trans_read;
  filter->_destroy = trans_destroy;
  filter->data = ts;
  return 0;
}

static struct _mu_filter_record _B_filter =
{
  "B",
  B_init,
  NULL,
  NULL,
  NULL
};



/* RFC 2047 "Q" Encoding */
static int
Q_decode (const char *iptr, size_t isize, char *optr, size_t osize,
	   size_t *nbytes, int *line_len MU_ARG_UNUSED)
{
  char c;
  size_t consumed = 0;
  
  *nbytes = 0;
  while (consumed < isize && *nbytes < osize)
    {
      c = *iptr++;

      if (c == '=')
	{
	  /* There must be 2 more characters before I consume this.  */
	  if (consumed + 2 >= isize)
	    break;
	  else
	    {
	      /* you get =XX where XX are hex characters.  */
	      char chr[3];
	      int  new_c;
	      
	      chr[2] = 0;
	      chr[0] = *iptr++;
	      /* Ignore LF.  */
	      if (chr[0] != '\n')
		{
		  chr[1] = *iptr++;
		  new_c = strtoul (chr, NULL, 16);
		  *optr++ = new_c;
		  (*nbytes)++;
		  consumed += 3;
		}
	      else
		consumed += 2;
	    }
	}
      /* CR character.  */
      else if (c == '\r')
	{
	  /* There must be at least 1 more character before
	     I consume this.  */
	  if (consumed + 1 >= isize)
	    break;
	  else
	    {
	      iptr++; /* Skip the CR character.  */
	      *optr++ = '\n';
	      (*nbytes)++;
	      consumed += 2;
	    }
	}
      else if (c == '_')
	{
	  *optr++ = ' ';
	  (*nbytes)++;
	  consumed++;
	}
      else
	{
	  *optr++ = c;
	  (*nbytes)++;
	  consumed++;
	}
    }	  
  return consumed;
}

static int
Q_printable_char_p (unsigned c)
{
  switch (c)
    {
      /* FIXME: This is not quite so. Says RFC 2047:
	 
      (3) 8-bit values which correspond to printable ASCII characters other
      than "=", "?", and "_" (underscore), MAY be represented as those
      characters.  (But see section 5 for restrictions.)  In
      particular, SPACE and TAB MUST NOT be represented as themselves
      within encoded words. (see Page 6)*/
					
    case '=':
    case '?':
    case '_':
    case ' ':
    case '\t':
      return 0;
    default:
      return c > 32 && c < 127;
    }
}

static int
Q_encode (const char *iptr, size_t isize, char *optr, size_t osize,
	   size_t *nbytes, int *line_len)
{
  unsigned int c;
  size_t consumed = 0;

  *nbytes = 0;

  while (consumed < isize)
    {
      c = *(unsigned char*) iptr;
      if (Q_printable_char_p (c))
	{
	  /* a non-quoted character uses up one byte */
	  if (*nbytes + 1 > osize) 
	    break;
	  
	  *optr++ = c;
	  (*nbytes)++;
	  (*line_len)++;
	  
	  iptr++;
	  consumed++;
	}
      else if (c == 0x20)
	{
	  /* RFC2047, 4.2.2:
	     Note that the "_"
	     always represents hexadecimal 20, even if the SPACE character
	     occupies a different code position in the character set in use. */
	  *optr++ = '_';
	  (*nbytes)++;
	  (*line_len)++;
	  iptr++;
	  consumed++;
	}
      else 
	{
	  /* a quoted character uses up three bytes */
	  if ((*nbytes) + 3 > osize) 
	    break;

	  *optr++ = '=';
	  *optr++ = _hexdigits[(c >> 4) & 0xf];
	  *optr++ = _hexdigits[c & 0xf];

	  (*nbytes) += 3;
	  (*line_len) += 3;

	  /* we've actuall used up one byte of input */
	  iptr++;
	  consumed++;
	}
    }
  return consumed;
}

static int
Q_init (mu_filter_t filter)
{
  struct _trans_stream *ts;
  ts = calloc (sizeof (*ts), 1);
  if (ts == NULL)
    return ENOMEM;

  ts->min_size = QP_LINE_MAX;
  ts->s_buf = calloc (ts->min_size, 1);
  if (ts->s_buf == NULL)
    {
      free (ts);
      return ENOMEM;
    }
  ts->transcoder = (filter->type == MU_FILTER_DECODE) ? Q_decode : Q_encode;

  filter->_read = trans_read;
  filter->_destroy = trans_destroy;
  filter->data = ts;
  return 0;
}

static struct _mu_filter_record _Q_filter =
{
  "Q",
  Q_init,
  NULL,
  NULL,
  NULL
};


/* Pass-through encodings */

static struct _mu_filter_record _bit8_filter =
{
  "8bit",
  NULL,
  NULL,
  NULL,
  NULL
};

static struct _mu_filter_record _bit7_filter =
{
  "7bit",
  NULL,
  NULL,
  NULL,
  NULL
};

static struct _mu_filter_record _binary_filter =
{
  "binary",
  NULL,
  NULL,
  NULL,
  NULL
};



/* Export.  */
mu_filter_record_t mu_qp_filter = &_qp_filter;
mu_filter_record_t mu_base64_filter = &_base64_filter;
mu_filter_record_t mu_binary_filter = &_binary_filter;
mu_filter_record_t mu_bit8_filter = &_bit8_filter;
mu_filter_record_t mu_bit7_filter = &_bit7_filter;
mu_filter_record_t mu_rfc_2047_Q_filter = &_Q_filter;
mu_filter_record_t mu_rfc_2047_B_filter = &_B_filter;




