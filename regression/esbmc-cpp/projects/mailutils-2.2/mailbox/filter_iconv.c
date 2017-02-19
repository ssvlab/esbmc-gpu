/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2010 Free Software Foundation, Inc.

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
#include <mailutils/nls.h>

#ifdef HAVE_ICONV_H
# include <iconv.h>
#endif

#ifndef ICONV_CONST
# define ICONV_CONST
#endif

#ifndef HAVE_ICONV
# undef iconv_open
# define iconv_open(tocode, fromcode) ((iconv_t) -1)

# undef iconv
# define iconv(cd, inbuf, inbytesleft, outbuf, outbytesleft) (errno = EILSEQ, (size_t) -1)

# undef iconv_close
# define iconv_close(cd) 0

#endif 

enum _icvt_state
  {
    state_closed,           /* Filter is closed */
    state_open,             /* Filter is open and running in conversion mode */
    state_copy_pass,        /* Filter is open and running in copy-pass mode */
    state_copy_octal,       /* Filter is open and running in copy-octal mode */
    state_iconv_error,      /* A fatal iconv error has occurred */
    state_transport_error   /* A fatal transport error has occurred */
  };

struct icvt_stream
{
  mu_stream_t stream;   /* I/O stream */
  int fallback_mode;
  iconv_t cd;        /* Conversion descriptor */
  char *buf;         /* Conversion buffer */
  size_t bufsize;    /* Size of buf */
  size_t bufpos;     /* Current position in buf */
  enum _icvt_state state;
  int ec;            /* Error code */
  char errbuf[128];  /* Error message buffer */
};

static int
_icvt_open (mu_stream_t stream)
{
  struct icvt_stream *s = mu_stream_get_owner (stream);
  if (s->cd == (iconv_t) -1)
    return EINVAL;
  s->state = state_open;
  return 0;
}

static int
_icvt_close (mu_stream_t stream)
{
  struct icvt_stream *s = mu_stream_get_owner (stream);
  if (s->state != state_closed)
    {
      int flags;
      mu_stream_get_flags (stream, &flags);
      if (!(flags & MU_STREAM_NO_CLOSE))
	mu_stream_close (s->stream);
      iconv_close (s->cd);
      s->cd = (iconv_t) -1;
      s->state = state_closed;
    }
  return 0;
}

static void
_icvt_destroy (mu_stream_t stream)
{
  struct icvt_stream *s = mu_stream_get_owner (stream);
  int flags;

  if (s->state != state_closed)
    _icvt_close (stream);
  mu_stream_get_flags (stream, &flags);
  if (!(flags & MU_STREAM_NO_CLOSE))
    mu_stream_destroy (&s->stream, mu_stream_get_owner (s->stream));
  free (s->buf);
  s->buf = NULL;
  if (s->cd != (iconv_t) -1)
    iconv_close (s->cd);
  free (s);
}

static int _icvt_read (mu_stream_t stream, char *optr, size_t osize,
		       mu_off_t offset, size_t *pnbytes);

static int
internal_icvt_read (mu_stream_t stream, char *optr, size_t osize, size_t *pnbytes)
{
  struct icvt_stream *s = mu_stream_get_owner (stream);
  size_t nbytes = 0;
  int rc, status = 0;
  char *ob = optr;
  size_t olen = osize;

  if (s->bufpos == 0)
    {
      status = mu_stream_sequential_read (s->stream, s->buf, s->bufsize, &nbytes);
      if (status)
	{
	  s->state = state_transport_error;
	  s->ec = rc;
	  return MU_ERR_FAILURE;
	}
      else if (nbytes == 0)
	{
	  if (pnbytes)
	    *pnbytes = 0;
	  return 0;
	}
    }
  
  do
    {
      char ICONV_CONST *ib = s->buf;
      size_t inlen = s->bufpos + nbytes;
      
      rc = iconv (s->cd, &ib, &inlen, &ob, &olen);
      if (ib > s->buf)
	{
	  memmove (s->buf, ib, inlen);
	  s->bufpos = inlen;
	}
      else
	s->bufpos += nbytes;
      
      if (rc == -1)
	{
	  if (errno == E2BIG)
	    {
	      if (ob > optr)
		break;
	      else
		{
		  s->ec = MU_ERR_BUFSPACE;
		  return MU_ERR_BUFSPACE;
		}
	    }
	  else if (errno == EILSEQ)
	    {
	      int flags = 0; 
	      mu_stream_get_flags (stream, &flags);
	      switch (s->fallback_mode)
		{
		case mu_fallback_none:
		  s->state = state_iconv_error;
		  s->ec = errno;
		  if (ob == optr)
		    return MU_ERR_FAILURE;
		  break;

		case mu_fallback_copy_pass:
		  s->state = state_copy_pass;
		  if (ob == optr)
		    return _icvt_read (stream, optr, osize, 0, pnbytes);
		  break;
		  
		case mu_fallback_copy_octal:
		  s->state = state_copy_octal;
		  if (ob == optr)
		    return _icvt_read (stream, optr, osize, 0, pnbytes);
		  break;
		}
	    }
	  else if (errno == EINVAL)
	    {
	      if (inlen == s->bufsize)
		{
		  /* Try to reallocate temp buffer */
		  char *p = realloc (s->buf, s->bufsize + 128);
		  if (!p)
		    return ENOMEM;
		  s->buf = p;
		  s->bufsize += 128;
		}
	      continue;
	    }
	  else
	    {
	      s->ec = errno;
	      s->state = state_iconv_error;
	      return MU_ERR_FAILURE;
	    }
	}
    }
  while (olen > 0 
	 && (status = mu_stream_sequential_read (s->stream,
					      s->buf + s->bufpos,
					      s->bufsize - s->bufpos,
					      &nbytes)) == 0
	 && nbytes);

  if (status)
    {
      s->state = state_transport_error;
      s->ec = status;
      if (ob == optr)
	return MU_ERR_FAILURE;
    }
      
  if (pnbytes)
    *pnbytes = ob - optr;
  return 0;
}

#define ISPRINT(c) (((c)>=' '&&(c)<127)||c=='\n')

static int
copy_octal (struct icvt_stream *s, char *optr, size_t osize, size_t *pnbytes)
{
  size_t i, j;
  int status;
  
  if (osize == 0)
    return MU_ERR_BUFSPACE;
  
  if (s->bufpos < osize)
    {
      size_t rdcount = osize;
      if (s->bufsize < rdcount)
	{
	  /* Try to reallocate temp buffer */
	  char *p = realloc (s->buf, rdcount);
	  if (p)
	    {
	      s->bufsize = rdcount;
	      s->buf = p;
	    }
	  else
	    rdcount = s->bufsize;
	}

      status = mu_stream_sequential_read (s->stream,
				       s->buf + s->bufpos,
				       rdcount - s->bufpos,
				       &rdcount);
      if (status)
	{
	  s->state = state_transport_error;
	  s->ec = status;
	  if (s->bufpos == 0)
	    return MU_ERR_FAILURE;
	}
      else
	s->bufpos += rdcount;
    }
  
  for (i = j = 0; j < osize && i < s->bufpos; i++)
    {
      if (ISPRINT (*(unsigned char*)(s->buf+i)))
	optr[j++] = s->buf[i];
      else if (j + 4 >= osize)
	break;
      else
	{
	  sprintf (optr + j, "\\%03o", *(unsigned char*)(s->buf+i));
	  j += 4;
	}
    }
  s->bufpos -= i;
  memmove (s->buf, s->buf + i, s->bufpos);
  if (pnbytes)
    *pnbytes = j;
  return 0;
}

static int
copy_pass (struct icvt_stream *s, char *optr, size_t osize, size_t *pnbytes)
{
  int status;
  size_t nbytes;
  
  if (s->bufpos)
    {
      size_t sz = s->bufpos < osize ? s->bufpos : osize;
      memcpy (optr, s->buf, sz);
      s->bufpos -= sz;
      if (s->bufpos)
	memmove (s->buf, s->buf + sz, s->bufpos);
      if (pnbytes)
	*pnbytes = sz;
      return 0;
    }

  status = mu_stream_sequential_read (s->stream, optr, osize, &nbytes);
  if (status)
    {
      s->state = state_transport_error;
      s->ec = status;
      if (s->bufpos == 0)
	return MU_ERR_FAILURE;
    }
  if (pnbytes)
    *pnbytes = nbytes;
  return 0;
}

static int
_icvt_read (mu_stream_t stream, char *optr, size_t osize,
	    mu_off_t offset MU_ARG_UNUSED, size_t *pnbytes)
{
  struct icvt_stream *s = mu_stream_get_owner (stream);

  switch (s->state)
    {
    case state_open:
      return internal_icvt_read (stream, optr, osize, pnbytes);
      
    case state_closed:
      return EINVAL;
      
    case state_copy_pass:
      return copy_pass (s, optr, osize, pnbytes);
      
    case state_copy_octal:
      return copy_octal (s, optr, osize, pnbytes);
	
    default:
      break;
    }
  return MU_ERR_FAILURE;
}

int
_icvt_strerror (mu_stream_t stream, const char **pstr)
{
  struct icvt_stream *s = mu_stream_get_owner (stream);
  switch (s->state)
    {
    case state_transport_error:
      snprintf (s->errbuf, sizeof s->errbuf,
		_("Transport error: %s"), mu_strerror (s->ec));
      *pstr = s->errbuf;
      break;

    case state_iconv_error:
      switch (s->ec)
	{
	case EILSEQ:
	  snprintf (s->errbuf, sizeof s->errbuf,
		    _("Illegal multibyte sequence near %*.*s"),
		    (int) s->bufpos, (int) s->bufpos, s->buf);
	  break;

	default:
	  snprintf (s->errbuf, sizeof s->errbuf,
		    _("Iconv error: %s"), mu_strerror (s->ec));
	}
      *pstr = s->errbuf;
      break;

    case state_closed:
      *pstr = _("Stream is closed"); 
      break;
      
    default:
      *pstr = mu_strerror (s->ec);
    }
  
  return 0;
}

int
_icvt_get_transport2 (mu_stream_t stream,
		     mu_transport_t *pt, mu_transport_t *pt2)
{
  struct icvt_stream *s = mu_stream_get_owner (stream);
  if (pt)
    *pt = s->stream;
  if (*pt2)
    *pt2 = NULL;
  return 0;
}

int
_icvt_wait (mu_stream_t stream, int *pflags, struct timeval *tvp)
{
  struct icvt_stream *s = mu_stream_get_owner (stream);
  return mu_stream_wait (s->stream, pflags, tvp);
}

int
mu_filter_iconv_create (mu_stream_t *s, mu_stream_t transport,
		     const char *fromcode, const char *tocode, int flags,
		     enum mu_iconv_fallback_mode fallback_mode)
{
  struct icvt_stream *iptr;
  iconv_t cd;
  int rc;
  
  cd = iconv_open (tocode, fromcode);
  if (cd == (iconv_t) -1)
    return MU_ERR_FAILURE;
  
  iptr = malloc (sizeof *iptr);
  if (!iptr)
    return ENOMEM;
  iptr->stream = transport;
  iptr->fallback_mode = fallback_mode;
  iptr->cd = cd;
  iptr->state = state_closed;
  iptr->bufsize = 128;
  iptr->buf = malloc (iptr->bufsize);
  if (!iptr->buf)
    {
      free (iptr);
      return ENOMEM;
    }
  iptr->bufpos = 0;
  
  rc = mu_stream_create (s, flags, iptr);
  if (rc)
    {
      free (iptr);
      return rc;
    }
  mu_stream_set_open (*s, _icvt_open, iptr);
  mu_stream_set_close (*s, _icvt_close, iptr);
  mu_stream_set_read (*s, _icvt_read, iptr);
  mu_stream_set_destroy (*s, _icvt_destroy, iptr);
  mu_stream_set_strerror (*s, _icvt_strerror, iptr);
  mu_stream_set_get_transport2 (*s, _icvt_get_transport2, iptr);
  mu_stream_set_wait (*s, _icvt_wait, iptr);
  return 0;
}
