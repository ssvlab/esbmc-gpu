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
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef HAVE_LIBGEN_H
# include <libgen.h>
#endif

#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/cctype.h>
#include <mailutils/cstr.h>
#include <mailutils/body.h>
#include <mailutils/filter.h>
#include <mailutils/header.h>
#include <mailutils/message.h>
#include <mailutils/stream.h>
#include <mailutils/errno.h>
#include <mailutils/mutil.h>

#define BUF_SIZE	2048

struct _mu_mime_io_buffer
{
  unsigned int refcnt;
  char *buf;
  size_t bufsize;
  size_t nbytes;
  char *charset;
  mu_header_t hdr;
  mu_message_t msg;
  size_t ioffset;
  size_t ooffset;
  mu_stream_t stream;	/* output file/decoding stream for saving attachment */
  mu_stream_t fstream;	/* output file stream for saving attachment */
};

#define MSG_HDR "Content-Type: %s; name=%s\nContent-Transfer-Encoding: %s\nContent-Disposition: attachment; filename=%s\n\n"

int
mu_message_create_attachment (const char *content_type, const char *encoding,
			      const char *filename, mu_message_t *newmsg)
{
  mu_header_t hdr;
  mu_body_t body;
  mu_stream_t fstream = NULL, tstream = NULL;
  char *header = NULL, *name = NULL, *fname = NULL;
  int ret;

  if (newmsg == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (filename == NULL)
    return EINVAL;

  if ((ret = mu_message_create (newmsg, NULL)) == 0)
    {
      if (content_type == NULL)
	content_type = "text/plain";
      if (encoding == NULL)
	encoding = "7bit";
      if ((fname = strdup (filename)) != NULL)
	{
	  name = strrchr (fname, '/');
	  if (name)
	    name++;
	  else
	    name = fname;
	  if ((header =
	       malloc (strlen (MSG_HDR) + strlen (content_type) +
		       strlen (name) * 2 + strlen (encoding) + 1)) == NULL)
	    ret = ENOMEM;
	  else
	    {
	      sprintf (header, MSG_HDR, content_type, name, encoding, name);
	      if ((ret =
		   mu_header_create (&hdr, header, strlen (header),
				     *newmsg)) == 0)
		{
		  mu_message_get_body (*newmsg, &body);
		  if ((ret =
		       mu_file_stream_create (&fstream, filename,
					      MU_STREAM_READ)) == 0)
		    {
		      if ((ret = mu_stream_open (fstream)) == 0)
			{
			  if ((ret =
			       mu_filter_create (&tstream, fstream, encoding,
						 MU_FILTER_ENCODE,
						 MU_STREAM_READ)) == 0)
			    {
			      mu_body_set_stream (body, tstream, *newmsg);
			      mu_message_set_header (*newmsg, hdr, NULL);
			    }
			}
		    }
		}
	      free (header);
	    }
	}
    }
  if (ret)
    {
      if (*newmsg)
	mu_message_destroy (newmsg, NULL);
      if (hdr)
	mu_header_destroy (&hdr, NULL);
      if (fstream)
	mu_stream_destroy (&fstream, NULL);
      if (fname)
	free (fname);
    }
  return ret;
}

int
mu_mime_io_buffer_create (mu_mime_io_buffer_t *pinfo)
{
  mu_mime_io_buffer_t info;
  
  if ((info = calloc (1, sizeof (*info))) == NULL)
    return ENOMEM;
  info->refcnt = 1;
  info->bufsize = BUF_SIZE;
  *pinfo = info;
  return 0;
}

void
mu_mime_io_buffer_set_size (mu_mime_io_buffer_t info, size_t size)
{
  info->bufsize = size;
}

void
mu_mime_io_buffer_get_size (mu_mime_io_buffer_t info, size_t *psize)
{
  *psize = info->bufsize;
}

int
mu_mime_io_buffer_set_charset (mu_mime_io_buffer_t info, const char *charset)
{
  char *cp = strdup (charset);
  if (!cp)
    return ENOMEM;
  free (info->charset);
  info->charset = cp;
  return 0;
}

void
mu_mime_io_buffer_sget_charset (mu_mime_io_buffer_t info, const char **charset)
{
  *charset = info->charset;
}

int
mu_mime_io_buffer_aget_charset (mu_mime_io_buffer_t info, const char **charset)
{
  *charset = strdup (info->charset);
  if (!charset)
    return ENOMEM;
  return 0;
}

void
mu_mime_io_buffer_destroy (mu_mime_io_buffer_t *pinfo)
{
  if (pinfo && *pinfo)
    {
      mu_mime_io_buffer_t info = *pinfo;
      free (info->charset);
      free (info->buf);
      free (info);
      *pinfo = NULL;
    }
}

static void
_attachment_free (struct _mu_mime_io_buffer *info, int free_message)
{
  if (free_message)
    {
      if (info->msg)
	mu_message_destroy (&info->msg, NULL);
      else if (info->hdr)
	mu_header_destroy (&info->hdr, NULL);
    }
  info->msg = NULL;
  info->hdr = NULL;
  info->ioffset = 0;
  info->ooffset = 0;
  info->stream = NULL;
  info->fstream = NULL;
  if (--info->refcnt == 0)
    {
      free (info->charset);
      free (info->buf);
      free (info);
    }
}

static int
_attachment_setup (mu_mime_io_buffer_t *pinfo, mu_message_t msg,
		   mu_stream_t *stream)
{
  int ret;
  mu_body_t body;
  mu_mime_io_buffer_t info;
  
  if ((ret = mu_message_get_body (msg, &body)) != 0 ||
      (ret = mu_body_get_stream (body, stream)) != 0)
    return ret;
  if (*pinfo)
    {
      info = *pinfo;
      info->refcnt++;
    }
  else
    {
      ret = mu_mime_io_buffer_create (&info);
      if (ret)
	return ret;
    }
  
  if (!info->buf && ((info->buf = malloc (info->bufsize)) == NULL))
    {
      _attachment_free (info, 0);
      return ENOMEM;
    }
  *pinfo = info;
  return 0;
}

int
mu_message_save_attachment (mu_message_t msg, const char *filename,
			    mu_mime_io_buffer_t info)
{
  mu_stream_t istream;
  int ret;
  size_t size;
  size_t nbytes;
  mu_header_t hdr;
  const char *fname = NULL;
  char *partname = NULL;

  if (msg == NULL)
    return EINVAL;

  if ((ret = _attachment_setup (&info, msg, &istream)) != 0)
    return ret;

  if (ret == 0 && (ret = mu_message_get_header (msg, &hdr)) == 0)
    {
      if (filename == NULL)
	{
	  ret = mu_message_aget_decoded_attachment_name (msg, info->charset,
							 &partname, NULL);
	  if (partname)
	    fname = partname;
	}
      else
	fname = filename;
      if (fname
	  && (ret =
	      mu_file_stream_create (&info->fstream, fname,
				     MU_STREAM_WRITE | MU_STREAM_CREAT)) == 0)
	{
	  if ((ret = mu_stream_open (info->fstream)) == 0)
	    {
	      char *content_encoding;
	      char *content_encoding_mem = NULL;

	      mu_header_get_value (hdr, "Content-Transfer-Encoding", NULL, 0,
				   &size);
	      if (size)
		{
		  content_encoding_mem = malloc (size + 1);
		  if (content_encoding_mem == NULL)
		    ret = ENOMEM;
		  content_encoding = content_encoding_mem;
		  mu_header_get_value (hdr, "Content-Transfer-Encoding",
				       content_encoding, size + 1, 0);
		}
	      else
		content_encoding = "7bit";
	      ret =
		mu_filter_create (&info->stream, istream, content_encoding,
				  MU_FILTER_DECODE,
				  MU_STREAM_READ | MU_STREAM_NO_CLOSE);
	      free (content_encoding_mem);
	    }
	}
    }
  if (info->stream && istream)
    {
      if (info->nbytes)
	memmove (info->buf, info->buf + (info->bufsize - info->nbytes),
		 info->nbytes);
      while ((ret == 0 && info->nbytes)
	     ||
	     ((ret =
	       mu_stream_read (info->stream, info->buf, info->bufsize,
			       info->ioffset, &info->nbytes)) == 0
	      && info->nbytes))
	{
	  info->ioffset += info->nbytes;
	  while (info->nbytes)
	    {
	      if ((ret =
		   mu_stream_write (info->fstream, info->buf, info->nbytes,
				    info->ooffset, &nbytes)) != 0)
		break;
	      info->nbytes -= nbytes;
	      info->ooffset += nbytes;
	    }
	}
    }
  if (ret != EAGAIN && info)
    {
      mu_stream_close (info->fstream);
      mu_stream_destroy (&info->stream, NULL);
      mu_stream_destroy (&info->fstream, NULL);
    }

  _attachment_free (info, ret); /* FIXME: or 0? */
  
  /* Free fname if we allocated it. */
  if (partname)
    free (partname);

  return ret;
}

int
mu_message_encapsulate (mu_message_t msg, mu_message_t *newmsg,
			mu_mime_io_buffer_t info)
{
  mu_stream_t istream, ostream;
  const char *header;
  int ret = 0;
  size_t nbytes;
  mu_body_t body;

  if (msg == NULL)
    return EINVAL;
  if (newmsg == NULL)
    return MU_ERR_OUT_PTR_NULL;

  if ((ret = _attachment_setup (&info, msg, &ostream)) != 0)
    return ret;

  if (info->msg == NULL
      && (ret = mu_message_create (&info->msg, NULL)) == 0)
    {
      header =
	"Content-Type: message/rfc822\nContent-Transfer-Encoding: 7bit\n\n";
      if ((ret =
	   mu_header_create (&info->hdr, header, strlen (header),
			     msg)) == 0)
	ret = mu_message_set_header (info->msg, info->hdr, NULL);
    }
  if (ret == 0 && (ret = mu_message_get_stream (msg, &istream)) == 0)
    {
      if ((ret = mu_message_get_body (info->msg, &body)) == 0 &&
	  (ret = mu_body_get_stream (body, &ostream)) == 0)
	{
	  if (info->nbytes)
	    memmove (info->buf, info->buf + (info->bufsize - info->nbytes),
		     info->nbytes);
	  while ((ret == 0 && info->nbytes)
		 ||
		 ((ret =
		   mu_stream_read (istream, info->buf, info->bufsize,
				   info->ioffset, &info->nbytes)) == 0
		  && info->nbytes))
	    {
	      info->ioffset += info->nbytes;
	      while (info->nbytes)
		{
		  if ((ret =
		       mu_stream_write (ostream, info->buf, info->nbytes,
					info->ooffset, &nbytes)) != 0)
		    break;
		  info->nbytes -= nbytes;
		  info->ooffset += nbytes;
		}
	    }
	}
    }
  if (ret == 0)
    *newmsg = info->msg;
  _attachment_free (info, ret && ret != EAGAIN);
  return ret;
}

#define MESSAGE_RFC822_STR "message/rfc822"

int
mu_message_unencapsulate (mu_message_t msg, mu_message_t *newmsg,
			  mu_mime_io_buffer_t info)
{
  size_t size, nbytes;
  int ret = 0;
  mu_header_t hdr;
  mu_stream_t istream, ostream;

  if (msg == NULL)
    return EINVAL;
  if (newmsg == NULL)
    return MU_ERR_OUT_PTR_NULL;

  if (info == NULL /* FIXME: not needed? */
      && (ret = mu_message_get_header (msg, &hdr)) == 0)
    {
      mu_header_get_value (hdr, "Content-Type", NULL, 0, &size);
      if (size)
	{
	  char *content_type;
	  if ((content_type = malloc (size + 1)) == NULL)
	    return ENOMEM;
	  mu_header_get_value (hdr, "Content-Type", content_type, size + 1,
			       0);
	  ret = mu_c_strncasecmp (content_type, MESSAGE_RFC822_STR,
				  sizeof (MESSAGE_RFC822_STR) - 1);
	  free (content_type);
	  if (ret != 0)
	    return EINVAL;
	}
      else
	return EINVAL;
    }
  if ((ret = _attachment_setup (&info, msg, &istream)) != 0)
    return ret;
  if (info->msg == NULL)
    ret = mu_message_create (&info->msg, NULL);
  if (ret == 0)
    {
      mu_message_get_stream (info->msg, &ostream);
      if (info->nbytes)
	memmove (info->buf, info->buf + (info->bufsize - info->nbytes),
		 info->nbytes);
      while ((ret == 0 && info->nbytes)
	     ||
	     ((ret =
	       mu_stream_read (istream, info->buf,
			       info->bufsize, info->ioffset,
			       &info->nbytes)) == 0 && info->nbytes))
	{
	  info->ioffset += info->nbytes;
	  while (info->nbytes)
	    {
	      if ((ret =
		   mu_stream_write (ostream, info->buf, info->nbytes,
				    info->ooffset, &nbytes)) != 0)
		break;
	      info->nbytes -= nbytes;
	      info->ooffset += nbytes;
	    }
	}
    }
  if (ret == 0)
    *newmsg = info->msg;
  _attachment_free (info, ret && ret != EAGAIN);
  return ret;
}
