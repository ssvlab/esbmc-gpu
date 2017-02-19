/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2004, 2007, 2009, 2010 Free
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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/cctype.h>
#include <mailutils/cstr.h>
#include <mailutils/message.h>
#include <mailutils/stream.h>
#include <mailutils/body.h>
#include <mailutils/header.h>
#include <mailutils/errno.h>
#include <mailutils/mutil.h>
#include <mime0.h>

#ifndef TRUE
#define TRUE (1)
#define FALSE (0)
#endif

/* TODO:
 *  Need to prevent re-entry into mime lib, but allow non-blocking re-entry
 *  into lib.
 */

static int
_mime_is_multipart_digest (mu_mime_t mime)
{
  if (mime->content_type)
    return (mu_c_strncasecmp
	    ("multipart/digest", mime->content_type,
	     strlen ("multipart/digest")) ? 0 : 1);
  return 0;
}

static int
_mime_append_part (mu_mime_t mime, mu_message_t msg, int offset, int len, int lines)
{
  struct _mime_part *mime_part, **part_arr;
  int             ret;
  size_t          size;
  mu_header_t        hdr;

  if ((mime_part = calloc (1, sizeof (*mime_part))) == NULL)
    return ENOMEM;

  if (mime->nmtp_parts >= mime->tparts)
    {
      if ((part_arr =
	   realloc (mime->mtp_parts,
		    (mime->tparts + 5) * sizeof (mime_part))) == NULL)
	{
	  free (mime_part);
	  return ENOMEM;
	}
      mime->mtp_parts = part_arr;
      mime->tparts += 5;
    }
  mime->mtp_parts[mime->nmtp_parts++] = mime_part;
  if (msg == NULL)
    {
      if ((ret = mu_message_create (&(mime_part->msg), mime_part)) == 0)
	{
	  if ((ret =
	       mu_header_create (&hdr, mime->header_buf,
				 mime->header_length, mime_part->msg)) != 0)
	    {
	      mu_message_destroy (&mime_part->msg, mime_part);
	      free (mime_part);
	      return ret;
	    }
	  mu_message_set_header (mime_part->msg, hdr, mime_part);
	}
      else
	{
	  free (mime_part);
	  return ret;
	}
      mime->header_length = 0;
      if ((ret =
	   mu_header_get_value (hdr, MU_HEADER_CONTENT_TYPE, NULL,
				0, &size)) != 0 || size == 0)
	{
	  if (_mime_is_multipart_digest (mime))
	    mu_header_set_value (hdr,
				 MU_HEADER_CONTENT_TYPE, "message/rfc822", 0);
	  else
	    mu_header_set_value (hdr, MU_HEADER_CONTENT_TYPE, "text/plain",
				 0);
	}
      mime_part->len = len;
      mime_part->lines = lines;
      mime_part->offset = offset;
    }
  else
    {
      mu_message_ref (msg);
      mu_message_size (msg, &mime_part->len);
      mu_message_lines (msg, &mime_part->lines);
      if (mime->nmtp_parts > 1)
	mime_part->offset = mime->mtp_parts[mime->nmtp_parts - 2]->len;
      mime_part->msg = msg;
    }
  mime_part->mime = mime;
  return 0;
}

#define _ISSPECIAL(c) (						 \
    ((c) == '(') || ((c) == ')') || ((c) == '<') || ((c) == '>') \
    || ((c) == '@') || ((c) == ',') || ((c) == ';') || ((c) == ':') \
    || ((c) == '\\') || ((c) == '.') || ((c) == '[') \
    || ((c) == ']') )

static void
_mime_munge_content_header (char *field_body)
{
  char           *p, *e, *str = field_body;
  int             quoted = 0;

  mu_str_stripws (field_body);

  if ((e = strchr (str, ';')) == NULL)
    return;
  while (*e == ';')
    {
      p = e;
      e++;
      while (*e && mu_isspace (*e)) /* remove space up to param */
	e++;
      memmove (p + 1, e, strlen (e) + 1);
      e = p + 1;

      while (*e && *e != '=')	/* find end of value */
	e++;
      e = p = e + 1;
      while (*e
	     && (quoted
		 || (!_ISSPECIAL (*e) && !mu_isspace (*e))))
	{
	  if (*e == '\\')
	    {			/* escaped */
	      memmove (e, e + 1, strlen (e));
	    }
	  else if (*e == '\"')
	    quoted = ~quoted;
	  e++;
	}
    }
}

static char    *
_mime_get_param (char *field_body, const char *param, int *len)
{
  char           *str, *p, *v, *e;
  int             quoted = 0, was_quoted;

  if (len == NULL || (str = field_body) == NULL)
    return NULL;

  p = strchr (str, ';');
  while (p)
    {
      p++;
      if ((v = strchr (p, '=')) == NULL)
	break;
      *len = 0;
      v = e = v + 1;
      was_quoted = 0;
      while (*e
	     && (quoted
		 || (!_ISSPECIAL (*e) && !mu_isspace (*e))))
	{			/* skip pass value and calc len */
	  if (*e == '\"')
	    quoted = ~quoted, was_quoted = 1;
	  else
	    (*len)++;
	  e++;
	}
      if (mu_c_strncasecmp (p, param, strlen (param)))
	{			/* no match jump to next */
	  p = strchr (e, ';');
	  continue;
	}
      else
	return was_quoted ? v + 1 : v;	/* return unquoted value */
    }
  return NULL;
}

static int
_mime_setup_buffers (mu_mime_t mime)
{
  if (mime->cur_buf == NULL
      && (mime->cur_buf = malloc (mime->buf_size)) == NULL)
    {
      return ENOMEM;
    }
  if (mime->cur_line == NULL
      && (mime->cur_line = calloc (mime->line_size, 1)) == NULL)
    {
      free (mime->cur_buf);
      return ENOMEM;
    }
  return 0;
}

static void
_mime_append_header_line (mu_mime_t mime)
{
  if (mime->header_length + mime->line_ndx > mime->header_buf_size)
    {
      char           *nhb;

      if ((nhb =
	   realloc (mime->header_buf,
		    mime->header_length + mime->line_ndx + 128)) == NULL)
	return;
      mime->header_buf = nhb;
      mime->header_buf_size = mime->header_length + mime->line_ndx + 128;
    }
  memcpy (mime->header_buf + mime->header_length, mime->cur_line,
	  mime->line_ndx);
  mime->header_length += mime->line_ndx;
}

static int
_mime_parse_mpart_message (mu_mime_t mime)
{
  char           *cp, *cp2;
  int             blength, mb_length, mb_offset, mb_lines, ret;
  size_t          nbytes;

  if (!(mime->flags & MIME_PARSER_ACTIVE))
    {
      char           *boundary;
      int             len;

      if ((ret = _mime_setup_buffers (mime)) != 0)
	return ret;
      if ((boundary =
	   _mime_get_param (mime->content_type, "boundary", &len)) == NULL)
	return EINVAL;
      if ((mime->boundary = calloc (1, len + 1)) == NULL)
	return ENOMEM;
      strncpy (mime->boundary, boundary, len);

      mime->cur_offset = 0;
      mime->line_ndx = 0;
      mime->parser_state = MIME_STATE_SCAN_BOUNDARY;
      mime->flags |= MIME_PARSER_ACTIVE;
    }
  mb_length = mime->body_length;
  mb_offset = mime->body_offset;
  mb_lines = mime->body_lines;
  blength = strlen (mime->boundary);

  while ((ret =
	  mu_stream_read (mime->stream, mime->cur_buf, mime->buf_size,
		       mime->cur_offset, &nbytes)) == 0 && nbytes)
    {
      cp = mime->cur_buf;
      while (nbytes)
	{
	  mime->cur_line[mime->line_ndx] = *cp;
	  if (*cp == '\n')
	    {
	      switch (mime->parser_state)
		{
		case MIME_STATE_BEGIN_LINE:
		  mime->cur_line[0] = *cp;
		  mime->line_ndx = 0;
		  mime->parser_state = MIME_STATE_SCAN_BOUNDARY;
		  break;

		case MIME_STATE_SCAN_BOUNDARY:
		  cp2 =
		    mime->cur_line[0] ==
		    '\n' ? mime->cur_line + 1 : mime->cur_line;
		  if (mime->line_ndx >= blength)
		    {
		      if ((!strncmp (cp2, "--", 2)
			   && !mu_c_strncasecmp (cp2 + 2, mime->boundary,
				  	         blength))
			  || !mu_c_strncasecmp (cp2, mime->boundary, blength))
			{
			  mime->parser_state = MIME_STATE_HEADERS;
			  mime->flags &= ~MIME_PARSER_HAVE_CR;
			  mb_length = mime->cur_offset 
			                   - mb_offset
			                   - mime->line_ndx;
			  if (mime->header_length)
			    /* this skips the preamble */
			    {
			      /* RFC 1521 [Page 30]:
				 NOTE: The CRLF preceding the encapsulation
				 line is conceptually attached to the boundary
				 so that it is possible to have a part that
				 does not end with a CRLF (line break). Body
				 parts that must be considered to end with line
				 breaks, therefore, must have two CRLFs
				 preceding the encapsulation line, the first
				 of which is part of the preceding body part,
				 and the second of which is part of the
				 encapsulation boundary. */
			      
			      if (mb_lines)
				/* to prevent negative values in case of a
				   malformed message */
				mb_lines--;
				  
			      _mime_append_part (mime, NULL,
						 mb_offset, mb_length,
						 mb_lines);
			    }

			  if ((&mime->cur_line[mime->line_ndx] - cp2 - 1 >
			       blength
			       && !strncmp (cp2 + blength + 2, "--", 2))
			      || (&mime->cur_line[mime->line_ndx] - cp2 - 1 ==
				  blength
				  && !strncmp (cp2 + blength, "--", 2)))
			    {	/* last boundary */
			      mime->parser_state = MIME_STATE_BEGIN_LINE;
			      mime->header_length = 0;
			    }
			  else
			    mime->line_ndx = -1; /* headers parsing requires
						    empty line */
			  break;
			}
		    }

		  if (mime->header_length)
		    mb_lines++;

		  mime->line_ndx = 0;
		  mime->cur_line[0] = *cp;	/* stay in this state but
						   leave '\n' at begining */
		  break;

		case MIME_STATE_HEADERS:
		  mime->line_ndx++;
		  _mime_append_header_line (mime);
		  if (mime->line_ndx == 1 || mime->cur_line[0] == '\r')
		    {
		      mime->parser_state = MIME_STATE_SCAN_BOUNDARY;
		      mb_offset = mime->cur_offset + 1;
		      mb_lines = 0;
		    }
		  mime->line_ndx = -1;
		  break;
		}
	    }
	  mime->line_ndx++;
	  if (mime->line_ndx >= mime->line_size)
	    {
	      size_t newsize = mime->line_size + MIME_MAX_HDR_LEN;
	      char *p = realloc (mime->cur_line, newsize);
	      if (!p)
		{
		  ret = ENOMEM;
		  break;
		}
	      mime->cur_line = p;
	      mime->line_size = newsize;
	    }
	  mime->cur_offset++;
	  nbytes--;
	  cp++;
	}
    }
  mime->body_lines = mb_lines;
  mime->body_length = mb_length;
  mime->body_offset = mb_offset;
  if (ret != EAGAIN)
    {				/* finished cleanup */
      if (mime->header_length)	/* this skips the preamble */
	_mime_append_part (mime, NULL, mb_offset, mb_length, mb_lines);
      mime->flags &= ~MIME_PARSER_ACTIVE;
      mime->body_offset = mime->body_length =
	mime->header_length = mime->body_lines = 0;
    }
  return ret;
}

/*------ Mime message functions for READING a multipart message -----*/

static int
_mimepart_body_stream_size (mu_stream_t stream, mu_off_t *psize)
{
  size_t s;
  mu_body_t body = mu_stream_get_owner (stream);
  int rc =  mu_body_size (body, &s);
  *psize = s;
  return rc;
}

static int
_mimepart_body_read (mu_stream_t stream,
		     char *buf, size_t buflen, mu_off_t off,
		     size_t *nbytes)
{
  mu_body_t          body = mu_stream_get_owner (stream);
  mu_message_t       msg = mu_body_get_owner (body);
  struct _mime_part *mime_part = mu_message_get_owner (msg);
  size_t          read_len;
  int             ret = 0;

  if (nbytes == NULL)
    return MU_ERR_OUT_NULL;

  *nbytes = 0;
  read_len = (int) mime_part->len - (int) off;
  if (read_len <= 0)
    {
      if (!mu_stream_is_seekable (mime_part->mime->stream))
	{
	  while ((ret =
		  mu_stream_read (mime_part->mime->stream,
			       buf, buflen,
			       mime_part->offset + off,
			       nbytes)) == 0 && *nbytes)
	    off += *nbytes;
	  *nbytes = 0;
	}
      return ret;
    }
  read_len = (buflen <= read_len) ? buflen : read_len;

  return mu_stream_read (mime_part->mime->stream, buf, read_len,
		      mime_part->offset + off, nbytes);
}

static int
_mimepart_body_transport (mu_stream_t stream, mu_transport_t *tr1,
			  mu_transport_t *tr2)
{
  mu_body_t          body = mu_stream_get_owner (stream);
  mu_message_t       msg = mu_body_get_owner (body);
  struct _mime_part *mime_part = mu_message_get_owner (msg);

  return mu_stream_get_transport2 (mime_part->mime->stream, tr1, tr2);
}

static int
_mimepart_body_size (mu_body_t body, size_t *psize)
{
  mu_message_t       msg = mu_body_get_owner (body);
  struct _mime_part *mime_part = mu_message_get_owner (msg);

  if (mime_part == NULL)
    return EINVAL;
  if (psize)
    *psize = mime_part->len;
  return 0;
}

static int
_mimepart_body_lines (mu_body_t body, size_t *plines)
{
  mu_message_t       msg = mu_body_get_owner (body);
  struct _mime_part *mime_part = mu_message_get_owner (msg);

  if (mime_part == NULL)
    return EINVAL;
  if (plines)
    *plines = mime_part->lines;
  return 0;
}

/*------ Mime message/header functions for CREATING multipart message -----*/
static int
_mime_set_content_type (mu_mime_t mime)
{
  const char  *content_type;
  mu_header_t     hdr = NULL;
  size_t          size;
  int             ret;

  /* Delayed the creation of the header 'til they create the final message via
     mu_mime_get_message()  */
  if (mime->hdrs == NULL)
    return 0;
  if (mime->nmtp_parts > 1)
    {
      char *cstr;
      
      if (mime->flags & MIME_ADDED_MULTIPART_CT)
	return 0;
      if (mime->flags & MU_MIME_MULTIPART_MIXED)
	content_type = "multipart/mixed; boundary=";
      else
	content_type = "multipart/alternative; boundary=";
      if (mime->boundary == NULL)
	{
	  char boundary[128];
	  
	  snprintf (boundary, sizeof boundary, "%ld-%ld=:%ld",
		    (long) random (), (long) time (0), (long) getpid ());
	  if ((mime->boundary = strdup (boundary)) == NULL)
	    return ENOMEM;
	}
      size = strlen (content_type) + 2 + strlen (mime->boundary) + 1;
      cstr = malloc (size);
      if (!cstr)
	return ENOMEM;
      strcpy (cstr, content_type);
      strcat (cstr, "\"");
      strcat (cstr, mime->boundary);
      strcat (cstr, "\"");
      mime->flags |= MIME_ADDED_MULTIPART_CT;

      ret = mu_header_set_value (mime->hdrs, MU_HEADER_CONTENT_TYPE, cstr, 1);
      free (cstr);
    }
  else
    {
      if ((mime->flags & (MIME_ADDED_CT | MIME_ADDED_MULTIPART_CT))
	  == MIME_ADDED_CT)
	return 0;
      mime->flags &= ~MIME_ADDED_MULTIPART_CT;
      if (mime->nmtp_parts)
	mu_message_get_header (mime->mtp_parts[0]->msg, &hdr);

      if (hdr == NULL
	  || mu_header_sget_value (hdr, MU_HEADER_CONTENT_TYPE,
				   &content_type))
	content_type = "text/plain; charset=us-ascii";

      ret = mu_header_set_value (mime->hdrs, MU_HEADER_CONTENT_TYPE,
				 content_type, 1);
      if (ret)
	return ret;

      if (hdr)
	{
	  const char *content_te;
	  
	  /* if the only part contains a transfer-encoding
	     field, set it on the message header too */
	  if (mu_header_sget_value (hdr,
				MU_HEADER_CONTENT_TRANSFER_ENCODING,
				&content_te) == 0)
	    ret = mu_header_set_value (mime->hdrs,
				       MU_HEADER_CONTENT_TRANSFER_ENCODING,
				       content_te, 1);

	  if (ret == 0
	      && mu_header_sget_value (hdr,
				       MU_HEADER_CONTENT_DESCRIPTION,
				       &content_te) == 0)
	    ret = mu_header_set_value (mime->hdrs,
				       MU_HEADER_CONTENT_DESCRIPTION,
				       content_te, 1);

	}
    }
  mime->flags |= MIME_ADDED_CT;
  return ret;
}

#define ADD_CHAR(buf, c, offset, buflen, nbytes) do {\
 *(buf)++ = c;\
 (offset)++;\
 (nbytes)++;\
 if (--(buflen) == 0) return 0;\
} while (0)

static int
_mime_body_read (mu_stream_t stream, char *buf, size_t buflen, mu_off_t off,
		 size_t *nbytes)
{
  mu_body_t          body = mu_stream_get_owner (stream);
  mu_message_t       msg = mu_body_get_owner (body);
  mu_mime_t          mime = mu_message_get_owner (msg);
  int                ret = 0;
  size_t             part_nbytes = 0;
  mu_stream_t        msg_stream = NULL;

  if (mime->nmtp_parts == 0)
    return EINVAL;

  if (off == 0)
    {				/* reset message */
      mime->cur_offset = 0;
      mime->cur_part = 0;
      mime->part_offset = 0;

      if (mime->nmtp_parts > 1)
	mime->flags |= MIME_INSERT_BOUNDARY;
    }

  if (off != mime->cur_offset)
    return ESPIPE;

  if (nbytes)
    *nbytes = 0;

  if ((ret = _mime_set_content_type (mime)) == 0)
    {
      do
	{
	  if (mime->nmtp_parts > 1)
	    {
	      int             len;

	      if (mime->flags & MIME_INSERT_BOUNDARY)
		{
		  if ((mime->flags & MIME_ADDING_BOUNDARY) == 0)
		    {
		      mime->boundary_len = strlen (mime->boundary);
		      mime->preamble = 2;
		      if (mime->cur_part == mime->nmtp_parts)
			mime->postamble = 2;
		      mime->flags |= MIME_ADDING_BOUNDARY;
		    }
		  while (mime->preamble)
		    {
		      mime->preamble--;
		      ADD_CHAR (buf, '-', mime->cur_offset, buflen, *nbytes);
		    }
		  len = strlen (mime->boundary) - mime->boundary_len;
		  while (mime->boundary_len)
		    {
		      mime->boundary_len--;
		      ADD_CHAR (buf,
				mime->
				boundary
				[len++], mime->cur_offset, buflen, *nbytes);
		    }
		  while (mime->postamble)
		    {
		      mime->postamble--;
		      ADD_CHAR (buf, '-', mime->cur_offset, buflen, *nbytes);
		    }
		  mime->flags &=
		    ~(MIME_INSERT_BOUNDARY | MIME_ADDING_BOUNDARY);
		  mime->part_offset = 0;
		  ADD_CHAR (buf, '\n', mime->cur_offset, buflen, *nbytes);
		}
	      if (mime->cur_part >= mime->nmtp_parts)
		return 0;
	      mu_message_get_stream (mime->mtp_parts[mime->cur_part]->msg,
				     &msg_stream);
	    }
	  else
	    {
	      mu_body_t          part_body;

	      if (mime->cur_part >= mime->nmtp_parts)
		return 0;
	      mu_message_get_body (mime->mtp_parts[mime->cur_part]->msg,
				   &part_body);
	      mu_body_get_stream (part_body, &msg_stream);
	    }
	  ret =
	    mu_stream_read (msg_stream, buf, buflen,
			 mime->part_offset, &part_nbytes);
	  if (part_nbytes)
	    {
	      mime->part_offset += part_nbytes;
	      mime->cur_offset += part_nbytes;
	      if (nbytes)
		*nbytes += part_nbytes;
	    }
	  if (ret == 0 && part_nbytes == 0)
	    {
	      mime->flags |= MIME_INSERT_BOUNDARY;
	      mime->cur_part++;
	      ADD_CHAR (buf, '\n', mime->cur_offset, buflen, *nbytes);
	    }
	}
      while (ret == 0 && part_nbytes == 0
	     && mime->cur_part <= mime->nmtp_parts);
    }
  return ret;
}

static int
_mime_body_transport (mu_stream_t stream, mu_transport_t *tr1,
		      mu_transport_t *tr2)
{
  mu_body_t          body = mu_stream_get_owner (stream);
  mu_message_t       msg = mu_body_get_owner (body);
  mu_mime_t          mime = mu_message_get_owner (msg);
  mu_stream_t        msg_stream = NULL;

  if (mime->nmtp_parts == 0 || mime->cur_offset == 0)
    return EINVAL;
  mu_message_get_stream (mime->mtp_parts[mime->cur_part]->msg, &msg_stream);
  return mu_stream_get_transport2 (msg_stream, tr1, tr2);
}

static int
_mime_body_size (mu_body_t body, size_t *psize)
{
  mu_message_t       msg = mu_body_get_owner (body);
  mu_mime_t          mime = mu_message_get_owner (msg);
  int                i, ret;
  size_t             size;

  if (mime->nmtp_parts == 0)
    return EINVAL;

  if ((ret = _mime_set_content_type (mime)) != 0)
    return ret;
  for (i = 0; i < mime->nmtp_parts; i++)
    {
      mu_message_size (mime->mtp_parts[i]->msg, &size);
      *psize += size;
      if (mime->nmtp_parts > 1)	/* boundary line */
	*psize += strlen (mime->boundary) + 3;
    }
  if (mime->nmtp_parts > 1)	/* ending boundary line */
    *psize += 2;

  return 0;
}

static int
_mime_body_lines (mu_body_t body, size_t *plines)
{
  mu_message_t       msg = mu_body_get_owner (body);
  mu_mime_t          mime = mu_message_get_owner (msg);
  int             i, ret;
  size_t          lines;

  if (mime->nmtp_parts == 0)
    return EINVAL;

  if ((ret = _mime_set_content_type (mime)) != 0)
    return ret;
  for (i = 0; i < mime->nmtp_parts; i++)
    {
      mu_message_lines (mime->mtp_parts[i]->msg, &lines);
      plines += lines;
      if (mime->nmtp_parts > 1)	/* boundary line */
	plines++;
    }
  return 0;
}

int
mu_mime_create (mu_mime_t *pmime, mu_message_t msg, int flags)
{
  mu_mime_t          mime = NULL;
  int             ret = 0;
  size_t          size;
  mu_body_t          body;

  if (pmime == NULL)
    return EINVAL;
  *pmime = NULL;
  if ((mime = calloc (1, sizeof (*mime))) == NULL)
    return ENOMEM;
  if (msg)
    {
      if ((ret = mu_message_get_header (msg, &mime->hdrs)) == 0)
	{
	  if ((ret =
	       mu_header_get_value (mime->hdrs,
				    MU_HEADER_CONTENT_TYPE,
				    NULL, 0, &size)) == 0 && size)
	    {
	      if ((mime->content_type = malloc (size + 1)) == NULL)
		ret = ENOMEM;
	      else if ((ret = mu_header_get_value (mime->hdrs,
						   MU_HEADER_CONTENT_TYPE,
						   mime->content_type,
						   size + 1, 0)) == 0)
		_mime_munge_content_header (mime->content_type);
	    }
	  else
	    {
	      if (ret == MU_ERR_NOENT)
		{
		  ret = 0;
		  if ((mime->content_type =
		       strdup ("text/plain; charset=us-ascii")) == NULL)
		    /* default as per spec. */
		    ret = ENOMEM;
		}
	    }
	  if (ret == 0)
	    {
	      mime->msg = msg;
	      mime->buf_size = MIME_DFLT_BUF_SIZE;
	      mime->line_size = MIME_MAX_HDR_LEN;
	      mu_message_get_body (msg, &body);
	      mu_body_get_stream (body, &(mime->stream));
	    }
	}
    }
  else
    {
      mime->flags |= MIME_NEW_MESSAGE | MU_MIME_MULTIPART_MIXED;
    }
  if (ret != 0)
    {
      if (mime->content_type)
	free (mime->content_type);
      free (mime);
    }
  else
    {
      mime->flags |= (flags & MIME_FLAG_MASK);
      *pmime = mime;
    }
  return ret;
}

void
mu_mime_destroy (mu_mime_t *pmime)
{
  mu_mime_t          mime;
  struct _mime_part *mime_part;
  int             i;

  if (pmime && *pmime)
    {
      mime = *pmime;
      if (mime->mtp_parts != NULL)
	{
	  for (i = 0; i < mime->nmtp_parts; i++)
	    {
	      mime_part = mime->mtp_parts[i];
	      if (mime_part->msg && mime->flags & MIME_NEW_MESSAGE)
		mu_message_unref (mime_part->msg);
	      else
		mu_message_destroy (&mime_part->msg, mime_part);
	      free (mime_part);
	    }
	  free (mime->mtp_parts);
	}
      if (mime->msg && mime->flags & MIME_NEW_MESSAGE)
	mu_message_destroy (&mime->msg, mime);
      if (mime->content_type)
	free (mime->content_type);
      if (mime->cur_buf)
	free (mime->cur_buf);
      if (mime->cur_line)
	free (mime->cur_line);
      if (mime->boundary)
	free (mime->boundary);
      if (mime->header_buf)
	free (mime->header_buf);
      free (mime);
      *pmime = NULL;
    }
}

int
mu_mime_get_part (mu_mime_t mime, size_t part, mu_message_t *msg)
{
  size_t          nmtp_parts;
  int             ret = 0, flags = 0;
  mu_stream_t        stream;
  mu_body_t          body;
  struct _mime_part *mime_part;

  if ((ret = mu_mime_get_num_parts (mime, &nmtp_parts)) == 0)
    {
      if (part < 1 || part > nmtp_parts)
	return MU_ERR_NOENT;
      if (nmtp_parts == 1 && mime->mtp_parts == NULL)
	*msg = mime->msg;
      else
	{
	  mime_part = mime->mtp_parts[part - 1];
	  if (!mime_part->body_created
	      && (ret = mu_body_create (&body, mime_part->msg)) == 0)
	    {
	      mu_body_set_size (body, _mimepart_body_size, mime_part->msg);
	      mu_body_set_lines (body, _mimepart_body_lines, mime_part->msg);
	      mu_stream_get_flags (mime->stream, &flags);
	      if ((ret =
		   mu_stream_create (&stream,
				  MU_STREAM_READ | (flags &
						    (MU_STREAM_SEEKABLE
						     | MU_STREAM_NONBLOCK)),
				  body)) == 0)
		{
		  mu_stream_set_read (stream, _mimepart_body_read, body);
		  mu_stream_set_get_transport2 (stream,
						_mimepart_body_transport,
						body);
		  mu_stream_set_size (stream, _mimepart_body_stream_size,
				      body);
		  mu_body_set_stream (body, stream, mime_part->msg);
		  mu_message_set_body (mime_part->msg, body, mime_part);
		  mime_part->body_created = 1;
		}
	    }
	  *msg = mime_part->msg;
	}
    }
  return ret;
}

int
mu_mime_get_num_parts (mu_mime_t mime, size_t *nmtp_parts)
{
  int             ret = 0;

  if ((mime->nmtp_parts == 0 && !mime->boundary)
      || mime->flags & MIME_PARSER_ACTIVE)
    {
      if (mu_mime_is_multipart (mime))
	{
	  if ((ret = _mime_parse_mpart_message (mime)) != 0)
	    return (ret);
	}
      else
	mime->nmtp_parts = 1;
    }
  *nmtp_parts = mime->nmtp_parts;
  return (ret);

}

int
mu_mime_add_part (mu_mime_t mime, mu_message_t msg)
{
  int             ret;

  if (mime == NULL || msg == NULL || (mime->flags & MIME_NEW_MESSAGE) == 0)
    return EINVAL;
  if ((ret = _mime_append_part (mime, msg, 0, 0, 0)) == 0)
    ret = _mime_set_content_type (mime);
  return ret;
}

int
mu_mime_get_message (mu_mime_t mime, mu_message_t *msg)
{
  mu_stream_t        body_stream;
  mu_body_t          body;
  int             ret = 0;

  if (mime == NULL || msg == NULL)
    return EINVAL;
  if (mime->msg == NULL)
    {
      if ((mime->flags & MIME_NEW_MESSAGE) == 0)
	return EINVAL;
      if ((ret = mu_message_create (&mime->msg, mime)) == 0)
	{
	  if ((ret = mu_header_create (&mime->hdrs, NULL, 0, mime->msg)) == 0)
	    {
	      mu_message_set_header (mime->msg, mime->hdrs, mime);
	      mu_header_set_value (mime->hdrs, MU_HEADER_MIME_VERSION, "1.0",
				   0);
	      if ((ret = _mime_set_content_type (mime)) == 0)
		{
		  if ((ret = mu_body_create (&body, mime->msg)) == 0)
		    {
		      mu_message_set_body (mime->msg, body, mime);
		      mu_body_set_size (body, _mime_body_size, mime->msg);
		      mu_body_set_lines (body, _mime_body_lines, mime->msg);
		      if ((ret =
			   mu_stream_create (&body_stream,
					     MU_STREAM_READ, body)) == 0)
			{
			  mu_stream_set_read (body_stream, _mime_body_read,
					      body);
			  mu_stream_set_get_transport2 (body_stream,
							_mime_body_transport,
							body);
			  mu_body_set_stream (body, body_stream, mime->msg);
			  *msg = mime->msg;
			  return 0;
			}
		    }
		}
	    }
	  mu_message_destroy (&mime->msg, mime);
	  mime->msg = NULL;
	}
    }
  if (ret == 0)
    *msg = mime->msg;
  return ret;
}

int
mu_mime_is_multipart (mu_mime_t mime)
{
  if (mime->content_type)
    return (mu_c_strncasecmp ("multipart", mime->content_type,
			      strlen ("multipart")) ? 0 : 1);
  return 0;
}
