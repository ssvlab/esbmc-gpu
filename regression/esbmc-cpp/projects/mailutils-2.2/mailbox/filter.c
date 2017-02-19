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

/* Notes:
First draft: Alain Magloire.

 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <errno.h>
#include <string.h>
#include <stdlib.h>

#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <filter0.h>
#include <stream0.h>
#include <mailutils/iterator.h>
#include <mailutils/stream.h>
#include <mailutils/errno.h>
#include <mailutils/mutil.h>
#include <mailutils/cstr.h>

static void
filter_destroy (mu_stream_t stream)
{
  mu_filter_t filter = mu_stream_get_owner (stream);
  if (!(stream->flags & MU_STREAM_NO_CLOSE))
    mu_stream_destroy (&filter->stream, mu_stream_get_owner (filter->stream));
  if (filter->_destroy)
    filter->_destroy (filter);
  if (filter->property)
    mu_property_destroy (&(filter->property), filter);
  free (filter);
}

static int
filter_read (mu_stream_t stream, char *buffer, size_t buflen, mu_off_t offset,
	     size_t *nbytes)
{
  mu_filter_t filter = mu_stream_get_owner (stream);
  if (filter->_read && (filter->direction & MU_STREAM_READ
		      || filter->direction & MU_STREAM_RDWR))
    return filter->_read (filter, buffer, buflen, offset, nbytes);
  return mu_stream_read (filter->stream, buffer, buflen, offset, nbytes);
}

static int
filter_readline (mu_stream_t stream, char *buffer, size_t buflen,
		 mu_off_t offset, size_t *nbytes)
{
  mu_filter_t filter = mu_stream_get_owner (stream);
  if (filter->_readline && (filter->direction & MU_STREAM_READ
			  || filter->direction & MU_STREAM_RDWR))
    return filter->_readline (filter, buffer, buflen, offset, nbytes);
  return mu_stream_readline (filter->stream, buffer, buflen, offset, nbytes);
}

static int
filter_write (mu_stream_t stream, const char *buffer, size_t buflen,
	      mu_off_t offset, size_t *nbytes)
{
  mu_filter_t filter = mu_stream_get_owner (stream);
  if (filter->_write && (filter->direction & MU_STREAM_WRITE
		       || filter->direction & MU_STREAM_RDWR))
    return filter->_write (filter, buffer, buflen, offset, nbytes);
  return mu_stream_write (filter->stream, buffer, buflen, offset, nbytes);
}

static int
filter_open (mu_stream_t stream)
{
  mu_filter_t filter = mu_stream_get_owner (stream);

  return mu_stream_open (filter->stream);
}

static int
filter_truncate (mu_stream_t stream, mu_off_t len)
{
  mu_filter_t filter = mu_stream_get_owner (stream);
  return mu_stream_truncate (filter->stream, len);
}

static int
filter_size (mu_stream_t stream, mu_off_t *psize)
{
  mu_filter_t filter = mu_stream_get_owner (stream);
  return mu_stream_size (filter->stream, psize);
}

static int
filter_flush (mu_stream_t stream)
{
  mu_filter_t filter = mu_stream_get_owner(stream);
  return mu_stream_flush (filter->stream);
}

static int
filter_get_transport2 (mu_stream_t stream, mu_transport_t *pin, mu_transport_t *pout)
{
  mu_filter_t filter = mu_stream_get_owner (stream);
  return mu_stream_get_transport2 (filter->stream, pin, pout);
}

static int
filter_close (mu_stream_t stream)
{
  mu_filter_t filter = mu_stream_get_owner (stream);
  if (stream->flags & MU_STREAM_NO_CLOSE)
    return 0;
  return mu_stream_close (filter->stream);
}

/* NOTE: We will leak here since the monitor of the filter will never
   be release.  That's ok we can leave with this, it's only done once.  */
static mu_list_t filter_list;
struct mu_monitor filter_monitor = MU_MONITOR_INITIALIZER;

int
mu_filter_get_list (mu_list_t *plist)
{
  if (plist == NULL)
    return MU_ERR_OUT_PTR_NULL;
  mu_monitor_wrlock (&filter_monitor);
  if (filter_list == NULL)
    {
      int status = mu_list_create (&filter_list);
      if (status != 0)
	return status;
      /* Default filters.  */
      mu_list_append (filter_list, mu_base64_filter);
      mu_list_append (filter_list, mu_qp_filter);
      mu_list_append (filter_list, mu_binary_filter);
      mu_list_append (filter_list, mu_bit8_filter);
      mu_list_append (filter_list, mu_bit7_filter);
      mu_list_append (filter_list, mu_rfc822_filter);
      mu_list_append (filter_list, mu_rfc_2047_Q_filter);
      mu_list_append (filter_list, mu_rfc_2047_B_filter);
      /* FIXME: add the default encodings?  */
    }
  *plist = filter_list;
  mu_monitor_unlock (&filter_monitor);
  return 0;
}

int
mu_filter_create (mu_stream_t *pstream, mu_stream_t stream, const char *name,
		  int type, int flags)
{
  mu_iterator_t iterator = NULL;
  mu_filter_record_t filter_record = NULL;
  int  (*f_init)  (mu_filter_t)  = NULL;
  int found = 0;
  int status;
  mu_list_t list = NULL;

  if (pstream == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (stream == NULL || name == NULL)
    return EINVAL;

  mu_filter_get_list (&list);
  status = mu_list_get_iterator (list, &iterator);
  if (status != 0)
    return status;

  for (mu_iterator_first (iterator); !mu_iterator_is_done (iterator);
       mu_iterator_next (iterator))
    {
      mu_iterator_current (iterator, (void **)&filter_record);
      if ((filter_record->_is_filter
	   && filter_record->_is_filter (filter_record, name))
	  || (mu_c_strcasecmp (filter_record->name, name) == 0))
        {
	  found = 1;
	  if (filter_record->_get_filter)
	    filter_record->_get_filter (filter_record, &f_init);
	  else
	    f_init = filter_record->_mu_filter;
	  break;
        }
    }
  mu_iterator_destroy (&iterator);

  if (found)
    {
      mu_filter_t filter;

      filter = calloc (1, sizeof (*filter));
      if (filter == NULL)
	return ENOMEM;

      status = mu_stream_create (pstream, flags | MU_STREAM_NO_CHECK, filter);
      if (status != 0)
	{
	  free (filter);
	  return status;
	}

      filter->stream = stream;
      filter->filter_stream = *pstream;
      filter->direction = (flags == 0) ? MU_STREAM_READ
	                      : (flags
				 & (MU_STREAM_READ |
				    MU_STREAM_WRITE |
				    MU_STREAM_RDWR));
      filter->type = type;

      status = mu_property_create (&(filter->property), filter);
      if (status != 0)
	{
	  mu_stream_destroy (pstream, filter);
	  free (filter);
	  return status;
	}
      mu_property_set_value (filter->property, "DIRECTION",
			  ((filter->direction == MU_STREAM_WRITE) ? "WRITE":
			   (filter->direction == MU_STREAM_RDWR) ? "RDWR" :
			   "READ"), 1);
      mu_property_set_value (filter->property, "TYPE", filter_record->name, 1);
      mu_stream_set_property (*pstream, filter->property, filter);

      if (f_init != NULL)
	{
	  status = f_init (filter);
	  if (status != 0)
	    {
	      mu_stream_destroy (pstream, filter);
	      free (filter);
	      return status;
	    }
        }

      mu_stream_set_open (*pstream, filter_open, filter);
      mu_stream_set_close (*pstream, filter_close, filter);
      mu_stream_set_read (*pstream, filter_read, filter);
      mu_stream_set_readline (*pstream, filter_readline, filter);
      mu_stream_set_write (*pstream, filter_write, filter);
      mu_stream_set_get_transport2 (*pstream, filter_get_transport2, filter );
      mu_stream_set_truncate (*pstream, filter_truncate, filter );
      mu_stream_set_size (*pstream, filter_size, filter );
      mu_stream_set_flush (*pstream, filter_flush, filter );
      mu_stream_set_destroy (*pstream, filter_destroy, filter);
    }
  else
    status = MU_ERR_NOENT;
  return status;
}
