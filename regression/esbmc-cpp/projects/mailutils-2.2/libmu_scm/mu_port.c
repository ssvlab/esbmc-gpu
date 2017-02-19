/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2006, 2007, 2009, 2010 Free Software
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

#include "mu_scm.h"
#include <mailutils/io.h>

#ifndef HAVE_SCM_T_OFF
typedef off_t scm_t_off;
#endif

struct mu_port
{
  mu_stream_t stream;         /* Associated stream */
  int offset;              /* Current offset in the stream */
  SCM msg;                 /* Message the port belongs to */		
};

#define DEFAULT_BUF_SIZE 1024
#define MU_PORT(x) ((struct mu_port *) SCM_STREAM (x))

static void
mu_port_alloc_buffer (SCM port, size_t read_size, size_t write_size)
{
  scm_port *pt = SCM_PTAB_ENTRY (port);
  static char *s_mu_port_alloc_buffer = "mu_port_alloc_buffer";
  
  if (!read_size)
    read_size = DEFAULT_BUF_SIZE;
  if (!write_size)
    write_size = DEFAULT_BUF_SIZE;

  if (SCM_INPUT_PORT_P (port))
    {
      pt->read_buf = malloc (read_size);
      if (pt->read_buf == NULL)
	scm_memory_error (s_mu_port_alloc_buffer);
      pt->read_pos = pt->read_end = pt->read_buf;
      pt->read_buf_size = read_size;
    }
  else
    {
      pt->read_pos = pt->read_buf = pt->read_end = &pt->shortbuf;
      pt->read_buf_size = 1;
    }
  
  if (SCM_OUTPUT_PORT_P (port))
    {
      pt->write_buf = malloc (write_size);
      if (pt->write_buf == NULL)
	scm_memory_error (s_mu_port_alloc_buffer);
      pt->write_pos = pt->write_buf;
      pt->write_buf_size = write_size;
      pt->write_end = pt->write_buf + pt->write_buf_size;
    }
  else
    {
      pt->write_buf = pt->write_pos = &pt->shortbuf;
      pt->write_buf_size = 1;
    }
  
  SCM_SET_CELL_WORD_0 (port, SCM_CELL_WORD_0 (port) & ~SCM_BUF0);
}

static long scm_tc16_smuport;

SCM
mu_port_make_from_stream (SCM msg, mu_stream_t stream, long mode)
{
  struct mu_port *mp;
  SCM port;
  scm_port *pt;
  
  mp = scm_gc_malloc (sizeof (struct mu_port), "mu-port");
  mp->msg = msg;
  mp->stream = stream;
  mp->offset = 0;

  port = scm_new_port_table_entry (scm_tc16_smuport | mode);
  pt = SCM_PTAB_ENTRY (port);
  pt->rw_random = mu_stream_is_seekable (stream);
  SCM_SETSTREAM (port, mp);
  mu_port_alloc_buffer (port, 0, 0);
  /* FIXME:
     SCM_PTAB_ENTRY (port)->file_name = "name";*/
  return port;
}

static SCM
mu_port_mark (SCM port)
{
  if (SCM_CELL_WORD_0 (port) & SCM_OPN)
    {
      struct mu_port *mp = MU_PORT (port);
      return mp->msg;
    }
  return SCM_BOOL_F;
}

static void
mu_port_flush (SCM port)
{
  struct mu_port *mp = MU_PORT (port);
  scm_port *pt = SCM_PTAB_ENTRY (port);
  int wrsize = pt->write_pos - pt->write_buf;
  size_t n;
  
  if (wrsize)
    {
      if (mu_stream_write (mp->stream, (const char*)pt->write_buf,
			   wrsize, mp->offset, &n))
	return;
      mp->offset += n;
    }
  pt->write_pos = pt->write_buf;
  pt->rw_active = SCM_PORT_NEITHER;
}

static int
mu_port_close (SCM port)
{
  struct mu_port *mp = MU_PORT (port);
  scm_port *pt = SCM_PTAB_ENTRY (port);

  mu_port_flush (port);
  mu_stream_close (mp->stream);
  SCM_SETSTREAM (port, NULL);
		
  if (pt->read_buf != &pt->shortbuf)
    free (pt->read_buf);
  if (pt->write_buf != &pt->shortbuf)
    free (pt->write_buf);
  free (mp);
  return 0;
}

static scm_sizet
mu_port_free (SCM port)
{
  mu_port_close (port);
  return 0;
}

static int
mu_port_fill_input (SCM port)
{
  struct mu_port *mp = MU_PORT (port);
  scm_port *pt = SCM_PTAB_ENTRY (port);
  size_t nread = 0;
  int status;
  
  status = mu_stream_read (mp->stream, (char*) pt->read_buf, pt->read_buf_size,
			   mp->offset, &nread);
  if (status)
    mu_scm_error ("mu_port_fill_input", status,
		  "Error reading from stream", SCM_BOOL_F);

  if (nread == 0)
    return EOF;

  mp->offset += nread;
  pt->read_pos = pt->read_buf;
  pt->read_end = pt->read_buf + nread;
  return *pt->read_buf;
}

static void
mu_port_write (SCM port, const void *data, size_t size)
{
  scm_port *pt = SCM_PTAB_ENTRY (port);
  size_t remaining = size;
  char *input = (char*) data;
  
  while (remaining > 0)
    {
      int space = pt->write_end - pt->write_pos;
      int write_len = (remaining > space) ? space : remaining;
      
      memcpy (pt->write_pos, input, write_len);
      pt->write_pos += write_len;
      remaining -= write_len;
      input += write_len;
      if (write_len == space)
	mu_port_flush (port);
    }
}

/* Perform the synchronisation required for switching from input to
   output on the port.
   Clear the read buffer and adjust the file position for unread bytes. */
static void
mu_port_end_input (SCM port, int offset)
{
  struct mu_port *mp = MU_PORT (port);
  scm_port *pt = SCM_PTAB_ENTRY (port);
  int delta = pt->read_end - pt->read_pos;
  
  offset += delta;

  if (offset > 0)
    {
      pt->read_pos = pt->read_end;
      mp->offset -= delta;
    }
  pt->rw_active = SCM_PORT_NEITHER;
}

static mu_off_t
mu_port_seek (SCM port, mu_off_t offset, int whence)
{
  struct mu_port *mp = MU_PORT (port);
  scm_port *pt = SCM_PTAB_ENTRY (port);
  mu_off_t size = 0;
  
  if (whence == SEEK_CUR && offset == 0)
    return mp->offset;

  if (pt->rw_active == SCM_PORT_WRITE)
    {
      mu_port_flush (port);
    }
  else if (pt->rw_active == SCM_PORT_READ)
    {
      scm_end_input (port);
    }

  mu_stream_size (mp->stream, &size);
  switch (whence)
    {
    case SEEK_SET:
      break;
    case SEEK_CUR:
      offset += mp->offset;
      break;
    case SEEK_END:
      offset += size;
    }

  if (offset > size)
    return -1;
  mp->offset = offset;
  return offset;
}

static void
mu_port_truncate (SCM port, mu_off_t length)
{
  struct mu_port *mp = MU_PORT (port);
  int status;
  status = mu_stream_truncate (mp->stream, length);
  if (status)
    mu_scm_error ("mu_stream_truncate", status,
		  "Error truncating stream", SCM_BOOL_F);
}
  
static int
mu_port_print (SCM exp, SCM port, scm_print_state *pstate)
{
  struct mu_port *mp = MU_PORT (exp);
  mu_off_t size = 0;
  
  scm_puts ("#<", port);
  scm_print_port_mode (exp, port);
  scm_puts ("mu-port", port);
  if (mu_stream_size (mp->stream, &size) == 0)
    {
      char *buf;
      if (mu_asprintf (&buf, " %5lu", (unsigned long) size) == 0)
	{
	  scm_puts (buf, port);
	  scm_puts (" chars", port);
	  free (buf);
	}
    }
  scm_putc ('>', port);
  return 1;
}
     
void
mu_scm_port_init ()
{
    scm_tc16_smuport = scm_make_port_type ("mu-port",
					   mu_port_fill_input, mu_port_write);
    scm_set_port_mark (scm_tc16_smuport, mu_port_mark);
    scm_set_port_free (scm_tc16_smuport, mu_port_free);
    scm_set_port_print (scm_tc16_smuport, mu_port_print);
    scm_set_port_flush (scm_tc16_smuport, mu_port_flush);
    scm_set_port_end_input (scm_tc16_smuport, mu_port_end_input);
    scm_set_port_close (scm_tc16_smuport, mu_port_close);
    scm_set_port_seek (scm_tc16_smuport, mu_port_seek);
    scm_set_port_truncate (scm_tc16_smuport, mu_port_truncate);
    /*    scm_set_port_input_waiting (scm_tc16_smuport, mu_port_input_waiting);*/
}
