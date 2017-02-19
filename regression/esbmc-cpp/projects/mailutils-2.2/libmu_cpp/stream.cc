/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2006, 2007, 2009, 2010 Free Software Foundation,
   Inc.

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
   Boston, MA 02110-1301 USA
*/

#include <mailutils/cpp/stream.h>

using namespace mailutils;

//
// Stream
//

Stream :: Stream ()
{
  this->stm = 0;
  this->opened = false;
  reference ();
}

Stream :: Stream (const mu_stream_t stm)
{
  if (stm == 0)
    throw Exception ("Stream::Stream", EINVAL);

  this->stm = stm;
  this->opened = false;
  reference ();
}

Stream :: Stream (Stream& s)
{
  s.reference ();
  this->stm = s.stm;
}

Stream :: ~Stream ()
{
  if (dereference ())
    {
      close ();
      if (this->stm)
	mu_stream_destroy (&stm, NULL);
    }
}

void
Stream :: open ()
{
  int status = mu_stream_open (stm);
  if (status == EAGAIN)
    throw Stream::EAgain ("Stream::open", status);
  else if (status)
    throw Exception ("Stream::open", status);

  this->opened = true;
}

void
Stream :: close ()
{
  if (this->opened)
    {
      int status = mu_stream_close (stm);
      if (status)
	throw Exception ("Stream::close", status);

      this->opened = false;
    }
}

void
Stream :: set_waitflags (int flags)
{
  this->wflags = flags;
}

void
Stream :: wait ()
{
  int status = mu_stream_wait (stm, &wflags, NULL);
  if (status)
    throw Exception ("Stream::wait", status);
}

void
Stream :: wait (int flags)
{
  this->wflags = flags;
  int status = mu_stream_wait (stm, &wflags, NULL);
  if (status)
    throw Exception ("Stream::wait", status);
}

int
Stream :: read (char* rbuf, size_t size, off_t offset)
{
  int status = mu_stream_read (stm, rbuf, size, offset, &read_count);
  if (status == EAGAIN)
    throw Stream::EAgain ("Stream::read", status);
  else if (status)
    throw Exception ("Stream::read", status);
  return status;
}

int
Stream :: write (const std::string& wbuf, size_t size, off_t offset)
{
  int status = mu_stream_write (stm, wbuf.c_str (), size, offset,
				&write_count);
  if (status == EAGAIN)
    throw Stream::EAgain ("Stream::write", status);
  else if (status)
    throw Exception ("Stream::write", status);
  return status;
}

int
Stream :: readline (char* rbuf, size_t size, off_t offset)
{
  int status = mu_stream_readline (stm, rbuf, size, offset, &read_count);
  if (status == EAGAIN)
    throw Stream::EAgain ("Stream::readline", status);
  else if (status)
    throw Exception ("Stream::readline", status);
  return status;
}

void
Stream :: sequential_readline (char* rbuf, size_t size)
{
  int status = mu_stream_sequential_readline (stm, rbuf, size, &read_count);
  if (status)
    throw Exception ("Stream::sequential_readline", status);
}

void
Stream :: sequential_write (const std::string& wbuf, size_t size)
{
  int status = mu_stream_sequential_write (stm, wbuf.c_str (), size);
  if (status)
    throw Exception ("Stream::sequential_write", status);
}

void
Stream :: flush ()
{
  int status = mu_stream_flush (stm);
  if (status)
    throw Exception ("Stream::flush", status);
}

namespace mailutils
{
  Stream&
  operator << (Stream& stm, const std::string& wbuf)
  {
    stm.write (wbuf, wbuf.length (), 0);
    return stm;
  }

  Stream&
  operator >> (Stream& stm, std::string& rbuf)
  {
    char tmp[1024];
    stm.read (tmp, sizeof (tmp), 0);
    rbuf = std::string (tmp);
    return stm;
  }
}

//
// TcpStream
//

TcpStream :: TcpStream (const std::string& host, int port, int flags)
{
  int status = mu_tcp_stream_create (&stm, host.c_str (), port, flags);
  if (status)
    throw Exception ("TcpStream::TcpStream", status);
}

//
// FileStream
//

FileStream :: FileStream (const std::string& filename, int flags)
{
  int status = mu_file_stream_create (&stm, filename.c_str (), flags);
  if (status)
    throw Exception ("FileStream::FileStream", status);
}

//
// StdioStream
//

StdioStream :: StdioStream (FILE* fp, int flags)
{
  int status = mu_stdio_stream_create (&stm, fp, flags);
  if (status)
    throw Exception ("StdioStream::StdioStream", status);
}

//
// ProgStream
//

ProgStream :: ProgStream (const std::string& progname, int flags)
{
  int status = mu_prog_stream_create (&stm, progname.c_str (), flags);
  if (status)
    throw Exception ("ProgStream::ProgStream", status);
}

//
// FilterProgStream
//

FilterProgStream :: FilterProgStream (const std::string& progname,
				      Stream& input)
{
  int status = mu_filter_prog_stream_create (&stm, progname.c_str (),
					     input.stm);
  this->input = new Stream (input);
  if (status)
    throw Exception ("FilterProgStream::FilterProgStream", status);
}

FilterProgStream :: FilterProgStream (const std::string& progname,
				      Stream* input)
{
  int status = mu_filter_prog_stream_create (&stm, progname.c_str (),
					     input->stm);
  this->input = new Stream (*input);
  if (status)
    throw Exception ("FilterProgStream::FilterProgStream", status);
}

