/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

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

#include <mailutils/cpp/body.h>

using namespace mailutils;

//
// Body
//

Body :: Body ()
{
  int status = mu_body_create (&body, this);
  if (status)
    throw Exception ("Body::Body", status);

  this->owner = true;
}

Body :: Body (const mu_body_t body)
{
  if (body == 0)
    throw Exception ("Body::Body", EINVAL);

  this->body = body;
  this->owner = false;
}

Body :: ~Body ()
{
  if (this->owner)
    mu_body_destroy (&body, this);
}

bool
Body :: is_modified ()
{
  return (bool) mu_body_is_modified (body);
}

void
Body :: clear_modified ()
{
  int status = mu_body_clear_modified (body);
  if (status)
    throw Exception ("Body::clear_modified", status);
}

Stream&
Body :: get_stream ()
{
  mu_stream_t c_stream;

  int status = mu_body_get_stream (body, &c_stream);
  if (status)
    throw Exception ("Body::get_stream", status);

  return *new Stream (c_stream);
}

size_t
Body :: size ()
{
  size_t c_size;
  int status = mu_body_size (body, &c_size);
  if (status)
    throw Exception ("Body::size", status);
  return c_size;
}

size_t
Body :: lines ()
{
  size_t c_lines;
  int status = mu_body_lines (body, &c_lines);
  if (status)
    throw Exception ("Body::lines", status);
  return c_lines;
}

