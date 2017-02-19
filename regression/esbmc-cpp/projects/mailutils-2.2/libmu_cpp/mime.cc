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

#include <mailutils/cpp/mime.h>

using namespace mailutils;

//
// Mime
//

Mime :: Mime (const Message& msg, int flags)
{
  int status = mu_mime_create (&mime, msg.msg, flags);
  if (status)
    throw Exception ("Mime::Mime", status);
}

Mime :: Mime (const mu_mime_t mime)
{
  if (mime == 0)
    throw Exception ("Mime::Mime", EINVAL);

  this->mime = mime;
}

Mime :: ~Mime ()
{
  mu_mime_destroy (&mime);
}

bool
Mime :: is_multipart ()
{
  return (bool) mu_mime_is_multipart (mime);
}

size_t
Mime :: get_num_parts ()
{
  size_t nparts;

  int status = mu_mime_get_num_parts (mime, &nparts);
  if (status)
    throw Exception ("Mime::get_num_parts", status);
  return nparts;
}

Message&
Mime :: get_part (size_t part)
{
  mu_message_t c_msg;

  int status = mu_mime_get_part (mime, part, &c_msg);
  if (status)
    throw Exception ("Mime::get_part", status);

  return *new Message (c_msg);
}

void
Mime :: add_part (const Message& msg)
{
  int status = mu_mime_add_part (mime, msg.msg);
  if (status)
    throw Exception ("Mime::add_part", status);
}

Message&
Mime :: get_message ()
{
  mu_message_t c_msg;

  int status = mu_mime_get_message (mime, &c_msg);
  if (status)
    throw Exception ("Mime::get_message", status);

  return *new Message (c_msg);
}

inline int
rfc2047_decode (const char *tocode, const char *fromstr, char **ptostr)
{
  return mu_rfc2047_decode (tocode, fromstr, ptostr);
}

inline int
rfc2047_encode (const char *charset, const char *encoding, 
		const char *text, char **result)
{
  return mu_rfc2047_encode (charset, encoding, text, result);
}

