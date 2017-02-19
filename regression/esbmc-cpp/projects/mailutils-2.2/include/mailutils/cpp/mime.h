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

#ifndef _MUCPP_MIME_H
#define _MUCPP_MIME_H

#include <string>
#include <errno.h>
#include <mailutils/mime.h>
#include <mailutils/cpp/error.h>
#include <mailutils/cpp/message.h>

namespace mailutils
{

class Mime
{
 protected:
  mu_mime_t mime;

 public:
  Mime (const Message&, int);
  Mime (const mu_mime_t);
  ~Mime ();

  bool is_multipart ();
  size_t get_num_parts ();

  Message& get_part (size_t part);
  void add_part (const Message& msg);
  Message& get_message ();
};

extern int rfc2047_decode (const char* tocode, const char* fromstr,
			   char** ptostr);
extern int rfc2047_encode (const char* charset, const char* encoding,
			   const char* text, char** result);

}

#endif // not _MUCPP_MIME_H

