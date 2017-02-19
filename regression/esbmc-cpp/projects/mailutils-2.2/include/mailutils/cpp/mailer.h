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

#ifndef _MUCPP_MAILER_H
#define _MUCPP_MAILER_H

#include <string>
#include <errno.h>
#include <mailutils/mailer.h>
#include <mailutils/cpp/debug.h>
#include <mailutils/cpp/message.h>
#include <mailutils/cpp/address.h>

namespace mailutils
{

class Mailer
{
 protected:
  mu_mailer_t mailer;

 public:
  Mailer (const std::string&);
  Mailer (const mu_mailer_t);
  ~Mailer ();

  void open ();
  void open (int flags);
  void close ();
  void send_message (const Message& msg, const Address& from,
		     const Address& to);

  Debug& get_debug ();
};

}

#endif // not _MUCPP_MAILER_H

