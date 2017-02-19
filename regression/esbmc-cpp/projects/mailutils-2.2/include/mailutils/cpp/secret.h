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

#ifndef _MUCPP_SECRET_H
#define _MUCPP_SECRET_H

#include <string>
#include <errno.h>
#include <mailutils/secret.h>
#include <mailutils/cpp/error.h>

namespace mailutils
{

class Secret
{
 protected:
  mu_secret_t secret;
  bool owner;

 public:
  Secret (const std::string&);
  Secret (const char* str, size_t len);
  Secret (const mu_secret_t);
  ~Secret ();

  std::string password ();
  void password_unref ();
};

}

#endif // not _MUCPP_SECRET_H

