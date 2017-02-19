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

#include <mailutils/cpp/secret.h>

using namespace mailutils;

//
// Secret
//

Secret :: Secret (const std::string& str)
{
  int status = mu_secret_create (&secret, str.c_str (), str.size ());
  if (status)
    throw Exception ("Secret::Secret", status);

  this->owner = true;
}

Secret :: Secret (const char* str, size_t len)
{
  int status = mu_secret_create (&secret, str, len);
  if (status)
    throw Exception ("Secret::Secret", status);

  this->owner = true;
}

Secret :: Secret (const mu_secret_t secret)
{
  if (secret == 0)
    throw Exception ("Secret::Secret", EINVAL);

  this->secret = secret;
  this->owner = false;
}

Secret :: ~Secret ()
{
  if (this->owner)
    mu_secret_destroy (&secret);
}

std::string
Secret :: password ()
{
  return std::string (mu_secret_password (secret));
}

void
Secret :: password_unref ()
{
  mu_secret_password_unref (secret);
}

