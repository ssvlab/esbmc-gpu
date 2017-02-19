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

#include <cstdlib>
#include <mailutils/cpp/mutil.h>

using namespace mailutils;

//
// MUtil
//

int
mailutils :: set_user_email (const std::string& str)
{
  return mu_set_user_email (str.c_str ());
}

int
mailutils :: set_user_email_domain (const std::string& str)
{
   return mu_set_user_email_domain (str.c_str ());
}

std::string
mailutils :: tempname ()
{
  std::string name;
  char *c_str = mu_tempname (NULL);
  if (c_str) {
    name = c_str;
    free (c_str);
  }
  return name;
}

std::string
mailutils :: tempname (const std::string& tmpdir)
{
  std::string name;
  char *c_str = mu_tempname (tmpdir.c_str ());
  if (c_str) {
    name = c_str;
    free (c_str);
  }
  return name;
}

