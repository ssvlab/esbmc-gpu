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

#include <mailutils/cpp/header.h>

using namespace mailutils;

//
// Header
//

Header :: Header ()
{
  this->hdr = NULL;
}

Header :: Header (const mu_header_t hdr)
{
  if (hdr == 0)
    throw Exception ("Header::Header", EINVAL);

  this->hdr = hdr;
}

bool
Header :: has_key (const std::string& name)
{
  const char* buf = NULL;

  int status = mu_header_sget_value (hdr, name.c_str (), &buf);
  if (status == MU_ERR_NOENT)
    return false;
  else if (status)
    throw Exception ("Header::has_key", status);

  return true;
}

std::string
Header :: get_value (const std::string& name)
{
  const char* buf = NULL;

  int status = mu_header_sget_value (hdr, name.c_str (), &buf);
  if (status)
    throw Exception ("Header::get_value", status);

  return std::string (buf);
}

std::string
Header :: get_value (const std::string& name, const std::string& def)
{
  const char* buf = NULL;

  int status = mu_header_sget_value (hdr, name.c_str (), &buf);
  if (status == MU_ERR_NOENT)
    return std::string (def);
  else if (status)
    throw Exception ("Header::get_value", status);

  return std::string (buf);
}

size_t
Header :: size ()
{
  size_t c_size;
  int status = mu_header_size (hdr, &c_size);
  if (status)
    throw Exception ("Header::size", status);
  return c_size;
}

size_t
Header :: lines ()
{
  size_t c_lines;
  int status = mu_header_lines (hdr, &c_lines);
  if (status)
    throw Exception ("Header::lines", status);
  return c_lines;
}

