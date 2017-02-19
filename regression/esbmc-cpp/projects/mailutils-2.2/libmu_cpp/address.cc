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

#include <mailutils/cpp/address.h>

using namespace mailutils;

//
// Address
//

Address :: Address ()
{
  addr = NULL;
}

Address :: Address (const std::string& str)
{
  int status = mu_address_create (&addr, str.c_str ());
  if (status)
    throw Exception ("Address::Address", status);
}

Address :: Address (const char *sv[], size_t len)
{
  int status = mu_address_createv (&addr, sv, len);
  if (status)
    throw Exception ("Address::Address", status);
}

Address :: Address (const mu_address_t addr)
{
  if (addr == 0)
    throw Exception ("Address::Address", EINVAL);

  this->addr = addr;
}

Address :: ~Address ()
{
  if (addr)
    mu_address_destroy (&addr);
}

Address&
Address :: operator = (const Address& a)
{
  if (this != &a)
    {
      if (this->addr)
	mu_address_destroy (&this->addr);
      this->addr = mu_address_dup (a.addr);
    }
  return *this;
}

bool
Address :: is_group (size_t n)
{
  int isgroup;
  int status = mu_address_is_group (addr, n, &isgroup);
  if (status == EINVAL)
    throw Address::EInval ("Address::is_group", status);
  else if (status == ENOENT)
    throw Address::ENoent ("Address::is_group", status);

  return (bool) isgroup;
}

size_t
Address :: get_count ()
{
  size_t count;
  mu_address_get_count (addr, &count);
  return count;
}

std::string
Address :: get_email (size_t n)
{
  const char* buf = NULL;
  int status = mu_address_sget_email (addr, n, &buf);
  if (status == EINVAL)
    throw Address::EInval ("Address::get_email", status);
  else if (status == ENOENT)
    throw Address::ENoent ("Address::get_email", status);

  return std::string (buf ? buf : "");
}

std::string
Address :: get_local_part (size_t n)
{
  const char* buf = NULL;
  int status = mu_address_sget_local_part (addr, n, &buf);
  if (status == EINVAL)
    throw Address::EInval ("Address::get_local_part", status);
  else if (status == ENOENT)
    throw Address::ENoent ("Address::get_local_part", status);

  return std::string (buf ? buf : "");
}

std::string
Address :: get_domain (size_t n)
{
  const char* buf = NULL;
  int status = mu_address_sget_domain (addr, n, &buf);
  if (status == EINVAL)
    throw Address::EInval ("Address::get_domain", status);
  else if (status == ENOENT)
    throw Address::ENoent ("Address::get_domain", status);

  return std::string (buf ? buf : "");
}

std::string
Address :: get_personal (size_t n)
{
  const char* buf = NULL;
  int status = mu_address_sget_personal (addr, n, &buf);
  if (status == EINVAL)
    throw Address::EInval ("Address::get_personal", status);
  else if (status == ENOENT)
    throw Address::ENoent ("Address::get_personal", status);

  return std::string (buf ? buf : "");
}

std::string
Address :: get_comments (size_t n)
{
  const char* buf = NULL;
  int status = mu_address_sget_comments (addr, n, &buf);
  if (status == EINVAL)
    throw Address::EInval ("Address::get_comments", status);
  else if (status == ENOENT)
    throw Address::ENoent ("Address::get_comments", status);

  return std::string (buf ? buf : "");
}

std::string
Address :: get_route (size_t n)
{
  const char* buf = NULL;
  int status = mu_address_sget_route (addr, n, &buf);
  if (status == EINVAL)
    throw Address::EInval ("Address::get_route", status);
  else if (status == ENOENT)
    throw Address::ENoent ("Address::get_route", status);

  return std::string (buf ? buf : "");
}

std::string
Address :: to_string ()
{
  size_t n;
  char buf[1024];
  int status = mu_address_to_string (addr, buf, sizeof (buf), &n);
  if (status)
    throw Exception ("Address::to_string", status);

  return std::string (buf);
}

namespace mailutils
{
  std::ostream& operator << (std::ostream& os, Address& addr) {
    return os << addr.to_string ();
  };
}

