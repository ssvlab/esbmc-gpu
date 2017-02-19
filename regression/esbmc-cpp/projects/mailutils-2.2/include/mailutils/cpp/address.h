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

#ifndef _MUCPP_ADDRESS_H
#define _MUCPP_ADDRESS_H

#include <ostream>
#include <errno.h>
#include <mailutils/address.h>
#include <mailutils/cpp/error.h>

namespace mailutils
{

class Address
{
 protected:
  mu_address_t addr;

  friend class Mailer;

 public:
  Address ();
  Address (const std::string&);
  Address (const char *sv[], size_t len);
  Address (const mu_address_t);
  ~Address ();

  Address& operator = (const Address&);

  size_t get_count ();
  bool is_group (size_t n);

  std::string get_email (size_t n);
  std::string get_local_part (size_t n);
  std::string get_domain (size_t n);
  std::string get_personal (size_t n);
  std::string get_comments (size_t n);
  std::string get_route (size_t n);

  std::string to_string ();
  friend std::ostream& operator << (std::ostream&, Address&);

  // Address Exceptions
  class EInval : public Exception {
  public:
    EInval (const char* m, int s) : Exception (m, s) {}
  };

  class ENoent : public Exception {
  public:
    ENoent (const char* m, int s) : Exception (m, s) {}
  };
};

}

#endif // not _MUCPP_ADDRESS_H

