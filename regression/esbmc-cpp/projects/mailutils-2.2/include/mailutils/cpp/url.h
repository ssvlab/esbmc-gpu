/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2006, 2007, 2008, 2009, 2010 Free Software
   Foundation, Inc.

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

#ifndef _MUCPP_URL_H
#define _MUCPP_URL_H

#include <string>
#include <vector>
#include <ostream>
#include <errno.h>
#include <mailutils/url.h>
#include <mailutils/cpp/error.h>
#include <mailutils/cpp/secret.h>

namespace mailutils
{

class Url
{
 protected:
  mu_url_t url;

 public:
  Url (const std::string&);
  Url (const char*);
  Url (const mu_url_t);
  ~Url ();

  void parse ();
  long get_port ();
  std::string get_scheme ();
  std::string get_user ();
  std::string get_auth ();
  std::string get_host ();
  std::string get_path ();
  std::vector<std::string> get_query ();
  Secret& get_secret ();

  std::string to_string ();
  friend std::ostream& operator << (std::ostream&, Url&);
};

}

#endif // not _MUCPP_URL_H

