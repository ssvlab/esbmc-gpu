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

#ifndef _MUCPP_ERROR_H
#define _MUCPP_ERROR_H

#include <string>
#include <mailutils/errno.h>
#include <mailutils/error.h>

namespace mailutils
{

class Exception
{
 protected:
  int pstatus;
  const char* pmethod;
  const char* pmsgerr;

 public:
  Exception (const char* method_name, int status) {
    pstatus = status;
    pmethod = method_name;
    pmsgerr = mu_strerror (status);
  }

  Exception (const std::string method_name, int status) {
    pstatus = status;
    pmethod = method_name.c_str ();
    pmsgerr = mu_strerror (status);
  }

  int status () const {
    return pstatus;
  }

  const char* method () const {
    return pmethod;
  }

  const char* msg_error () const {
    return pmsgerr;
  }

  const char* what () const {
    return pmsgerr;
  }
};

inline int
verror (const char* fmt, va_list ap)
{
  return mu_verror (fmt, ap);
}

inline int
error (const char* fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  mu_verror (fmt, ap);
  va_end (ap);
  return 0;
}

}

#endif /* not _MUCPP_ERROR_H */

