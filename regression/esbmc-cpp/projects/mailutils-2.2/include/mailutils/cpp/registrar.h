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

#ifndef _MUCPP_REGISTRAR_H
#define _MUCPP_REGISTRAR_H

#include <string>
#include <errno.h>
#include <mailutils/registrar.h>
#include <mailutils/cpp/error.h>
#include <mailutils/cpp/url.h>

namespace mailutils
{

class Record
{
 protected:
  mu_record_t record;

 public:
  Record (const mu_record_t);
  ~Record ();

  // Defaults
  int set_default_scheme (const std::string&);
  std::string get_default_scheme ();
  int get_default_record (mu_record_t* prec);
  void set_default_record ();

  // Registration
  int registrar ();
  int unregistrar ();
};

// Defaults
inline void
registrar_set_default_record (const mu_record_t record)
{
  mu_registrar_set_default_record (record);
}

// Registration
inline int
registrar_record (const mu_record_t record)
{
  return mu_registrar_record (record);
}

inline int
unregistrar_record (const mu_record_t record)
{
  return mu_unregistrar_record (record);
}


inline void
register_all_mbox_formats ()
{
  mu_register_all_mbox_formats ();
}

inline void
register_local_mbox_formats ()
{
  mu_register_local_mbox_formats ();
}

inline void
register_remote_mbox_formats ()
{
  mu_register_remote_mbox_formats ();
}

inline void
register_all_mailer_formats ()
{
  mu_register_all_mailer_formats ();
}

inline void
register_extra_formats ()
{
  mu_register_extra_formats ();
}

inline void
register_all_formats ()
{
  mu_register_all_formats ();
}

}

#endif // not _MUCPP_REGISTRAR_H

