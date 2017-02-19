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

#ifndef _MUCPP_ATTRIBUTE_H
#define _MUCPP_ATTRIBUTE_H

#include <ostream>
#include <errno.h>
#include <mailutils/attribute.h>
#include <mailutils/cpp/error.h>

namespace mailutils
{

class Attribute
{
 protected:
  mu_attribute_t attr;

 public:
  Attribute ();
  Attribute (const mu_attribute_t);

  bool is_modified ();
  void clear_modified ();
  void set_modified ();

  bool is_userflag (int flag);
  bool is_seen ();
  bool is_answered ();
  bool is_flagged ();
  bool is_deleted ();
  bool is_draft ();
  bool is_recent ();
  bool is_read ();

  void set_userflag (int flag);
  void set_seen ();
  void set_answered ();
  void set_flagged ();
  void set_deleted ();
  void set_draft ();
  void set_recent ();
  void set_read ();

  void unset_userflag (int flag);
  void unset_seen ();
  void unset_answered ();
  void unset_flagged ();
  void unset_deleted ();
  void unset_draft ();
  void unset_recent ();
  void unset_read ();

  std::string to_string ();
  friend std::ostream& operator << (std::ostream&, Attribute&);
};

}

#endif // not _MUCPP_ATTRIBUTE_H

