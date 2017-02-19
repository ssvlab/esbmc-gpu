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

#include <mailutils/cpp/attribute.h>

using namespace mailutils;

//
// Attribute
//

Attribute :: Attribute ()
{
}

Attribute :: Attribute (const mu_attribute_t attr)
{
  if (attr == 0)
    throw Exception ("Attribute::Attribute", EINVAL);

  this->attr = attr;
}

inline bool
Attribute :: is_modified ()
{
  return (bool) mu_attribute_is_modified (attr);
}

inline void
Attribute :: clear_modified ()
{
  mu_attribute_clear_modified (attr);
}

inline void
Attribute :: set_modified ()
{
  mu_attribute_set_modified (attr);
}

//
// is_*
//

inline bool
Attribute :: is_userflag (int flag)
{
  return (bool) mu_attribute_is_userflag (attr, flag);
}

inline bool
Attribute :: is_seen ()
{
  return (bool) mu_attribute_is_seen (attr);
}

inline bool
Attribute :: is_answered () {
  return (bool) mu_attribute_is_answered (attr);
}

inline bool
Attribute :: is_flagged ()
{
  return (bool) mu_attribute_is_flagged (attr);
}

inline bool
Attribute :: is_deleted ()
{
  return (bool) mu_attribute_is_deleted (attr);
}

inline bool
Attribute :: is_draft ()
{
  return (bool) mu_attribute_is_draft (attr);
}

inline bool
Attribute :: is_recent ()
{
  return (bool) mu_attribute_is_recent (attr);
}

inline bool
Attribute :: is_read ()
{
  return (bool) mu_attribute_is_read (attr);
}

//
// set_*
//

inline void
Attribute :: set_userflag (int flag)
{
  mu_attribute_set_userflag (attr, flag);
}

inline void
Attribute :: set_seen ()
{
  mu_attribute_set_seen (attr);
}

inline void
Attribute :: set_answered ()
{
  mu_attribute_set_answered (attr);
}

inline void
Attribute :: set_flagged ()
{
  mu_attribute_set_flagged (attr);
}

inline void
Attribute :: set_deleted ()
{
  mu_attribute_set_deleted (attr);
}

inline void
Attribute :: set_draft ()
{
  mu_attribute_set_draft (attr);
}

inline void
Attribute :: set_recent ()
{
  mu_attribute_set_recent (attr);
}

inline void
Attribute :: set_read ()
{
  mu_attribute_set_read (attr);
}

//
// unset_*
//

inline void
Attribute :: unset_userflag (int flag)
{
  mu_attribute_unset_userflag (attr, flag);
}

inline void
Attribute :: unset_seen ()
{
  mu_attribute_unset_seen (attr);
}

inline void
Attribute :: unset_answered ()
{
  mu_attribute_unset_answered (attr);
}

inline void
Attribute :: unset_flagged ()
{
  mu_attribute_unset_flagged (attr);
}

inline void
Attribute :: unset_deleted ()
{
  mu_attribute_unset_deleted (attr);
}

inline void
Attribute :: unset_draft ()
{
  mu_attribute_unset_draft (attr);
}

inline void
Attribute :: unset_recent ()
{
  mu_attribute_unset_recent (attr);
}

inline void
Attribute :: unset_read ()
{
  mu_attribute_unset_read (attr);
}

std::string
Attribute :: to_string ()
{
  char buf[MU_STATUS_BUF_SIZE];
  size_t na = 0;
  mu_attribute_to_string (attr, buf, sizeof (buf), &na);
  return std::string (buf);
}

namespace mailutils
{
  std::ostream& operator << (std::ostream& os, Attribute& attr) {
    return os << attr.to_string ();
  };
}

