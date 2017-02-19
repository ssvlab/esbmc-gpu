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

#include <mailutils/cpp/registrar.h>

using namespace mailutils;

//
// Registrar
//

Record :: Record (const mu_record_t record)
{
  if (record == 0)
    throw Exception ("Record::Record", EINVAL);

  this->record = record;
}

Record :: ~Record ()
{
}

// Record Class Defaults
int
Record :: set_default_scheme (const std::string& scheme)
{
  return mu_registrar_set_default_scheme (scheme.c_str ());
}

std::string
Record :: get_default_scheme ()
{
  return std::string (mu_registrar_get_default_scheme ());
}

int
Record :: get_default_record (mu_record_t* prec)
{
  return mu_registrar_get_default_record (prec);
}

void
Record :: set_default_record ()
{
  mailutils::registrar_set_default_record (this->record);
}

// Record Class Registration
int
Record :: registrar ()
{
  return mailutils::registrar_record (this->record);
}

int
Record :: unregistrar ()
{
  return mailutils::unregistrar_record (this->record);
}

