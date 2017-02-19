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

#include <mailutils/cpp/envelope.h>

using namespace mailutils;

//
// Envelope
//

Envelope :: Envelope ()
{
  this->env = NULL;
}

Envelope :: Envelope (const mu_envelope_t env)
{
  if (env == 0)
    throw Exception ("Envelope::Envelope", EINVAL);

  this->env = env;
}

std::string
Envelope :: get_sender ()
{
  const char* buf = NULL;
  int status = mu_envelope_sget_sender (env, &buf);
  if (status)
    throw Exception ("Envelope::get_sender", status);
  return std::string (buf);
}

std::string
Envelope :: get_date ()
{
  const char* buf = NULL;
  int status = mu_envelope_sget_date (env, &buf);
  if (status)
    throw Exception ("Envelope::get_date", status);
  return std::string (buf);
}

