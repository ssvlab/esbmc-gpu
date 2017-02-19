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

#include <mailutils/cpp/debug.h>

using namespace mailutils;

//
// Debug
//

Debug :: Debug ()
{
  this->debug = NULL;
}

Debug :: Debug (const mu_debug_t debug)
{
  if (debug == 0)
    throw Exception ("Debug::Debug", EINVAL);

  this->debug = debug;
}

void
Debug :: set_level (const mu_log_level_t level)
{
  int status = mu_debug_set_level (debug, level);
  if (status)
    throw Exception ("Debug::set_level", status);
}

