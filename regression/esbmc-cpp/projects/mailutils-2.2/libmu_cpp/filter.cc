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

#include <mailutils/cpp/filter.h>

using namespace mailutils;

//
// FilterStream
//

FilterStream :: FilterStream (Stream& transport,
			      const std::string& code,
			      int mode, int flag)
{
  int status = mu_filter_create (&this->stm,
				 transport.stm,
				 code.c_str (),
				 mode, flag);
  if (status)
    throw Exception ("FilterStream::FilterStream", status);
  this->input = new Stream (transport);
}

//
// FilterIconvStream
//

FilterIconvStream :: FilterIconvStream (Stream& transport,
					const std::string& fromcode,
					const std::string& tocode,
					int flags,
					enum mu_iconv_fallback_mode fm)
{
  int status = mu_filter_iconv_create (&this->stm, transport.stm,
				       fromcode.c_str (),
				       tocode.c_str (),
				       flags, fm);
  if (status)
    throw Exception ("FilterIconvStream::FilterIconvStream", status);
  this->input = new Stream (transport);
}

