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

#ifndef _MUCPP_FILTER_H
#define _MUCPP_FILTER_H

#include <mailutils/filter.h>
#include <mailutils/cpp/stream.h>

namespace mailutils
{

class FilterStream : public Stream
{
 private:
  Stream *input;

 public:
  FilterStream (Stream& transport, const std::string& code,
		int mode, int flag);
};

class FilterIconvStream : public Stream
{
 private:
  Stream *input;

 public:
  FilterIconvStream (Stream& transport,
		     const std::string& fromcode,
		     const std::string& tocode,
		     int flags,
		     enum mu_iconv_fallback_mode fallback_mode);
};

}

#endif // not _MUCPP_FILTER_H

