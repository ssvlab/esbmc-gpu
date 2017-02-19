#
#  GNU Mailutils -- a suite of utilities for electronic mail
#  Copyright (C) 2009, 2010 Free Software Foundation, Inc.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 3 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General
#  Public License along with this library; if not, write to the
#  Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
#  Boston, MA 02110-1301 USA
#

from mailutils.c_api import filter
from mailutils.stream import *
from mailutils.error import StreamError

# Type
MU_FILTER_DECODE = 0
MU_FILTER_ENCODE = 1

# Direction
MU_FILTER_READ  = MU_STREAM_READ
MU_FILTER_WRITE = MU_STREAM_WRITE
MU_FILTER_RDWR  = MU_STREAM_RDWR

class FilterStream (Stream):
    def __init__ (self, transport, name, type = MU_FILTER_DECODE,
                  direction = MU_FILTER_READ):
        Stream.__init__ (self)
        status = filter.create (self.stm, transport.stm, name,
                                type, direction)
        if status:
            raise StreamError (status)

class FilterIconvStream (Stream):
    def __init__ (self, transport, fromcode, tocode, direction=MU_FILTER_READ):
        Stream.__init__ (self)
        status = filter.iconv_create (self.stm, transport.stm,
                                      fromcode, tocode, direction)
        if status:
            raise StreamError (status)
