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

from mailutils.c_api import debug
from mailutils.error import DebugError

MU_DEBUG_ERROR  = 0
MU_DEBUG_TRACE0 = 1
MU_DEBUG_TRACE  = MU_DEBUG_TRACE0
MU_DEBUG_TRACE1 = 2
MU_DEBUG_TRACE2 = 3
MU_DEBUG_TRACE3 = 4
MU_DEBUG_TRACE4 = 5
MU_DEBUG_TRACE5 = 6
MU_DEBUG_TRACE6 = 7
MU_DEBUG_TRACE7 = 8
MU_DEBUG_PROT   = 9

class Debug:
    def __init__ (self, dbg):
        self.dbg = dbg

    def __del__ (self):
        del self.dbg

    def set_level (self, level = MU_DEBUG_PROT):
        status = debug.set_level (self.dbg, level)
        if status:
            raise DebugError (status)
