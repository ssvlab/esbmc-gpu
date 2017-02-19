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

from mailutils.c_api import body
from mailutils import stream
from mailutils.error import BodyError

class Body:
    def __init__ (self, bd):
        self.bd = bd

    def __del__ (self):
        del self.bd

    def __getattr__ (self, name):
        if name == 'size':
            return self.get_size ()
        elif name == 'lines':
            return self.get_lines ()
        else:
            raise AttributeError, name

    def __len__ (self):
        return self.get_size ()

    def get_size (self):
        status, size = body.size (self.bd)
        if status:
            raise BodyError (status)
        return size

    def get_lines (self):
        status, lines = body.lines (self.bd)
        if status:
            raise BodyError (status)
        return lines

    def get_stream (self):
        status, stm = body.get_stream (self.bd)
        if status:
            raise BodyError (status)
        return stream.Stream (stm)
