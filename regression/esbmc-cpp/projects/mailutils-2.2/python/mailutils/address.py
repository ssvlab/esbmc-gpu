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

import types
from mailutils.c_api import address
from mailutils.error import AddressError

class Address:
    def __init__ (self, addr):
        self.addr = address.AddressType ()
        if isinstance (addr, types.ListType):
            status = address.createv (self.addr, addr)
        else:
            status = address.create (self.addr, addr)
        if status:
            raise AddressError (status)

    def __del__ (self):
        address.destroy (self.addr)
        del self.addr

    def __len__ (self):
        return self.get_count ()

    def __str__ (self):
        status, str = address.to_string (self.addr)
        if status:
            raise AddressError (status)
        return str

    def __iter__ (self):
        self.__count = 0
        self.__len = self.get_count ()
        return self

    def next (self):
        if self.__count >= self.__len:
            self.__count = 0
            raise StopIteration
        else:
            self.__count += 1
            return self.get_email (self.__count)

    def is_group (self, n):
        """Return True if this address is just the name of a group,
        False otherwise."""
        status, isgroup = address.is_group (self.addr, n)
        if status:
            raise AddressError (status)
        return isgroup

    def get_count (self):
        """Return a count of the addresses in the address list."""
        return address.get_count (self.addr)

    def get_email (self, n):
        """Return email part of the Nth email address."""
        status, email = address.get_email (self.addr, n)
        if status:
            raise AddressError (status)
        return email

    def get_local_part (self, n):
        """Return local part of the Nth email address."""
        status, local_part = address.get_local_part (self.addr, n)
        if status:
            raise AddressError (status)
        return local_part

    def get_domain (self, n):
        """Return domain part of the Nth email address."""
        status, domain  = address.get_domain (self.addr, n)
        if status:
            raise AddressError (status)
        return domain

    def get_personal (self, n):
        """Return personal part of the Nth email address."""
        status, personal  = address.get_personal (self.addr, n)
        if status:
            raise AddressError (status)
        return personal

    def get_comments (self, n):
        """Return comment part of the Nth email address."""
        status, comments  = address.get_comments (self.addr, n)
        if status:
            raise AddressError (status)
        return comments

    def get_route (self, n):
        """Return the route part of the Nth email address."""
        status, route  = address.get_route (self.addr, n)
        if status:
            raise AddressError (status)
        return route
