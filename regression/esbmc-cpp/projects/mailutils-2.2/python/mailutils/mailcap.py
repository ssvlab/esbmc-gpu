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

from mailutils.c_api import mailcap
from mailutils.error import MailcapError

class Mailcap:
    def __init__ (self, stream):
        self.mc = mailcap.MailcapType ()
        status = mailcap.create (self.mc, stream.stm)
        if status:
            raise MailcapError (status)

    def __del__ (self):
        mailcap.destroy (self.mc)
        del self.mc

    def __len__ (self):
        return self.entries_count ()

    def __getitem__ (self, item):
        return self.get_entry (item)

    def __iter__ (self):
        self.__count = 0
        self.__len = self.entries_count ()
        return self

    def next (self):
        if self.__count >= self.__len:
            self.__count = 0
            raise StopIteration
        else:
            self.__count += 1
            return self.__getitem__ (self.__count)

    def entries_count (self):
        """Return the number of entries found in the mailcap."""
        status, count = mailcap.entries_count (self.mc)
        if status:
            raise MailcapError (status)
        return count

    def get_entry (self, item):
        """Return in MailcapEntry the mailcap entry of 'item'."""
        status, entry = mailcap.get_entry (self.mc, item)
        if status:
            raise MailcapError (status)
        return MailcapEntry (entry)


class MailcapEntry:
    def __init__ (self, entry):
        self.entry = entry

    def __len__ (self):
        return self.fields_count ()

    def __getitem__ (self, item):
        return self.get_field (item)

    def __iter__ (self):
        self.__count = 0
        self.__len = self.fields_count ()
        return self

    def next (self):
        if self.__count >= self.__len:
            self.__count = 0
            raise StopIteration
        else:
            self.__count += 1
            return self.__getitem__ (self.__count)

    def fields_count (self):
        status, count = mailcap.entry_fields_count (self.entry)
        if status:
            raise MailcapEntry (status)
        return count

    def get_field (self, i):
        status, field = mailcap.entry_get_field (self.entry, i)
        if status:
            raise MailcapEntry (status)
        return field

    def get_typefield (self):
        status, typefield = mailcap.entry_get_typefield (self.entry)
        if status:
            raise MailcapEntry (status)
        return typefield

    def get_viewcommand (self):
        status, viewcommand = mailcap.entry_get_viewcommand (self.entry)
        if status:
            raise MailcapEntry (status)
        return viewcommand
