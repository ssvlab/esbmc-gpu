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

from mailutils.c_api import folder
from mailutils import stream
from mailutils import auth
from mailutils import url
from mailutils.error import FolderError

class Folder:
    __owner = False

    def __init__ (self, f):
        if isinstance (f, folder.FolderType):
            self.folder = f
        else:
            self.folder = folder.FolderType ()
            self.__owner = True
            status = folder.create (self.folder, f)
            if status:
                raise FolderError (status)

    def __del__ (self):
        if self.__owner:
            folder.destroy (self.folder)
        del self.folder

    def open (self):
        status = folder.open (self.folder)
        if status:
            raise FolderError (status)

    def close (self):
        status = folder.close (self.folder)
        if status:
            raise FolderError (status)

    def get_stream (self):
        status, stream = folder.get_stream (self.folder)
        if status:
            raise FolderError (status)
        return stream.Stream (stm)

    def set_stream (self, stream):
        status = folder.set_stream (self.folder, stream.stm)
        if status:
            raise FolderError (status)

    def get_authority (self):
        status, authority = folder.get_authority (self.folder)
        if status:
            raise FolderError (status)
        return auth.Authority (authority)

    def set_authority (self, authority):
        status = folder.set_authority (self.folder, authority.authority)
        if status:
            raise FolderError (status)

    def get_url (self):
        status, u = folder.get_url (self.folder)
        if status:
            raise FolderError (status)
        return url.Url (u)

    def list (self, dirname, pattern, max_level = 0):
        status, lst = folder.list (self.folder, dirname, pattern, max_level)
        if status:
            raise FolderError (status)
        return lst
