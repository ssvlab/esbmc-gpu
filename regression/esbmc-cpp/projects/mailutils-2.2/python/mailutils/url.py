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

from mailutils.c_api import url
from mailutils import secret
from mailutils.error import *

class Url:
    __owner = False

    def __init__ (self, u):
        if isinstance (u, url.UrlType):
            self.url = u
        else:
            self.url = url.UrlType ()
            self.__owner = True
            status = url.create (self.url, u)
            if status:
                raise UrlError (status)

    def __del__ (self):
        if self.__owner:
            url.destroy (self.url)
        del self.url

    def __str__ (self):
        return url.to_string (self.url)

    def parse (self):
        """Parses the url, after calling this the get functions
        can be called."""
        if self.__owner:
            status = url.parse (self.url)
            if status:
                raise UrlError (status)

    def get_port (self):
        status, port = url.get_port (self.url)
        if status:
            raise UrlError (status)
        return port

    def get_scheme (self):
        status, scheme = url.get_scheme (self.url)
        if status == MU_ERR_NOENT:
           return ''
        elif status:
            raise UrlError (status)
        return scheme

    def get_user (self):
        status, user = url.get_user (self.url)
        if status == MU_ERR_NOENT:
           return ''
        elif status:
            raise UrlError (status)
        return user

    def get_secret (self):
        status, sec = url.get_secret (self.url)
        if status == MU_ERR_NOENT:
            return secret.Secret ('')
        elif status:
            raise UrlError (status)
        return secret.Secret (sec)

    def get_auth (self):
        status, auth = url.get_auth (self.url)
        if status == MU_ERR_NOENT:
           return ''
        elif status:
            raise UrlError (status)
        return auth

    def get_host (self):
        status, host = url.get_host (self.url)
        if status == MU_ERR_NOENT:
           return ''
        elif status:
            raise UrlError (status)
        return host

    def get_path (self):
        status, path = url.get_path (self.url)
        if status == MU_ERR_NOENT:
           return ''
        elif status:
            raise UrlError (status)
        return path

    def get_query (self):
        status, query = url.get_query (self.url)
        if status == MU_ERR_NOENT:
           return ''
        elif status:
            raise UrlError (status)
        return query
