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

from mailutils.c_api import secret
from mailutils.error import SecretError

def clear_passwd (passwd):
    secret.clear_passwd (passwd)

class Secret:
    __owner = False

    def __init__ (self, sec):
        if isinstance (sec, secret.SecretType):
            self.secret = sec
        else:
            self.secret = secret.SecretType ()
            self.__owner = True
            status = secret.create (self.secret, sec, len (sec))
            if status:
                raise SecretError (status)

    def __del__ (self):
        if self.__owner:
            secret.destroy (self.secret)
        del self.secret

    def password (self):
        return secret.password (self.secret)

    def password_unref (self):
        secret.password_unref (self.secret)
