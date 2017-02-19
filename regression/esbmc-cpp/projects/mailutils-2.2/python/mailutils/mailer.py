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

from mailutils.c_api import mailer
from mailutils import address
from mailutils import debug
from mailutils.error import MailerError

class Mailer:
    def __init__ (self, url):
        self.mlr = mailer.MailerType ()
        status = mailer.create (self.mlr, url)
        if status:
            raise MailerError (status)

    def __del__ (self):
        mailer.destroy (self.mlr)
        del self.mlr

    def __getattr__ (self, name):
        if name == 'debug':
            return self.get_debug ()
        else:
            raise AttributeError, name

    def open (self, flags = 0):
        status = mailer.open (self.mlr, flags)
        if status:
            raise MailerError (status)

    def close (self):
        status = mailer.close (self.mlr)
        if status:
            raise MailerError (status)

    def send_message (self, msg, frm, to):
        if isinstance (frm, address.Address):
            frm = frm.addr
        if isinstance (to, address.Address):
            to = to.addr
        status = mailer.send_message (self.mlr, msg.msg, frm, to)
        if status:
            raise MailerError (status)

    def get_debug (self):
        status, dbg = mailer.get_debug (self.mlr)
        if status:
            raise MailerError (status)
        return debug.Debug (dbg)
