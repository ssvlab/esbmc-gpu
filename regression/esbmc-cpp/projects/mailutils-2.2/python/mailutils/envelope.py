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

from mailutils.c_api import envelope
from mailutils.error import EnvelopeError

MU_ENVELOPE_DATE_FORMAT = "%a %b %d %H:%M:%S %Y"
MU_ENVELOPE_DATE_LENGTH = 24

class Envelope:
    def __init__ (self, env):
        self.env = env

    def __del__ (self):
        del self.env

    def get_sender (self):
        """Get the address that this message was reportedly received from."""
        status, sender = envelope.get_sender (self.env)
        if status:
            raise EnvelopeError (status)
        return sender

    def get_date (self):
        """Get the date that the message was delivered to the mailbox, in
        something close to ANSI ctime() format: Mon Jul 05 13:08:27 1999."""
        status, date = envelope.get_date (self.env)
        if status:
            raise EnvelopeError (status)
        return date
