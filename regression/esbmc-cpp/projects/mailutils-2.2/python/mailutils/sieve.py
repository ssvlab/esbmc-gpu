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

from mailutils.c_api import sieve
from mailutils.error import SieveMachineError

class Machine:
    def __init__ (self):
        self.mach = sieve.SieveMachineType ()
        status = sieve.machine_init (self.mach)
        if status:
            raise SieveMachineError (status)

    def __del__ (self):
        sieve.machine_destroy (self.mach)
        del self.mach

    def set_debug (self, fnc):
        status = sieve.set_debug (self.mach, fnc)
        if status:
            raise SieveMachineError (status)

    def set_error (self, fnc):
        status = sieve.set_error (self.mach, fnc)
        if status:
            raise SieveMachineError (status)

    def set_parse_error (self, fnc):
        status = sieve.set_parse_error (self.mach, fnc)
        if status:
            raise SieveMachineError (status)

    def set_logger (self, fnc):
        status = sieve.set_logger (self.mach, fnc)
        if status:
            raise SieveMachineError (status)

    def compile (self, name):
        """Compile the sieve script from the file 'name'."""
        status = sieve.compile (self.mach, name)
        if status:
            raise SieveMachineError (status)

    def disass (self):
        """Dump the disassembled code of the sieve machine."""
        status = sieve.disass (self.mach)
        if status:
            raise SieveMachineError (status)

    def mailbox (self, mbox):
        """Execute the code from the given instance of sieve machine
        over each message in the mailbox."""
        status = sieve.mailbox (self.mach, mbox.mbox)
        if status:
            raise SieveMachineError (status)

    def message (self, msg):
        """Execute the code from the given instance of sieve machine
        over the 'msg'."""
        status = sieve.message (self.mach, msg.msg)
        if status:
            raise SieveMachineError (status)
