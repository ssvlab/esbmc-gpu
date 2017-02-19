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
from mailutils.c_api import registrar

def register_format (name = None):
    """Register desired mailutils 'name' format.

    A list or tuple of strings can be given.
    Each element must be one of the following strings:

    Argument             Meaning
    -------------------------------------------------------------- 
    'mbox'               Regular UNIX mbox format
    'mh'                 MH mailbox format
    'maildir'            Maildir mailbox format
    'pop'                POP mailbox format
    'pops'               POPS mailbox format
    'imap'               IMAP mailbox format
    'imaps'              IMAPS mailbox format
    'sendmail'           sendmail mailer format
    'smtp'               SMTP mailer format

    If called without arguments, the function registers all available
    formats.

    """
    if name == None:
        registrar.register_format ()
    elif isinstance (name, types.TupleType) \
      or isinstance (name, types.ListType):
        for n in name:
            registrar.register_format (n)
    else:
        registrar.register_format (name)

def set_default_format (name):
    registrar.set_default_format (name)
