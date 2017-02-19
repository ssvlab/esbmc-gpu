#  GNU Mailutils -- a suite of utilities for electronic mail
#  Copyright (C) 2009, 2010 Free Software Foundation, Inc.
#
#  GNU Mailutils is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3, or (at your option)
#  any later version.
#
#  GNU Mailutils is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with GNU Mailutils; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301 USA

import sys
from mailutils import *
from mailutils.header import *

args = ''
if len (sys.argv) > 1:
  args = sys.argv[1]

registrar.register_format ()

mbox = mailbox.MailboxDefault (args)
mbox.open ()

print "Total: %d" % len (mbox)

for msg in mbox:
  print "%s %s" % (msg.header[MU_HEADER_FROM],
                   msg.header.get_value (MU_HEADER_SUBJECT, "(NO SUBJECT)"))

mbox.close ()
