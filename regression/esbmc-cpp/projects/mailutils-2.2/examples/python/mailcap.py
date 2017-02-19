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

from mailutils import stream, mailcap

stm = stream.FileStream ("/etc/mailcap")
stm.open ()

mc = mailcap.Mailcap (stm)

for i, entry in enumerate (mc):
    print "entry[%d]" % (i + 1)

    # typefield
    print "\ttypefield: %s" % entry.get_typefield ()

    # view-command
    print "\tview-command: %s" % entry.get_viewcommand ()

    # fields
    for j, ent in enumerate (entry):
        print "\tfields[%d]: %s" % ((j + 1), ent)

    print
