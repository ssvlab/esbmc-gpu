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
from mailutils import stream
from mailutils import filter

if len (sys.argv) != 3:
    print "usage: %s from-code to-code" % sys.argv[0]
    sys.exit (0)

sti = stream.StdioStream (sys.stdin)
sti.open ()

cvt = filter.FilterIconvStream (sti, sys.argv[1], sys.argv[2])
cvt.open ()

out = stream.StdioStream (sys.stdout, 0)
out.open ()

total = 0
while True:
    buf = cvt.read (total)
    out.sequential_write (buf)
    total += cvt.read_count
    if not cvt.read_count:
        break

out.flush ()
out.close ()
sti.close ()
