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
from mailutils import address
from mailutils import util
from mailutils.error import *

def parse (str):
    util.set_user_email_domain ("localhost")

    try:
        addr = address.Address (str)

        count = addr.get_count ()
        print "%s => count %d" % (addr, len (addr))

        for no in range (1, count + 1):
            isgroup = addr.is_group (no)
            print "%d " % no,

            if isgroup:
                print "group <%s>" % addr.get_personal (no)
            else:
                print "email <%s>" % addr.get_email (no)

            if not isgroup:
                print "   personal <%s>" % addr.get_personal (no)

            print "   comments <%s>" % addr.get_comments (no)
            print "   local-part <%s> domain <%s>" % (addr.get_local_part (no),
                                                      addr.get_domain (no))
        print "   route <%s>" % addr.get_route (no)
    except AddressError, e:
        print e
    print

def parseinput ():
    try:
        while True:
            line = sys.stdin.readline ().strip ()
            if line == '':
                break
            parse (line)
    except KeyboardInterrupt:
        sys.exit ()

if __name__ == '__main__':
    if len (sys.argv) == 1:
        parseinput ()
        sys.exit ()

    for arg in sys.argv[1:]:
        if arg == '-':
            parseinput ()
        else:
            parse (arg)
