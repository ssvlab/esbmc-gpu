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
from mailutils import url
from mailutils.error import *

def parse (str):
    try:
	u = url.Url (str)
	u.parse ()
        print "URL: %s" % u

	print "\tscheme <%s>" % u.get_scheme ()
	print "\tuser <%s>" %   u.get_user ()

        sec = u.get_secret ()
	print "\tpasswd <%s>" % sec.password ()
	sec.password_unref ()

 	print "\tauth <%s>" %   u.get_auth ()
 	print "\thost <%s>" %   u.get_host ()
 	print "\tport %d" %     u.get_port ()
 	print "\tpath <%s>" %   u.get_path ()

        for i, param in enumerate (u.get_query ()):
            print "\tquery[%d] %s" % (i, param)

    except UrlError, e:
        print e

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
