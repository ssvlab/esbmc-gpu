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
from mailutils import folder
from mailutils import registrar
from mailutils.error import *

def ls_folders (fname, ref, pattern, level):
    try:
        fld = folder.Folder (fname)
        fld.open ()

        list = fld.list (ref, pattern, level)
        for f in list:
            print f
        print "Number of folders: %d" % len (list)

        fld.close ()
    except Error, e:
        print e

if __name__ == '__main__':
    pattern = "*"
    level = 0
    argc = len (sys.argv)

    if argc == 5:
        level = int (sys.argv[4])
        pattern = sys.argv[3]
        ref = sys.argv[2]
        fname = sys.argv[1]
    elif argc == 4:
        pattern = sys.argv[3]
        ref = sys.argv[2]
        fname = sys.argv[1]
    elif argc == 3:
        ref = sys.argv[2]
        fname = sys.argv[1]
    elif argc == 2:
        ref = None
        fname = sys.argv[1]
    else:
        print "usage: lsf folder [ref] [pattern] [recursion-level]"
        sys.exit (0)

    registrar.register_format ()
    ls_folders (fname, ref, pattern, level)
