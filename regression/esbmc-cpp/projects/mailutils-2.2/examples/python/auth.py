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
import getopt
import getpass
from mailutils import auth
from mailutils.error import AuthError

if __name__ == '__main__':
    key_type = 'name'

    try:
        opts, args = getopt.getopt (sys.argv[1:], 'p:un',
                                    ['password=', 'uid', 'name'])
        for o, arg in opts:
            if o in ('-p', '--password'):
                password = arg
            elif o in ('-u', '--uid'):
                key_type = 'uid'
            elif o in ('-n', '--name'):
                key_type = 'name'
    except getopt.GetoptError:
        print "Usage: %s [OPTION...] key" % sys.argv[0]
        print """%s -- test mailutils authentication and authorization schemes

  -n, --name                 test getpwnam functions
  -p, --password=STRING      user password
  -u, --uid                  test getpwuid functions
  """ % sys.argv[0]
        sys.exit (0)

    if not len (args):
        print "%s: not enough arguments, try --help" % sys.argv[0]
        sys.exit (0)

    if key_type == 'uid':
        key = int (args[0])
    else:
        key = args[0]

    auth.register_module (('system', 'generic'))

    if key_type == 'name':
        auth_data = auth.get_auth_by_name (key)
    elif key_type == 'uid':
        auth_data = auth.get_auth_by_uid (key)

    if not auth_data:
        print '"%s" not found' % key
        sys.exit (0)

    print "source:     %s" % auth_data.source
    print "user name:  %s" % auth_data.name
    print "password:   %s" % auth_data.passwd
    print "uid:        %d" % auth_data.uid
    print "gid:        %d" % auth_data.gid
    print "gecos:      %s" % auth_data.gecos
    print "home:       %s" % auth_data.dir
    print "shell:      %s" % auth_data.shell
    print "mailbox:    %s" % auth_data.mailbox
    print "quota:      %d" % auth_data.quota
    print "change_uid: %d" % auth_data.change_uid

    if not vars ().has_key ('password'):
        password = getpass.getpass ()

    try:
        auth.authenticate (auth_data, password)
        print 'Authenticated!'
    except AuthError, e:
        print e
