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
from mailutils import *

USAGE = "usage: mailer [-hd] [-m mailer] [-f from] [to]..."
HELP  = """
  -h    print this helpful message
  -m    a mailer URL (default is \"sendmail:\")
  -f    the envelope from address (default is from user environment)
  to    a list of envelope to addresses (default is from message)

An RFC2822 formatted message is read from stdin and delivered using
the mailer."""

optmailer = "sendmail:"
optfrom = False
optdebug = False

try:
    opts, args = getopt.getopt (sys.argv[1:], 'hdm:f:')
    for o, a in opts:
        if o == '-h':
            print USAGE
            print HELP
            sys.exit (0)
        elif o == '-d':
            optdebug = True
        elif o == '-m':
            optmailer = a
        elif o == '-f':
            optfrom = a
except getopt.GetoptError:
    print USAGE
    sys.exit (0)

registrar.register_format ('sendmail')

frm = None
to = None

if optfrom:
    frm = address.Address (optfrom)
if args:
    to = address.Address (args)

sti = stream.StdioStream (sys.stdin, stream.MU_STREAM_SEEKABLE)
sti.open ()

msg = message.Message ()
msg.set_stream (sti)

mlr = mailer.Mailer (optmailer)
if optdebug:
    mlr.debug.set_level (debug.MU_DEBUG_PROT)

mlr.open ()
mlr.send_message (msg, frm, to)
mlr.close ()
