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
from mailutils.header import *

print_attachments = False
indent_level = 4

def print_file (fname, indent):
    try:
        fp = open (fname, 'r')
        for line in fp:
            print "%*.*s%s" % (indent, indent, '', line)
        fp.close ()
        remove (fname)
    except OSError, e:
        print e

def print_message_part_sizes (part, indent):
    print "%*.*sMessage part size - %d/%d: %d/%d, %d/%d" % \
          (indent, indent, '',
           part.size, part.lines,
           part.header.size, part.header.lines,
           part.body.size, part.body.lines)

def message_display_parts (msg, indent):
  # How many parts does the message has?
  nparts = msg.get_num_parts ()

  # Iterate through all the parts. Treat type "message/rfc822"
  # differently, since it is a message of its own that can have other
  # subparts (recursive).
  for j in range (1, nparts + 1):
      part = msg.get_part (j)
      hdr = part.get_header ()

      type = hdr.get_value (MU_HEADER_CONTENT_TYPE, 'text/plain')
      encoding = hdr.get_value (MU_HEADER_CONTENT_TRANSFER_ENCODING, '7bit')

      print "%*.*sType of part %d = %s" % (indent, indent, '', j, type)
      print_message_part_sizes (part, indent)

      ismulti = part.is_multipart ()
      if type == "message/rfc822" or ismulti:
          if not ismulti:
              part = part.unencapsulate ()

          hdr = part.get_header ()
	  frm = hdr.get_value (MU_HEADER_FROM, "[none]")
	  subject = hdr.get_value (MU_HEADER_SUBJECT, "[none]")

          print "%*.*sEncapsulated message : %s\t%s" % \
                (indent, indent, '', frm, subject)
          print "%*.*sBegin" % (indent, indent, '')

	  nsubparts = part.get_num_parts ()
          message_display_parts (part, indent + indent_level)
      elif (type.startswith ("text/plain") or
            type.startswith ("text/html") or type == ''):
          print "%*.*sText Message" % (indent, indent, '')
          print "%*.*sBegin" % (indent, indent, '')

	  flt = filter.FilterStream (part.body.get_stream (), encoding)
          offset = 0

          while True:
              buf = flt.readline (offset)
              offset += flt.read_count
              if not flt.read_count:
                  break
              print "%*.*s%s" % (indent, indent, '', buf),
      else:
          # Save the attachements.
          try:
              fname, lang = part.get_attachment_name ()
	  except:
              fname = util.tempname ()

          print "%*.*sAttachment - saving [%s]" % \
                (indent, indent, '', fname)
          print "%*.*sBegin" % (indent, indent, '')

          part.save_attachment ()
          if print_attachments:
              print_file (fname, indent)

      print "%*.*sEnd" % (indent, indent, '')


if __name__ == '__main__':
  optdebug = False

  try:
      opts, args = getopt.getopt (sys.argv[1:], 'dpi:')
      for o, a in opts:
          if o == '-d':
              optdebug = True
          elif o == '-p':
              print_attachments = True
          elif o == '-i':
              indent_level = int (a)
  except getopt.GetoptError:
      sys.exit (0)

  # Registration.
  registrar.register_format (('imap', 'pop', 'mbox'))
  registrar.set_default_format ('mbox')

  if args:
      args = args[0]
  else:
      args = ''

  mbox = mailbox.MailboxDefault (args)

  # Debugging trace.
  if optdebug:
      mbox.debug.set_level (debug.MU_DEBUG_PROT)

  # Open the mailbox for reading only.
  mbox.open ()

  # Iterate through the entire message set.
  for i, msg in enumerate (mbox):
      print "Message: %d" % (i + 1)
      print "From: %s" % msg.header.get_value (MU_HEADER_FROM, "[none]")
      print "Subject: %s" % msg.header.get_value (MU_HEADER_SUBJECT, "[none]")
      print "Number of parts in message - %d" % msg.get_num_parts ()
      print "Total message size - %d/%d" % (msg.size, msg.lines)

      try:
          message_display_parts (msg, 0)
      except Error, e:
          print e

  mbox.close ()
