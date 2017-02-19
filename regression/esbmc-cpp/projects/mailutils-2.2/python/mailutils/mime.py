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

from mailutils.c_api import mime
from mailutils import message
from mailutils.error import MimeError

MU_MIME_MULTIPART_MIXED = 0x1
MU_MIME_MULTIPART_ALT   = 0x2

class Mime:
    def __init__ (self, msg, flags = 0):
        self.mime = mime.MimeType ()
        status = mime.create (self.mime, msg.msg, flags)
        if status:
            raise MimeError (status)

    def __del__ (self):
        mime.destroy (self.mime)
        del self.mime

    def is_multipart (self):
        return mime.is_multipart (self.mime)

    def get_num_parts (self):
        status, nparts = mime.get_num_parts (self.mime)
        if status:
            raise MimeError (status)
        return nparts

    def get_part (self, npart):
        status, msg = mime.get_part (self.mime, npart)
        if status:
            raise MimeError (status)
        return message.Message (msg)

    def add_part (self, name, msg):
        status = mime.add_part (self.mime, msg.msg)
        if status:
            raise MimeError (status)

    def get_message (self):
        status, msg = mime.get_message (self.mime)
        if status:
            raise MimeError (status)
        return message.Message (msg)

def rfc2047_decode (tocode, text):
    return mime.rfc2047_decode (tocode, text)

def rfc2047_encode (charset, encoding, text):
    return mime.rfc2047_encode (charset, encoding, text)
