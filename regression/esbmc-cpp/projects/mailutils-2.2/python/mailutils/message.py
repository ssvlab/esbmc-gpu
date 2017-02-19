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

from mailutils.c_api import message
from mailutils import envelope
from mailutils import header
from mailutils import body
from mailutils import attribute
from mailutils.error import MessageError

class Message:
    __owner = False

    def __init__ (self, msg = None):
        if msg == None:
            self.msg = message.MessageType ()
            self.__owner = True
            status = message.create (self.msg)
            if status:
                raise MessageError (status)
        else:
            self.msg = msg

    def __del__ (self):
        if self.__owner:
            message.destroy (self.msg)
        del self.msg

    def __str__ (self):
        try:
            env = self.get_envelope ()
            envelope = '%s %s' % (env.get_sender ().strip (),
                                  env.get_date ().strip ())
        except MessageError:
            envelope = 'UNKNOWN'
        return '<Message "%s" %d %d>' % (envelope, self.get_lines (),
                                         self.get_size ())

    def __getattr__ (self, name):
        if name == 'header':
            return self.get_header ()
        elif name == 'body':
            return self.get_body ()
        elif name == 'envelope':
            return self.get_envelope ()
        elif name == 'attribute':
            return self.get_attribute ()
        elif name == 'size':
            return self.get_size ()
        elif name == 'lines':
            return self.get_lines ()
        else:
            raise AttributeError, name

    def __len__ (self):
        return self.get_size ()

    def is_multipart (self):
        status, ismulti = message.is_multipart (self.msg)
        if status:
            raise MessageError (status)
        return ismulti

    def get_size (self):
        status, size = message.size (self.msg)
        if status:
            raise MessageError (status)
        return size

    def get_lines (self):
        status, lines = message.lines (self.msg)
        if status:
            raise MessageError (status)
        return lines

    def get_envelope (self):
        status, env = message.get_envelope (self.msg)
        if status:
            raise MessageError (status)
        return envelope.Envelope (env)

    def get_header (self):
        status, hdr = message.get_header (self.msg)
        if status:
            raise MessageError (status)
        return header.Header (hdr)

    def get_body (self):
        status, bd = message.get_body (self.msg)
        if status:
            raise MessageError (status)
        return body.Body (bd)

    def get_attribute (self):
        status, attr = message.get_attribute (self.msg)
        if status:
            raise MessageError (status)
        return attribute.Attribute (attr)

    def get_num_parts (self):
        status, num_parts = message.get_num_parts (self.msg)
        if status:
            raise MessageError (status)
        return num_parts

    def get_part (self, npart):
        status, part = message.get_part (self.msg, npart)
        if status:
            raise MessageError (status)
        return Message (part)

    def get_uid (self):
        status, uid = message.get_uid (self.msg)
        if status:
            raise MessageError (status)
        return uid

    def get_uidl (self):
        status, uidl = message.get_uidl (self.msg)
        if status:
            raise MessageError (status)
        return uidl

    def get_attachment_name (self, charset=None):
        status, name, lang = message.get_attachment_name (self.msg, charset)
        if status:
            raise MessageError (status)
        return name, lang

    def save_attachment (self, filename = ''):
        status = message.save_attachment (self.msg, filename)
        if status:
            raise MessageError (status)

    def unencapsulate (self):
        status, msg = message.unencapsulate (self.msg)
        if status:
            raise MessageError (status)
        return Message (msg)

    def set_stream (self, stream):
        status = message.set_stream (self.msg, stream.stm)
        if status:
            raise MessageError (status)
