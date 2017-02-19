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

import types
from mailutils.c_api import header
from mailutils.error import *

class Header:
    def __init__ (self, hdr):
        self.hdr = hdr

    def __del__ (self):
        del self.hdr

    def __getitem__ (self, item):
        if isinstance (item, types.IntType):
            return self.get_field_name (item), self.get_field_value (item)
        else:
            return self.get_value (item)

    def __setitem__ (self, name, value):
        self.set_value (name, value)

    def __contains__ (self, name):
        status, value = header.get_value (self.hdr, name)
        if status == MU_ERR_NOENT:
            return False
        elif status:
            raise HeaderError (status)
        return True

    def __getattr__ (self, name):
        if name == 'size':
            return self.get_size ()
        elif name == 'lines':
            return self.get_lines ()
        else:
            raise AttributeError, name

    def __len__ (self):
        return self.get_field_count ()

    def __iter__ (self):
        self.__count = 0
        self.__len = self.get_field_count ()
        return self

    def next (self):
        if self.__count >= self.__len:
            self.__count = 0
            raise StopIteration
        else:
            self.__count += 1
            return self.__getitem__ (self.__count)

    def has_key (self, name):
        return self.__contains__ (name)

    def get_size (self):
        status, size = header.size (self.hdr)
        if status:
            raise HeaderError (status)
        return size

    def get_lines (self):
        status, lines = header.lines (self.hdr)
        if status:
            raise HeaderError (status)
        return lines

    def get_value (self, name, default = None):
        status, value = header.get_value (self.hdr, name)
        if status == MU_ERR_NOENT:
            if default != None:
                return default
            else:
                raise KeyError (name)
        elif status:
            raise HeaderError (status)
        return value

    def get_value_n (self, name, n = 1, default = None):
        status, value = header.get_value_n (self.hdr, name, n)
        if status == MU_ERR_NOENT:
            if default != None:
                return default
            else:
                raise KeyError (name)
        elif status:
            raise HeaderError (status)
        return value

    def set_value (self, name, value, replace = True):
        status = header.set_value (self.hdr, name, value, replace)
        if status:
            raise HeaderError (status)

    def get_field_count (self):
        status, count = header.get_field_count (self.hdr)
        if status:
            raise HeaderError (status)
        return count

    def get_field_name (self, idx):
        status, name = header.get_field_name (self.hdr, idx)
        if status == MU_ERR_NOENT:
            raise IndexError (idx)
        elif status:
            raise HeaderError (status)
        return name

    def get_field_value (self, idx):
        status, value = header.get_field_value (self.hdr, idx)
        if status == MU_ERR_NOENT:
            raise IndexError (idx)
        elif status:
            raise HeaderError (status)
        return value

MU_HEADER_UNIX_FROM =                 "From "
MU_HEADER_RETURN_PATH =               "Return-Path"
MU_HEADER_RECEIVED =                  "Received"
MU_HEADER_DATE =                      "Date"
MU_HEADER_DCC =                       "Dcc"
MU_HEADER_FROM =                      "From"
MU_HEADER_SENDER =                    "Sender"
MU_HEADER_RESENT_FROM =               "Resent-From"
MU_HEADER_SUBJECT =                   "Subject"
MU_HEADER_SENDER =                    "Sender"
MU_HEADER_RESENT_SENDER =             "Resent-SENDER"
MU_HEADER_TO =                        "To"
MU_HEADER_RESENT_TO =                 "Resent-To"
MU_HEADER_CC =                        "Cc"
MU_HEADER_RESENT_CC =                 "Resent-Cc"
MU_HEADER_BCC =                       "Bcc"
MU_HEADER_RESENT_BCC =                "Resent-Bcc"
MU_HEADER_REPLY_TO =                  "Reply-To"
MU_HEADER_RESENT_REPLY_TO =           "Resent-Reply-To"
MU_HEADER_MESSAGE_ID =                "Message-ID"
MU_HEADER_RESENT_MESSAGE_ID =         "Resent-Message-ID"
MU_HEADER_IN_REPLY_TO =               "In-Reply-To"
MU_HEADER_REFERENCE =                 "Reference"
MU_HEADER_REFERENCES =                "References"
MU_HEADER_ENCRYPTED =                 "Encrypted"
MU_HEADER_PRECEDENCE =                "Precedence"
MU_HEADER_STATUS =                    "Status"
MU_HEADER_CONTENT_LENGTH =            "Content-Length"
MU_HEADER_CONTENT_LANGUAGE =          "Content-Language"
MU_HEADER_CONTENT_TRANSFER_ENCODING = "Content-transfer-encoding"
MU_HEADER_CONTENT_ID =                "Content-ID"
MU_HEADER_CONTENT_TYPE =              "Content-Type"
MU_HEADER_CONTENT_DESCRIPTION =       "Content-Description"
MU_HEADER_CONTENT_DISPOSITION =       "Content-Disposition"
MU_HEADER_CONTENT_MD5 =               "Content-MD5"
MU_HEADER_MIME_VERSION =              "MIME-Version"
MU_HEADER_X_MAILER =                  "X-Mailer"
MU_HEADER_X_UIDL =                    "X-UIDL"
MU_HEADER_X_UID =                     "X-UID"
MU_HEADER_X_IMAPBASE =                "X-IMAPbase"
MU_HEADER_ENV_SENDER =                "X-Envelope-Sender"
MU_HEADER_ENV_DATE =                  "X-Envelope-Date"
MU_HEADER_FCC =                       "Fcc"
MU_HEADER_DELIVERY_DATE =             "Delivery-date"
MU_HEADER_ENVELOPE_TO =               "Envelope-to"
MU_HEADER_X_EXPIRE_TIMESTAMP =        "X-Expire-Timestamp"

MU_HEADER_REPLACE = 0x01
MU_HEADER_BEFORE  = 0x02
