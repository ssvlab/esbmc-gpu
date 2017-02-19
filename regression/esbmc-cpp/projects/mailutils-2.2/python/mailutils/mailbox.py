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
from mailutils.c_api import mailbox
from mailutils import message
from mailutils import folder
from mailutils import url
from mailutils import debug
from mailutils.error import MailboxError

class MailboxBase:
    def open (self, mode = 0):
        """Open the connection.

        'mode' may be a string, consisting of the characters described
        below, giving the access mode for the mailbox.

        mode   Meaning
        -------------------------------------------------------- 
        r      Open for reading.
        w      Open for writing.
        a      Open for appending to the end of the mailbox.
        c      Create the mailbox if it does not exist.

        """
        if isinstance (mode, types.StringType):
            from mailutils import stream
            flags = 0
            for m in mode:
                if m == 'r':
                    flags = flags | stream.MU_STREAM_READ
                elif m == 'w':
                    flags = flags | stream.MU_STREAM_WRITE
                elif m == 'a':
                    flags = flags | stream.MU_STREAM_APPEND
                elif m == 'c':
                    flags = flags | stream.MU_STREAM_CREAT
            if flags & stream.MU_STREAM_READ and flags & stream.MU_STREAM_WRITE:
                flags = (flags & ~(stream.MU_STREAM_READ | \
                                   stream.MU_STREAM_WRITE)) | \
                                   stream.MU_STREAM_RDWR
            mode = flags
        status = mailbox.open (self.mbox, mode)
        if status:
            raise MailboxError (status)

    def close (self):
        """Close the connection."""
        status = mailbox.close (self.mbox)
        if status:
            raise MailboxError (status)

    def flush (self, expunge = False):
        """Flush the mailbox."""
        status = mailbox.flush (self.mbox, expunge)
        if status:
            raise MailboxError (status)

    def messages_count (self):
        """Return the number of messages in mailbox."""
        status, total = mailbox.messages_count (self.mbox)
        if status:
            raise MailboxError (status)
        return total

    def messages_recent (self):
        """Return the number of recent messages in mailbox."""
        status, recent = mailbox.messages_recent (self.mbox)
        if status:
            raise MailboxError (status)
        return recent

    def message_unseen (self):
        """Return the number of first unseen message in mailbox."""
        status, recent = mailbox.message_unseen (self.mbox)
        if status:
            raise MailboxError (status)
        return unseen

    def get_message (self, msgno):
        """Retrieve message number 'msgno'."""
        status, c_msg = mailbox.get_message (self.mbox, msgno)
        if status:
            raise MailboxError (status)
        return message.Message (c_msg)

    def append_message (self, msg):
        """Append 'msg' to the mailbox."""
        status = mailbox.append_message (self.mbox, msg.msg)
        if status:
            raise MailboxError (status)

    def expunge (self):
        """Remove all messages marked for deletion."""
        status = mailbox.expunge (self.mbox)
        if status:
            raise MailboxError (status)

    def sync (self):
        """Synchronize the mailbox."""
        status = mailbox.sync (self.mbox)
        if status:
            raise MailboxError (status)

    def get_uidls (self):
        """Get UIDL list."""
        status, uidls = mailbox.get_uidls (self.mbox)
        if status:
            raise MailboxError (status)
        return uidls

    def lock (self):
        """Lock the mailbox."""
        status = mailbox.lock (self.mbox)
        if status:
            raise MailboxError (status)

    def unlock (self):
        """Unlock the mailbox."""
        status = mailbox.unlock (self.mbox)
        if status:
            raise MailboxError (status)

    def get_size (self):
        """Return the mailbox size."""
        status, size = mailbox.get_size (self.mbox)
        if status:
            raise MailboxError (status)
        return size

    def get_folder (self):
        """Get the Folder object."""
        status, fld = mailbox.get_folder (self.mbox)
        if status:
            raise MailboxError (status)
        return folder.Folder (fld)

    def get_debug (self):
        """Get the Debug object."""
        status, dbg = mailbox.get_debug (self.mbox)
        if status:
            raise MailboxError (status)
        return debug.Debug (dbg)

    def get_url (self):
        """Get the Url object."""
        status, u = mailbox.get_url (self.mbox)
        if status:
            raise MailboxError (status)
        return url.Url (u)

    def next (self):
        if self.__count >= self.__len:
            self.__count = 0
            raise StopIteration
        else:
            self.__count += 1
            return self.get_message (self.__count)

    def __getitem__ (self, msgno):
        return self.get_message (msgno)

    def __iter__ (self):
        self.__count = 0
        self.__len = self.messages_count ()
        return self

    def __getattr__ (self, name):
        if name == 'size':
            return self.get_size ()
        elif name == 'folder':
            return self.get_folder ()
        elif name == 'url':
            return self.get_url ()
        elif name == 'debug':
            return self.get_debug ()
        else:
            raise AttributeError, name

    def __len__ (self):
        return self.messages_count ()

    def __str__ (self):
        return '<Mailbox %s (%d)>' % (self.get_url (), self.messages_count ())

class Mailbox (MailboxBase):
    __owner = False
    def __init__ (self, name):
        if isinstance (name, mailbox.MailboxType):
            self.mbox = name
        else:
            self.mbox = mailbox.MailboxType ()
            self.__owner = True
            status = mailbox.create (self.mbox, name)
            if status:
                raise MailboxError (status)

    def __del__ (self):
        if self.__owner:
            mailbox.destroy (self.mbox)
        del self.mbox

class MailboxDefault (MailboxBase):
    def __init__ (self, name = None):
        """MailboxDefault creates a Mailbox object for the supplied
        mailbox 'name'. Before creating, the name is expanded using
        the rules below:

        %           --> system mailbox for the real uid
        %user       --> system mailbox for the given user
        ~/file      --> /home/user/file
        ~user/file  --> /home/user/file
        +file       --> /home/user/Mail/file
        =file       --> /home/user/Mail/file
        """
        self.mbox = mailbox.MailboxType ()
        status = mailbox.create_default (self.mbox, name)
        if status:
            raise MailboxError (status)

    def __del__ (self):
        mailbox.destroy (self.mbox)
        del self.mbox
