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

from mailutils.c_api import stream
from mailutils.error import StreamError

MU_STREAM_READ =	0x00000001
MU_STREAM_WRITE =       0x00000002
MU_STREAM_RDWR =        0x00000004
MU_STREAM_APPEND =      0x00000008
MU_STREAM_CREAT =       0x00000010
MU_STREAM_NONBLOCK =    0x00000020
MU_STREAM_NO_CHECK =    0x00000040
MU_STREAM_SEEKABLE =    0x00000080
MU_STREAM_NO_CLOSE =    0x00000100
MU_STREAM_ALLOW_LINKS = 0x00000200
MU_STREAM_NONLOCK =     0x00000400
MU_STREAM_QACCESS =     0x00000800
MU_STREAM_IRGRP =       0x00001000
MU_STREAM_IWGRP =       0x00002000
MU_STREAM_IROTH =       0x00004000
MU_STREAM_IWOTH =       0x00008000
MU_STREAM_IMASK =       0x0000F000

class Stream:
    __refcount = 0

    def __init__ (self, stm = None):
        if isinstance (stm, stream.StreamType):
            self.stm = stm
        else:
            self.stm = stream.StreamType ()
        self.__reference ()
        self.read_count = 0
        self.write_count = 0

    def __del__ (self):
        if self.__dereference ():
            stream.destroy (self.stm)
        del self.stm

    def __reference (self):
        self.__refcount += 1

    def __dereference (self):
        self.__refcount -= 1
        return self.__refcount == 0

    def open (self):
        status = stream.open (self.stm)
        if status:
            raise StreamError (status)

    def close (self):
        status = stream.close (self.stm)
        if status:
            raise StreamError (status)

    def flush (self):
        status = stream.flush (self.stm)
        if status:
            raise StreamError (status)

    def wait (self, wflags):
        status = stream.wait (self.stm, wflags)
        if status:
            raise StreamError (status)

    def read (self, offset = 0):
        status, rbuf, self.read_count = stream.read (self.stm, offset)
        if status:
            raise StreamError (status)
        return rbuf

    def write (self, wbuf, offset = 0):
        status, self.write_count = stream.write (self.stm, wbuf, offset)
        if status:
            raise StreamError (status)

    def readline (self, offset = 0):
        status, rbuf, self.read_count = stream.readline (self.stm, offset)
        if status:
            raise StreamError (status)
        return rbuf

    def sequential_readline (self):
        status, rbuf, self.read_count = stream.readline (self.stm)
        if status:
            raise StreamError (status)
        return rbuf

    def sequential_write (self, wbuf, size = None):
        if size == None:
            size = len (wbuf)
        status = stream.sequential_write (self.stm, wbuf, size)
        if status:
            raise StreamError (status)

class TcpStream (Stream):
    def __init__ (self, host, port, flags = MU_STREAM_READ):
        Stream.__init__ (self)
        status = stream.tcp_stream_create (self.stm, host, port, flags)
        if status:
            raise StreamError (status)

class FileStream (Stream):
    def __init__ (self, filename, flags = MU_STREAM_READ):
        Stream.__init__ (self)
        status = stream.file_stream_create (self.stm, filename, flags)
        if status:
            raise StreamError (status)

class StdioStream (Stream):
    def __init__ (self, file, flags = MU_STREAM_READ):
        Stream.__init__ (self)
        status = stream.stdio_stream_create (self.stm, file, flags)
        if status:
            raise StreamError (status)

class ProgStream (Stream):
    def __init__ (self, progname, flags = MU_STREAM_READ):
        Stream.__init__ (self)
        status = stream.prog_stream_create (self.stm, progname, flags)
        if status:
            raise StreamError (status)
