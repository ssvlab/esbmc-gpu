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

"""
A Python interface to Mailutils framework.

GNU Mailutils is a rich and powerful protocol-independent mail
framework.  It contains a series of useful mail libraries, clients,
and servers.  These are the primary mail utilities for the GNU system.
The central library is capable of handling electronic mail in various
mailbox formats and protocols, both local and remote.  Specifically,
this project contains a POP3 server, an IMAP4 server, and a Sieve mail
filter. It also provides a POSIX `mailx' client, and a collection of
other handy tools.

This software is part of the GNU Project and belongs to the Free
Software Foundation. All libraries are licensed using the GNU LGPL.
The documentation is licensed under the GNU FDL, and everything else
is licensed using the GNU GPL.

See http://www.gnu.org/software/mailutils/ for more information about
GNU Mailutils.

"""

__all__ = [
    "error",
    "address",
    "attribute",
    "auth",
    "body",
    "debug",
    "envelope",
    "filter",
    "folder",
    "header",
    "mailer",
    "mailbox",
    "mailcap",
    "message",
    "mime",
    "nls",
    "registrar",
    "secret",
    "sieve",
    "stream",
    "url",
    "util",
]
