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
from mailutils.c_api import auth
from mailutils.error import AuthError

MU_AF_QUOTA = 0x1

def register_module (name = None):
    if name == None:
        status = auth.register_module ()
    elif isinstance (name, types.TupleType) \
      or isinstance (name, types.ListType):
        for n in name:
            status = auth.register_module (n)
    else:
        status = auth.register_module (name)
    if status:
        raise AuthError (status)

def get_auth_by_name (username):
    return auth.get_auth_by_name (username)

def get_auth_by_uid (uid):
    return auth.get_auth_by_uid (uid)

def authenticate (auth_data, password):
    status = auth.authenticate (auth_data, password)
    if status:
        raise AuthError (status)

def set_pam_service (pam_service):
    auth.set_pam_service (pam_service)


class Authority:
    __owner = False

    def __init__ (self, authority = None):
        if isinstance (authority, auth.AuthorityType):
            self.authority = authority
        else:
            self.authority = auth.AuthorityType ()
            self.__owner = True
            status = auth.authority_create (self.authority)
            if status:
                raise AuthError (status)

    def __del__ (self):
        if self.__owner:
            auth.authority_destroy (self.authority)
        del self.authority

    def get_ticket (self):
        status, ticket = auth.authority_get_ticket (self.authority)
        if status:
            raise AuthError (status)
        return Ticket (ticket)

    def set_ticket (self, ticket):
        status = auth.authority_set_ticket (self.authority, ticket.ticket)
        if status:
            raise AuthError (status)

    def authenticate (self):
        status = auth.authority_authenticate (self.authority)
        if status:
            raise AuthError (status)

class Ticket:
    __owner = False

    def __init__ (self, ticket = None):
        if isinstance (ticket, auth.TicketType):
            self.ticket = ticket
        else:
            self.ticket = auth.TicketType ()
            self.__owner = True
            status = auth.ticket_create (self.ticket)
            if status:
                raise AuthError (status)

    def __del__ (self):
        if self.__owner:
            auth.ticket_destroy (self.ticket)
        del self.ticket

    def set_secret (self, secret):
        status = auth.ticket_set_secret (self.ticket, secret.secret)
        if status:
            raise AuthError (status)

class Wicket:
    __owner = False

    def __init__ (self, wicket = None):
        if isinstance (wicket, auth.WicketType):
            self.wicket = wicket
        else:
            self.wicket = auth.WicketType ()
            self.__owner = True
            status = auth.wicket_create (self.wicket)
            if status:
                raise AuthError (status)

    def __del__ (self):
        if self.__owner:
            auth.wicket_destroy (self.wicket)
        del self.wicket

    def get_ticket (self, user):
        status, ticket = auth.wicket_get_ticket (self.wicket, user)
        if status:
            raise AuthError (status)
        return Ticket (ticket)
