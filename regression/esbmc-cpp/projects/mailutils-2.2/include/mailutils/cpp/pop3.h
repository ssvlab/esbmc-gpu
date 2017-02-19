/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2006, 2007, 2009, 2010 Free Software Foundation,
   Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA
*/

#ifndef _MUCPP_POP3_H
#define _MUCPP_POP3_H

#include <errno.h>
#include <mailutils/pop3.h>
#include <mailutils/cpp/list.h>
#include <mailutils/cpp/iterator.h>
#include <mailutils/cpp/stream.h>

namespace mailutils
{

class Pop3
{
 protected:
  mu_pop3_t pop3;
  Stream* pStream;

 public:
  Pop3 ();
  Pop3 (const mu_pop3_t);
  ~Pop3 ();

  void set_carrier (const Stream& carrier);
  Stream& get_carrier ();
  void connect ();
  void disconnect ();
  void set_timeout (int timeout);
  int get_timeout ();

  void apop (const char* name, const char* digest);
  void stls ();
  Iterator& capa ();
  void dele (unsigned int msgno);
  size_t list (unsigned int msgno);
  Iterator& list_all ();
  void noop ();
  void pass (const char* pass);
  void quit ();
  Stream& retr (unsigned int msgno);
  void rset ();
  void stat (unsigned int* count, size_t* octets);
  Stream& top (unsigned int msgno, unsigned int lines);
  std::string uidl  (unsigned int msgno);
  Iterator& uidl_all ();
  void user (const char* user);

  size_t readline (char* buf, size_t buflen);
  size_t response (char* buf, size_t buflen);
  void sendline (const char* line);
  void send ();
};

}

#endif // not _MUCPP_POP3_H

