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

#include <cstdlib>
#include <mailutils/cpp/pop3.h>

using namespace mailutils;

//
// POP3
//

Pop3 :: Pop3 ()
{
  int status = mu_pop3_create (&pop3);
  if (status)
    throw Exception ("Pop3::Pop3", status);

  this->pStream = 0;
}

Pop3 :: Pop3 (const mu_pop3_t pop3)
{
  if (pop3 == 0)
    throw Exception ("Pop3::Pop3", EINVAL);

  this->pop3 = pop3;
  this->pStream = 0;
}

Pop3 :: ~Pop3 ()
{
  mu_pop3_destroy (&pop3);
}

void
Pop3 :: set_carrier (const Stream& carrier)
{
  int status = mu_pop3_set_carrier (pop3, carrier.stm);
  if (status)
    throw Exception ("Pop3::set_carrier", status);

  this->pStream = (Stream*) &carrier;
}

Stream&
Pop3 :: get_carrier ()
{
  return *pStream;
}

void
Pop3 :: connect ()
{
  int status = mu_pop3_connect (pop3);
  if (status)
    throw Exception ("Pop3::connect", status);
}

void
Pop3 :: disconnect ()
{
  int status = mu_pop3_disconnect (pop3);
  if (status)
    throw Exception ("Pop3::disconnect", status);
}

void
Pop3 :: set_timeout (int timeout)
{
  int status = mu_pop3_set_timeout (pop3, timeout);
  if (status)
    throw Exception ("Pop3::set_timeout", status);
}

int
Pop3 :: get_timeout ()
{
  int timeout;

  int status = mu_pop3_get_timeout (pop3, &timeout);
  if (status)
    throw Exception ("Pop3::get_timeout", status);

  return timeout;
}

void
Pop3 :: apop (const char* name, const char* digest)
{
  int status = mu_pop3_apop (pop3, name, digest);
  if (status)
    throw Exception ("Pop3::apop", status);
}

void
Pop3 :: stls ()
{
  int status = mu_pop3_stls (pop3);
  if (status)
    throw Exception ("Pop3::stls", status);
}

Iterator&
Pop3 :: capa ()
{
  mu_iterator_t mu_itr;

  int status = mu_pop3_capa (pop3, &mu_itr);
  if (status)
    throw Exception ("Pop3::capa", status);

  return *new Iterator (mu_itr);
}

void
Pop3 :: dele (unsigned int msgno)
{
  int status = mu_pop3_dele (pop3, msgno);
  if (status)
    throw Exception ("Pop3::dele", status);
}

size_t
Pop3 :: list (unsigned int msgno)
{
  size_t msg_octet;

  int status = mu_pop3_list (pop3, msgno, &msg_octet);
  if (status)
    throw Exception ("Pop3::list", status);

  return msg_octet;
}

Iterator&
Pop3 :: list_all ()
{
  mu_iterator_t mu_itr;

  int status = mu_pop3_list_all (pop3, &mu_itr);
  if (status)
    throw Exception ("Pop3::list_all", status);

  return *new Iterator (mu_itr);
}

void
Pop3 :: noop ()
{
  int status = mu_pop3_noop (pop3);
  if (status)
    throw Exception ("Pop3::noop", status);
}

void
Pop3 :: pass (const char* pass)
{
  int status = mu_pop3_pass (pop3, pass);
  if (status)
    throw Exception ("Pop3::pass", status);
}

void
Pop3 :: quit ()
{
  int status = mu_pop3_quit (pop3);
  if (status)
    throw Exception ("Pop3::quit", status);
}

Stream&
Pop3 :: retr (unsigned int msgno)
{
  mu_stream_t c_stm;

  int status = mu_pop3_retr (pop3, msgno, &c_stm);
  if (status)
    throw Exception ("Pop3::retr", status);

  return *new Stream (c_stm);
}

void
Pop3 :: rset ()
{
  int status = mu_pop3_rset (pop3);
  if (status)
    throw Exception ("Pop3::rset", status);
}

void
Pop3 :: stat (unsigned int* count, size_t* octets)
{
  int status = mu_pop3_stat (pop3, count, octets);
  if (status)
    throw Exception ("Pop3::stat", status);
}

Stream&
Pop3 :: top (unsigned int msgno, unsigned int lines)
{
  mu_stream_t c_stm;

  int status = mu_pop3_top (pop3, msgno, lines, &c_stm);
  if (status)
    throw Exception ("Pop3::top", status);

  return *new Stream (c_stm);
}

std::string
Pop3 :: uidl  (unsigned int msgno)
{
  char *c_uidl = NULL;

  int status = mu_pop3_uidl (pop3, msgno, &c_uidl);
  if (status)
    throw Exception ("Pop3::uidl", status);

  if (c_uidl) {
    std::string uidl (c_uidl);
    free (c_uidl);
    return uidl;
  }
  return NULL;
}

Iterator&
Pop3 :: uidl_all ()
{
  mu_iterator_t mu_itr;

  int status = mu_pop3_uidl_all (pop3, &mu_itr);
  if (status)
    throw Exception ("Pop3::uidl_all", status);

  return *new Iterator (mu_itr);
}

void
Pop3 :: user (const char* user)
{
  int status = mu_pop3_user (pop3, user);
  if (status)
    throw Exception ("Pop3::user", status);
}

size_t
Pop3 :: readline (char* buf, size_t buflen)
{
  size_t nread;

  int status = mu_pop3_readline (pop3, buf, buflen, &nread);
  if (status)
    throw Exception ("Pop3::readline", status);
}

size_t
Pop3 :: response (char* buf, size_t buflen)
{
  size_t nread;

  int status = mu_pop3_response (pop3, buf, buflen, &nread);
  if (status)
    throw Exception ("Pop3::response", status);
}

void
Pop3 :: sendline (const char* line)
{
  int status = mu_pop3_sendline (pop3, line);
  if (status)
    throw Exception ("Pop3::sendline", status);
}

void
Pop3 :: send ()
{
  int status = mu_pop3_send (pop3);
  if (status)
    throw Exception ("Pop3::send", status);
}

