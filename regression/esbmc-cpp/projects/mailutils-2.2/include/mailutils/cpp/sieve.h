/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

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

#ifndef _MUCPP_SIEVE_H
#define _MUCPP_SIEVE_H

#include <string>
#include <errno.h>
#include <mailutils/sieve.h>
#include <mailutils/cpp/error.h>
#include <mailutils/cpp/mailbox.h>
#include <mailutils/cpp/message.h>

namespace mailutils
{

class SieveMachine
{
 protected:
  mu_sieve_machine_t mach;

 public:
  SieveMachine ();
  SieveMachine (const mu_sieve_machine_t);
  ~SieveMachine ();

  SieveMachine& operator = (const SieveMachine&);

  void compile (const std::string& name);
  void disass ();
  void mailbox (const Mailbox& mbox);
  void message (const Message& msg);
  void set_debug (mu_sieve_printf_t printer);
  void set_error (mu_sieve_printf_t printer);
  void set_parse_error (mu_sieve_parse_error_t printer);
  void set_logger (mu_sieve_action_log_t printer);
};

}

#endif // not _MUCPP_SIEVE_H

