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

#include <mailutils/cpp/sieve.h>

using namespace mailutils;

//
// SieveMachine
//

SieveMachine :: SieveMachine ()
{
  int status = mu_sieve_machine_init (&mach, NULL);
  if (status)
    throw Exception ("SieveMachine::SieveMachine", status);
}

SieveMachine :: SieveMachine (const mu_sieve_machine_t mach)
{
  if (mach == 0)
    throw Exception ("SieveMachine::SieveMachine", EINVAL);

  this->mach = mach;
}

SieveMachine :: ~SieveMachine ()
{
  if (mach)
    mu_sieve_machine_destroy (&mach);
}

SieveMachine&
SieveMachine :: operator = (const SieveMachine& m)
{
  if (this != &m)
    {
      if (this->mach)
	mu_sieve_machine_destroy (&this->mach);
      mu_sieve_machine_dup (m.mach, &this->mach);
    }
  return *this;
}

void
SieveMachine :: compile (const std::string& name)
{
  int status = mu_sieve_compile (mach, name.c_str ());
  if (status)
    throw Exception ("SieveMachine::compile", status);
}

void
SieveMachine :: disass ()
{
  int status = mu_sieve_disass (mach);
  if (status)
    throw Exception ("SieveMachine::disass", status);
}

void
SieveMachine :: mailbox (const Mailbox& mbox)
{
  int status = mu_sieve_mailbox (mach, mbox.mbox);
  if (status)
    throw Exception ("SieveMachine::mailbox", status);
}

void
SieveMachine :: message (const Message& msg)
{
  int status = mu_sieve_message (mach, msg.msg);
  if (status)
    throw Exception ("SieveMachine::message", status);
}

void
SieveMachine :: set_debug (mu_sieve_printf_t printer)
{
  mu_sieve_set_debug (mach, printer);
}

void
SieveMachine :: set_error (mu_sieve_printf_t printer)
{
  mu_sieve_set_error (mach, printer);
}

void
SieveMachine :: set_parse_error (mu_sieve_parse_error_t printer)
{
  mu_sieve_set_parse_error (mach, printer);
}

void
SieveMachine :: set_logger (mu_sieve_action_log_t printer)
{
  mu_sieve_set_logger (mach, printer);
}

