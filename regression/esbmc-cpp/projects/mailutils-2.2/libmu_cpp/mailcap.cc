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

#include <mailutils/cpp/mailcap.h>

using namespace mailutils;

//
// Mailcap
//

Mailcap :: Mailcap (const Stream& stm)
{
  int status = mu_mailcap_create (&mailcap, stm.stm);
  if (status)
    throw Exception ("Mailcap::Mailcap", status);
}

Mailcap :: Mailcap (const mu_mailcap_t mailcap)
{
  if (mailcap == 0)
    throw Exception ("Mailcap::Mailcap", EINVAL);

  this->mailcap = mailcap;
}

Mailcap :: ~Mailcap ()
{
  mu_mailcap_destroy (&mailcap);
}

size_t
Mailcap :: entries_count ()
{
  size_t count = 0;
  int status = mu_mailcap_entries_count (mailcap, &count);
  if (status)
    throw Exception ("Mailcap::entries_count", status);
  return count;
}

MailcapEntry&
Mailcap :: get_entry (size_t i)
{
  mu_mailcap_entry_t c_entry;

  int status = mu_mailcap_get_entry (mailcap, i, &c_entry);
  if (status)
    throw Exception ("Mailcap::get_entry", status);

  MailcapEntry* entry = new MailcapEntry (c_entry);
  return *entry;
}

//
// MailcapEntry
//

MailcapEntry :: MailcapEntry (mu_mailcap_entry_t entry)
{
  if (entry == 0)
    throw Exception ("MailcapEntry::MailcapEntry", EINVAL);

  this->entry = entry;
}

size_t
MailcapEntry :: fields_count ()
{
  size_t count = 0;
  int status = mu_mailcap_entry_fields_count (entry, &count);
  if (status)
    throw Exception ("MailcapEntry::fields_count", status);
  return count;
}

std::string
MailcapEntry :: get_field (size_t i)
{
  int status = mu_mailcap_entry_get_field (entry, i, buf, 
					   sizeof (buf), NULL);
  if (status)
    throw Exception ("MailcapEntry::get_field", status);
  return std::string (buf);
}

std::string
MailcapEntry :: get_typefield ()
{
  int status = mu_mailcap_entry_get_typefield (entry, buf,
					       sizeof (buf), NULL);
  if (status)
    throw Exception ("MailcapEntry::get_typefield", status);
  return std::string (buf);
}

std::string
MailcapEntry :: get_viewcommand ()
{
  int status = mu_mailcap_entry_get_viewcommand (entry, buf, 
						 sizeof (buf), NULL);
  if (status)
    throw Exception ("MailcapEntry::get_viewcommand", status);
  return std::string (buf);
}

