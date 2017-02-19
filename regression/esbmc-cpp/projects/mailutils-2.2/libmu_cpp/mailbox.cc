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

#include <mailutils/cpp/mailbox.h>

using namespace mailutils;

//
// MailboxBase
//

void
MailboxBase :: open ()
{
  int status = mu_mailbox_open (mbox, MU_STREAM_READ);
  if (status)
    throw Exception ("MailboxBase::open", status);
}

void
MailboxBase :: open (int flag)
{
  int status = mu_mailbox_open (mbox, flag);
  if (status)
    throw Exception ("MailboxBase::open", status);
}

void
MailboxBase :: close ()
{
  int status = mu_mailbox_close (mbox);
  if (status)
    throw Exception ("MailboxBase::close", status);
}

void
MailboxBase :: flush (bool expunge = false)
{
  int status = mu_mailbox_flush (mbox, expunge);
  if (status)
    throw Exception ("MailboxBase::flush", status);
}

size_t
MailboxBase :: messages_count ()
{
  size_t total;
  int status = mu_mailbox_messages_count (mbox, &total);
  if (status)
    throw Exception ("MailboxBase::messages_count", status);
  return total;
}

size_t
MailboxBase :: messages_recent ()
{
  size_t recent;
  int status = mu_mailbox_messages_recent (mbox, &recent);
  if (status)
    throw Exception ("MailboxBase::messages_recent", status);
  return recent;
}

size_t
MailboxBase :: message_unseen ()
{
  size_t unseen;
  int status = mu_mailbox_message_unseen (mbox, &unseen);
  if (status)
    throw Exception ("MailboxBase::message_unseen", status);
  return unseen;
}

Message&
MailboxBase :: get_message (size_t num)
{
  mu_message_t c_msg;

  int status = mu_mailbox_get_message (mbox, num, &c_msg);
  if (status)
    throw Exception ("MailboxBase::get_message", status);

  return *new Message (c_msg);
}

void
MailboxBase :: append_message (const Message& msg)
{
  int status = mu_mailbox_append_message (mbox, msg.msg);
  if (status)
    throw Exception ("MailboxBase::append_message", status);
}

void
MailboxBase :: expunge ()
{
  int status = mu_mailbox_expunge (mbox);
  if (status)
    throw Exception ("MailboxBase::expunge", status);
}

void
MailboxBase :: sync ()
{
  int status = mu_mailbox_sync (mbox);
  if (status)
    throw Exception ("MailboxBase::sync", status);
}

List&
MailboxBase :: get_uidls ()
{
  mu_list_t c_list;

  int status = mu_mailbox_get_uidls (mbox, &c_list);
  if (status)
    throw Exception ("MailboxBase::get_uidls", status);

  return *new List (c_list);
}

void
MailboxBase :: lock ()
{
  int status = mu_mailbox_lock (mbox);
  if (status)
    throw Exception ("MailboxBase::lock", status);
}

void
MailboxBase :: unlock ()
{
  int status = mu_mailbox_unlock (mbox);
  if (status)
    throw Exception ("MailboxBase::unlock", status);
}

mu_off_t
MailboxBase :: get_size ()
{
  mu_off_t size;
  int status = mu_mailbox_get_size (mbox, &size);
  if (status)
    throw Exception ("MailboxBase::get_size", status);
  return size;
}

Debug&
MailboxBase :: get_debug ()
{
  mu_debug_t c_dbg;

  int status = mu_mailbox_get_debug (mbox, &c_dbg);
  if (status)
    throw Exception ("MailboxBase::get_debug", status);

  return *new Debug (c_dbg);
}

Folder&
MailboxBase :: get_folder ()
{
  mu_folder_t c_folder;

  int status = mu_mailbox_get_folder (mbox, &c_folder);
  if (status)
    throw Exception ("MailboxBase::get_folder", status);

  return *new Folder (c_folder);
}

Url&
MailboxBase :: get_url ()
{
  mu_url_t c_url;

  int status = mu_mailbox_get_url (mbox, &c_url);
  if (status)
    throw Exception ("MailboxBase::get_url", status);

  return *new Url (c_url);
}

//
// Mailbox
//

Mailbox :: Mailbox (const std::string& name)
{
  int status = mu_mailbox_create (&mbox, name.c_str ());
  if (status)
    throw Exception ("Mailbox::Mailbox", status);
}

Mailbox :: Mailbox (const mu_mailbox_t mbox)
{
  if (mbox == 0)
    throw Exception ("Mailbox::Mailbox", EINVAL);

  this->mbox = mbox;
}

Mailbox :: ~Mailbox ()
{
  mu_mailbox_destroy (&mbox);
}

//
// MailboxDefault
//

MailboxDefault :: MailboxDefault ()
{
  int status = mu_mailbox_create_default (&mbox, NULL);
  if (status)
    throw Exception ("MailboxDefault::MailboxDefault", status);
}

MailboxDefault :: MailboxDefault (const std::string& name)
{
  int status = mu_mailbox_create_default (&mbox, name.c_str ());
  if (status)
    throw Exception ("MailboxDefault::MailboxDefault", status);
}

MailboxDefault :: MailboxDefault (const mu_mailbox_t mbox)
{
  if (mbox == 0)
    throw Exception ("MailboxDefault::MailboxDefault", EINVAL);

  this->mbox = mbox;
}

MailboxDefault :: ~MailboxDefault ()
{
  mu_mailbox_destroy (&mbox);
}

