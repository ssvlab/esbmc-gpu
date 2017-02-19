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

#include <mailutils/cpp/address.h>
#include <mailutils/cpp/attribute.h>
#include <mailutils/cpp/body.h>
#include <mailutils/cpp/error.h>
#include <mailutils/cpp/filter.h>
#include <mailutils/cpp/folder.h>
#include <mailutils/cpp/header.h>
#include <mailutils/cpp/iterator.h>
#include <mailutils/cpp/list.h>
#include <mailutils/cpp/mailbox.h>
#include <mailutils/cpp/mailcap.h>
#include <mailutils/cpp/mailer.h>
#include <mailutils/cpp/message.h>
#include <mailutils/cpp/mime.h>
#include <mailutils/cpp/mutil.h>
#include <mailutils/cpp/pop3.h>
#include <mailutils/cpp/registrar.h>
#include <mailutils/cpp/secret.h>
#include <mailutils/cpp/sieve.h>
#include <mailutils/cpp/stream.h>
#include <mailutils/cpp/url.h>

