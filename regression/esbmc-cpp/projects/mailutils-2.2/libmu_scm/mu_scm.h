/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2006, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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
   Boston, MA 02110-1301 USA */

#if defined(HAVE_CONFIG_H)
# include <config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <string.h>
#include <confpaths.h>

#include <mailutils/mailbox.h>
#include <mailutils/message.h>
#include <mailutils/header.h>
#include <mailutils/body.h>
#include <mailutils/registrar.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/address.h>
#include <mailutils/registrar.h>
#include <mailutils/mutil.h>
#include <mailutils/stream.h>
#include <mailutils/debug.h>
#include <mailutils/attribute.h>
#include <mailutils/mailer.h>
#include <mailutils/envelope.h>
#include <mailutils/url.h>
#include <mailutils/mime.h>
#include <mailutils/registrar.h>
#include <mailutils/mu_auth.h>
#include <mailutils/cstr.h>

#include <mailutils/guile.h>

