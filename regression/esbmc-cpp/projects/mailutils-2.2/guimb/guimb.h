/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#if defined(HAVE_CONFIG_H)
# include <config.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>  /* strerror(3), strdup(3) */

#include <mailutils/errno.h>
#include <mailutils/mailbox.h>
#include <mailutils/message.h>
#include <mailutils/header.h>
#include <mailutils/body.h>
#include <mailutils/registrar.h>
#include <mailutils/error.h>
#include <mailutils/address.h>
#include <mailutils/registrar.h>
#include <mailutils/stream.h>
#include <mailutils/guile.h>
#include <mailutils/nls.h>
#include <mailutils/list.h>
#include <mailutils/mutil.h>
#include <mailutils/attribute.h>
#include <mailutils/envelope.h>
#include <mu_asprintf.h>

extern char *program_file;
extern char *program_expr;
extern char *user_name;
extern char *default_mailbox;
extern mu_mailbox_t mbox;
extern size_t nmesg;
extern size_t current_mesg_no;
extern mu_message_t current_message;
extern int debug_guile;
extern char *maildir;

void collect_open_default (void);
void collect_open_mailbox_file (void);
int collect_append_file (char *name);
void collect_create_mailbox (void);
void collect_drop_mailbox (void);
int collect_output (void);

void util_error (const char *fmt, ...) MU_PRINTFLIKE(1, 2);
int util_tempfile (char **namep);

