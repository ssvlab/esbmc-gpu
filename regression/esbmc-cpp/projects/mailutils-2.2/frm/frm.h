/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2007, 2010 Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#ifdef HAVE_TERMIOS_H
# include <termios.h>
#endif
#include <sys/ioctl.h>
#include <sys/stat.h>

#ifdef HAVE_ICONV_H
# include <iconv.h>
#endif
#ifndef MB_LEN_MAX
# define MB_LEN_MAX 4
#endif

#include <mbswidth.h>
#include <xalloc.h>

#ifdef HAVE_FRIBIDI_FRIBIDI_H
# include <fribidi/fribidi.h>
#endif

#include <mailutils/address.h>
#include <mailutils/attribute.h>
#include <mailutils/debug.h>
#include <mailutils/errno.h>
#include <mailutils/header.h>
#include <mailutils/list.h>
#include <mailutils/mailbox.h>
#include <mailutils/message.h>
#include <mailutils/observer.h>
#include <mailutils/registrar.h>
#include <mailutils/stream.h>
#include <mailutils/url.h>
#include <mailutils/nls.h>
#include <mailutils/tls.h>
#include <mailutils/error.h>
#include <mailutils/mutil.h>
#include <mailutils/mime.h>

#include "mailutils/libargp.h"

typedef int (*frm_select_t) (size_t index, mu_message_t msg);

extern char *show_field;   /* Show this header field instead of the default
			      `From: Subject:' pair. -f option */
extern int show_to;        /* Additionally display To: field. -l option */ 
extern int show_number;    /* Prefix each line with the message number. -n */
extern int frm_debug;

extern void frm_scan (char *mailbox_name, frm_select_t fun, size_t *total);
extern int util_getcols (void);
extern void init_output (size_t s);


