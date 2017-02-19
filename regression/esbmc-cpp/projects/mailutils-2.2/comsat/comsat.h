/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2007, 2008, 2009, 2010 Free
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <errno.h>
#include <termios.h>
#include <syslog.h>
#include <string.h>
#include <pwd.h>

#include <confpaths.h>

#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/body.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/header.h>
#include <mailutils/list.h>
#include <mailutils/mailbox.h>
#include <mailutils/message.h>
#include <mailutils/mutil.h>
#include <mailutils/registrar.h>
#include <mailutils/stream.h>
#include <mailutils/mu_auth.h>
#include <mailutils/argcv.h>
#include <mailutils/nls.h>
#include <mailutils/daemon.h>
#include <mailutils/acl.h>
#include <mailutils/server.h>
#include <mailutils/cctype.h>

#ifndef INADDR_NONE
# define INADDR_NONE -1
#endif

#define BIFF_RC ".biffrc"

extern int allow_biffrc;
extern unsigned maxrequests;
extern time_t request_control_interval;
extern time_t overflow_control_interval;
extern time_t overflow_delay_time;
extern int maxlines;
extern const char *username;
extern char hostname[];
extern struct daemon_param daemon_param;

void run_user_action (FILE *tty, const char *cr, mu_message_t msg);
void convert_config (const char *config_file, FILE *outfile);

