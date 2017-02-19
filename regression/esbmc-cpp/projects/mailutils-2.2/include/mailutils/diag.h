/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2010 Free Software
   Foundation, Inc.

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

#ifndef _MAILUTILS_DIAG_H
#define _MAILUTILS_DIAG_H

#include <stdarg.h>

#include <mailutils/types.h>
#include <mailutils/debug.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const char *mu_program_name;

#define MU_DIAG_EMERG    0
#define MU_DIAG_ALERT    1
#define MU_DIAG_CRIT     2
#define MU_DIAG_ERROR    3
#define MU_DIAG_ERR MU_DIAG_ERROR
#define MU_DIAG_WARNING  4 
#define MU_DIAG_NOTICE   5
#define MU_DIAG_INFO     6 
#define MU_DIAG_DEBUG    7
  
void mu_set_program_name (const char *);
void mu_diag_init (void);
void mu_diag_get_debug (mu_debug_t *);
void mu_diag_set_debug (mu_debug_t);
void mu_diag_vprintf (mu_log_level_t, const char *, va_list);
void mu_diag_printf (mu_log_level_t, const char *, ...) MU_PRINTFLIKE(2,3);
void mu_diag_voutput (mu_log_level_t, const char *, va_list);
void mu_diag_output (mu_log_level_t, const char *, ...) MU_PRINTFLIKE(2,3);

int mu_diag_syslog_printer (void *, mu_log_level_t, const char *);
int mu_diag_stderr_printer (void *, mu_log_level_t, const char *);

int mu_diag_level_to_syslog (mu_log_level_t level);
const char *mu_diag_level_to_string (mu_log_level_t level);

void mu_diag_funcall (mu_log_level_t level, const char *func,
		      const char *arg, int err);
  
#ifdef __cplusplus
}
#endif

#endif
