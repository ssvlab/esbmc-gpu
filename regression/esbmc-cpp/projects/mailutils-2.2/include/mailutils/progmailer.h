/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2008, 2010 Free Software Foundation, Inc.

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

#ifndef _MAILUTILS_PROGMAILER_H
#define _MAILUTILS_PROGMAILER_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

int mu_progmailer_create (mu_progmailer_t *pm);
int mu_progmailer_set_command (mu_progmailer_t pm, const char *command);
int mu_progmailer_sget_command (mu_progmailer_t pm, const char **command);
  /* FIXME: missing _aget_ and _get_ */
int mu_progmailer_set_debug (mu_progmailer_t pm, mu_debug_t debug);  
int mu_progmailer_open (mu_progmailer_t pm, char **argv);
int mu_progmailer_send (mu_progmailer_t pm, mu_message_t msg);
int mu_progmailer_close (mu_progmailer_t pm);
void mu_progmailer_destroy (mu_progmailer_t *pm);

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_PROGMAILER_H */
  
