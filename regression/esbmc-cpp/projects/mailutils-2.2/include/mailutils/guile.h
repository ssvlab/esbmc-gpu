/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2006, 2007, 2009, 2010
   Free Software Foundation, Inc.

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

#ifndef _MU_SCM_H
#define _MU_SCM_H

#include <libguile.h>

#define MU_SCM_SYMBOL_VALUE(p) SCM_VARIABLE_REF(scm_c_lookup(p))

typedef struct
{
  int debug_guile;
  mu_mailbox_t mbox;
  char *user_name;
  int (*init) (void *data);
  SCM (*catch_body) (void *data, mu_mailbox_t mbox);
  SCM (*catch_handler) (void *data, SCM tag, SCM throw_args);
  int (*next) (void *data, mu_mailbox_t mbox);
  int (*exit) (void *data, mu_mailbox_t mbox);
  void *data;
} mu_guimb_param_t;

#ifdef __cplusplus
extern "C" {
#endif

void mu_scm_error (const char *func_name, int status,
		   const char *fmt, SCM args);

extern void mu_scm_init (void);

extern void mu_scm_mailbox_init (void);
extern SCM mu_scm_mailbox_create (mu_mailbox_t mbox);
extern int mu_scm_is_mailbox (SCM scm);

extern void mu_scm_message_init (void);
extern SCM mu_scm_message_create (SCM owner, mu_message_t msg);
extern int mu_scm_is_message (SCM scm);
extern mu_message_t mu_scm_message_get (SCM MESG);

extern int mu_scm_is_body (SCM scm);
extern void mu_scm_body_init (void);
extern SCM mu_scm_body_create (SCM mesg, mu_body_t body);

extern void mu_scm_address_init (void);
extern void mu_scm_logger_init (void);

extern void mu_scm_port_init (void);
extern SCM mu_port_make_from_stream (SCM msg, mu_stream_t stream, long mode);

extern void mu_scm_mime_init (void);
extern void mu_scm_message_add_owner (SCM MESG, SCM owner);

extern void mu_scm_mutil_init (void);

SCM mu_scm_make_debug_port (mu_debug_t debug, mu_log_level_t level);
void mu_scm_debug_port_init (void);


extern void mu_guile_init (int debug);
extern int mu_guile_load (const char *filename, int argc, char **argv);
extern int mu_guile_eval (const char *string);
extern int mu_guile_mailbox_apply (mu_mailbox_t mbx, char *funcname);
extern int mu_guile_message_apply (mu_message_t msg, char *funcname);

extern int mu_guile_safe_exec (SCM (*handler) (void *data), void *data,
			       SCM *result);
extern int mu_guile_safe_proc_call (SCM proc, SCM arglist, SCM *presult);

#ifdef __cplusplus
}
#endif

#endif
