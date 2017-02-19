/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

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

#include "mu_scm.h"
#include <setjmp.h>

static SCM
eval_catch_body (void *list)
{
  return scm_primitive_eval ((SCM)list);
}

static SCM
eval_catch_handler (void *data, SCM tag, SCM throw_args)
{
  scm_handle_by_message_noexit ("mailutils", tag, throw_args);
  longjmp (*(jmp_buf*)data, 1);
}

struct scheme_exec_data
{
  SCM (*handler) (void *data);
  void *data;
  SCM result;
};

static SCM
scheme_safe_exec_body (void *data)
{
  struct scheme_exec_data *ed = data;
  ed->result = ed->handler (ed->data);
  return SCM_BOOL_F;
}

int
mu_guile_safe_exec (SCM (*handler) (void *data), void *data, SCM *result)
{
  jmp_buf jmp_env;
  struct scheme_exec_data ed;

  if (setjmp (jmp_env))
    return 1;
  ed.handler = handler;
  ed.data = data;
  scm_c_catch (SCM_BOOL_T,
	       scheme_safe_exec_body, (void*)&ed,
	       eval_catch_handler, &jmp_env,
	       NULL, NULL);
  if (result)
    *result = ed.result;
  return 0;
}



SCM
lookup_handler (void *data)
{
  const char *symbol = (const char *)data;
  return MU_SCM_SYMBOL_VALUE (symbol);
}

int
mu_guile_sym_lookup (const char *symbol, SCM *result)
{
  return mu_guile_safe_exec (lookup_handler, (void*) symbol, result);
}


int
mu_guile_safe_proc_call (SCM proc, SCM arglist, SCM *presult)
{
  jmp_buf jmp_env;
  SCM cell, result;

  if (setjmp (jmp_env))
    return 1;

  cell = scm_cons (proc, arglist);
  result = scm_c_catch (SCM_BOOL_T,
			eval_catch_body, cell,
			eval_catch_handler, &jmp_env,
			NULL, NULL);
  if (presult)
    *presult = result;
  return 0;
}



void
mu_guile_init (int debug)
{
  scm_init_guile ();
  scm_load_goops ();

  if (debug)
    {
#ifdef GUILE_DEBUG_MACROS
      SCM_DEVAL_P = 1;
      SCM_BACKTRACE_P = 1;
      SCM_RECORD_POSITIONS_P = 1;
      SCM_RESET_DEBUG_MODE;
#endif
    }
  mu_scm_init ();
}


struct load_closure
{
  const char *filename;
  int argc;
  char **argv;
};

static SCM
load_path_handler (void *data)
{
  struct load_closure *lp = data;

  scm_set_program_arguments (lp->argc, lp->argv, (char*)lp->filename);
  scm_primitive_load (scm_from_locale_string (lp->filename));
  return SCM_UNDEFINED;
}

int
mu_guile_load (const char *filename, int argc, char **argv)
{
  struct load_closure lc;
  lc.filename = filename;
  lc.argc = argc;
  lc.argv = argv;
  return mu_guile_safe_exec (load_path_handler, &lc, NULL);
}

static SCM
eval_handler (void *data)
{
  const char *string = data;
  scm_c_eval_string (string);
  return SCM_UNDEFINED;
}

int
mu_guile_eval (const char *string)
{
  return mu_guile_safe_exec (eval_handler, (void*) string, NULL);
}


/* See comment on this function in mu_mailbox.c */
extern SCM mu_scm_mailbox_create0 (mu_mailbox_t mbox, int noclose);

int
mu_guile_mailbox_apply (mu_mailbox_t mbx, char *funcname)
{
  SCM proc;

  if (mu_guile_sym_lookup (funcname, &proc))
    return MU_ERR_NOENT;
  if (scm_procedure_p (proc) != SCM_BOOL_T)
    return EINVAL;

  if (mu_guile_safe_proc_call (proc,
			       scm_list_1 (mu_scm_mailbox_create0 (mbx, 1)),
			       NULL))
    return MU_ERR_FAILURE;

  return 0;
}

int
mu_guile_message_apply (mu_message_t msg, char *funcname)
{
  SCM proc;

  if (mu_guile_sym_lookup (funcname, &proc))
    return MU_ERR_NOENT;
  if (scm_procedure_p (proc) != SCM_BOOL_T)
    return EINVAL;

  if (mu_guile_safe_proc_call (proc,
		       scm_list_1 (mu_scm_message_create (SCM_BOOL_F, msg)),
			   NULL))
    return MU_ERR_FAILURE;

  return 0;
}
