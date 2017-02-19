/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2008, 2009, 2010
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif  
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sieve-priv.h>

#define SIEVE_ARG(m,n,t) ((m)->prog[(m)->pc+(n)].t)
#define SIEVE_ADJUST(m,n) (m)->pc+=(n)

#define INSTR_DEBUG(m) \
 (((m)->debug_level & (MU_SIEVE_DEBUG_INSTR|MU_SIEVE_DEBUG_DISAS)) \
  && (m)->debug_printer)
#define INSTR_DISASS(m) ((m)->debug_level & MU_SIEVE_DEBUG_DISAS)

void
_mu_sv_instr_nop (mu_sieve_machine_t mach)
{
  if (INSTR_DEBUG (mach))
    mu_sieve_debug (mach, "%4lu: NOP\n",
		 (unsigned long) (mach->pc - 1));
}

void
_mu_sv_instr_source (mu_sieve_machine_t mach)
{
  mach->locus.source_file = (char*) SIEVE_ARG (mach, 0, string);
  if (INSTR_DEBUG (mach))
    mu_sieve_debug (mach, "%4lu: SOURCE %s\n",
		    (unsigned long) (mach->pc - 1),
		    mach->locus.source_file);
  SIEVE_ADJUST (mach, 1);
}
		 
void
_mu_sv_instr_line (mu_sieve_machine_t mach)
{
  mach->locus.source_line = SIEVE_ARG (mach, 0, line);
  if (INSTR_DEBUG (mach))
    mu_sieve_debug (mach, "%4lu: LINE %lu\n",
		 (unsigned long) (mach->pc - 1),
		 (unsigned long) mach->locus.source_line);
  SIEVE_ADJUST (mach, 1);
}
		 
static int
instr_run (mu_sieve_machine_t mach)
{
  mu_sieve_handler_t han = SIEVE_ARG (mach, 0, handler);
  mu_list_t arg_list = SIEVE_ARG (mach, 1, list);
  mu_list_t tag_list = SIEVE_ARG (mach, 2, list);
  int rc = 0;
  
  SIEVE_ADJUST(mach, 4);

  if (INSTR_DEBUG (mach))
    {
      mu_sieve_debug (mach, "Arguments: ");
      mu_sv_print_value_list (arg_list, mach->debug_printer, mach->data);
      mu_sieve_debug (mach, "\nTags:");
      mu_sv_print_tag_list (tag_list, mach->debug_printer, mach->data);
      mu_sieve_debug (mach, "\n");
    }

  if (!INSTR_DISASS(mach))
    rc = han (mach, arg_list, tag_list);
  return rc;
}

void
_mu_sv_instr_action (mu_sieve_machine_t mach)
{
  mach->identifier = SIEVE_ARG (mach, 3, string);
  if (INSTR_DEBUG (mach))
    mu_sieve_debug (mach, "%4lu: ACTION: %s\n",
		 (unsigned long) (mach->pc - 1),
		 mach->identifier);
  mach->action_count++;
  instr_run (mach);
  mach->identifier = NULL;
}

void
_mu_sv_instr_test (mu_sieve_machine_t mach)
{
  mach->identifier = SIEVE_ARG (mach, 3, string);
  if (INSTR_DEBUG (mach))
    mu_sieve_debug (mach, "%4lu: TEST: %s\n",
		 (unsigned long) (mach->pc - 1),
		 mach->identifier);
  mach->reg = instr_run (mach);
  mach->identifier = NULL;
}

void
_mu_sv_instr_push (mu_sieve_machine_t mach)
{
  if (INSTR_DEBUG (mach))
    {
      mu_sieve_debug (mach, "%4lu: PUSH\n", (unsigned long)(mach->pc - 1));
      if (INSTR_DISASS (mach))
	return;
    }
  
  if (!mach->stack && mu_list_create (&mach->stack))
    {
      mu_sieve_error (mach, _("cannot create stack"));
      mu_sieve_abort (mach);
    }
  mu_list_prepend (mach->stack, (void*) mach->reg);
}

void
_mu_sv_instr_pop (mu_sieve_machine_t mach)
{
  if (INSTR_DEBUG (mach))
    {
      mu_sieve_debug (mach, "%4lu: POP\n", (unsigned long)(mach->pc - 1));
      if (INSTR_DISASS (mach))
	return;
    }

  if (!mach->stack || mu_list_is_empty (mach->stack))
    {
      mu_sieve_error (mach, _("stack underflow"));
      mu_sieve_abort (mach);
    }
  mu_list_get (mach->stack, 0, (void **)&mach->reg);
  mu_list_remove (mach->stack, (void *)mach->reg);
}

void
_mu_sv_instr_not (mu_sieve_machine_t mach)
{
  if (INSTR_DEBUG (mach))
    {
      mu_sieve_debug (mach, "%4lu: NOT\n", (unsigned long)(mach->pc - 1));
      if (INSTR_DISASS (mach))
	return;
    }
  mach->reg = !mach->reg;
}

void
_mu_sv_instr_branch (mu_sieve_machine_t mach)
{
  long num = SIEVE_ARG (mach, 0, number);

  SIEVE_ADJUST (mach, 1);
  if (INSTR_DEBUG (mach))
    {
      mu_sieve_debug (mach, "%4lu: BRANCH %lu\n",
		   (unsigned long)(mach->pc-2),
		   (unsigned long)(mach->pc + num));
      if (INSTR_DISASS (mach))
	return;
    }

  mach->pc += num;
}

void
_mu_sv_instr_brz (mu_sieve_machine_t mach)
{
  long num = SIEVE_ARG (mach, 0, number);
  SIEVE_ADJUST (mach, 1);

  if (INSTR_DEBUG (mach))
    {
      mu_sieve_debug (mach, "%4lu: BRZ %lu\n",
		   (unsigned long)(mach->pc-2),
		   (unsigned long)(mach->pc + num));
      if (INSTR_DISASS (mach))
	return;
    }
  
  if (!mach->reg)
    mach->pc += num;
}
  
void
_mu_sv_instr_brnz (mu_sieve_machine_t mach)
{
  long num = SIEVE_ARG (mach, 0, number);
  SIEVE_ADJUST (mach, 1);

  if (INSTR_DEBUG (mach))
    {
      mu_sieve_debug (mach, "%4lu: BRNZ %lu\n",
		   (unsigned long)(mach->pc-2),
		   (unsigned long)(mach->pc + num));
      if (INSTR_DISASS (mach))
	return;
    }
  
  if (mach->reg)
    mach->pc += num;
}
  
void
mu_sieve_abort (mu_sieve_machine_t mach)
{
  longjmp (mach->errbuf, 1);
}

void *
mu_sieve_get_data (mu_sieve_machine_t mach)
{
  return mach->data;
}

int
mu_sieve_get_locus (mu_sieve_machine_t mach, mu_sieve_locus_t *loc)
{
  if (mach->source_list)
    {
      *loc = mach->locus;
      return 0;
    }
  return 1;
}

mu_message_t
mu_sieve_get_message (mu_sieve_machine_t mach)
{
  if (!mach->msg)
    mu_mailbox_get_message (mach->mailbox, mach->msgno, &mach->msg);
  return mach->msg;
}

size_t
mu_sieve_get_message_num (mu_sieve_machine_t mach)
{
  return mach->msgno;
}

int
mu_sieve_get_debug_level (mu_sieve_machine_t mach)
{
  return mach->debug_level;
}

const char *
mu_sieve_get_identifier (mu_sieve_machine_t mach)
{
  return mach->identifier;
}

int
mu_sieve_is_dry_run (mu_sieve_machine_t mach)
{
  return mach->debug_level & MU_SIEVE_DRY_RUN;
}

int
sieve_run (mu_sieve_machine_t mach)
{
  if (setjmp (mach->errbuf))
    return 1;

  mach->action_count = 0;
  
  for (mach->pc = 1; mach->prog[mach->pc].handler; )
    (*mach->prog[mach->pc++].instr) (mach);

  if (mach->action_count == 0)
    mu_sieve_log_action (mach, "IMPLICIT KEEP", NULL);
  
  if (INSTR_DEBUG (mach))
    mu_sieve_debug (mach, "%4lu: STOP\n", (unsigned long) mach->pc);
  
  return 0;
}

int
mu_sieve_disass (mu_sieve_machine_t mach)
{
  int level = mach->debug_level;
  int rc;

  mach->debug_level = MU_SIEVE_DEBUG_INSTR | MU_SIEVE_DEBUG_DISAS;
  rc = sieve_run (mach);
  mach->debug_level = level;
  return rc;
}
  
static int
_sieve_action (mu_observer_t obs, size_t type, void *data, void *action_data)
{
  mu_sieve_machine_t mach;
  
  if (type != MU_EVT_MESSAGE_ADD)
    return 0;

  mach = mu_observer_get_owner (obs);
  mach->msgno++;
  mu_mailbox_get_message (mach->mailbox, mach->msgno, &mach->msg);
  sieve_run (mach);
  return 0;
}

int
mu_sieve_mailbox (mu_sieve_machine_t mach, mu_mailbox_t mbox)
{
  int rc;
  size_t total;
  mu_observer_t observer;
  mu_observable_t observable;
  
  if (!mach || !mbox)
    return EINVAL;

  mu_observer_create (&observer, mach);
  mu_observer_set_action (observer, _sieve_action, mach);
  mu_mailbox_get_observable (mbox, &observable);
  mu_observable_attach (observable, MU_EVT_MESSAGE_ADD, observer);
  
  mach->mailbox = mbox;
  mach->msgno = 0;
  rc = mu_mailbox_scan (mbox, 1, &total);
  if (rc)
    mu_sieve_error (mach, _("mu_mailbox_scan: %s"), mu_strerror (errno));

  mu_observable_detach (observable, observer);
  mu_observer_destroy (&observer, mach);

  mach->mailbox = NULL;
  
  return rc;
}

int
mu_sieve_message (mu_sieve_machine_t mach, mu_message_t msg)
{
  int rc;
  
  if (!mach || !msg)
    return EINVAL;

  mach->msgno = 1;
  mach->msg = msg;
  mach->mailbox = NULL;
  rc = sieve_run (mach);
  mach->msg = NULL;
  
  return rc;
}
