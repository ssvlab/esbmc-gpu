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

struct _mu_debug_port
{
  mu_debug_t      debug;
  mu_log_level_t  level;
};

static long     scm_tc16_mu_debug_port;

SCM
mu_scm_make_debug_port (mu_debug_t debug, mu_log_level_t level)
{
  struct _mu_debug_port *dp;
  SCM             port;
  scm_port       *pt;

  dp = scm_gc_malloc (sizeof (struct _mu_debug_port), "mu-debug-port");
  dp->debug = debug;
  dp->level = level;
  port = scm_cell (scm_tc16_mu_debug_port, 0);
  pt = scm_add_to_port_table (port);
  SCM_SETPTAB_ENTRY (port, pt);
  pt->rw_random = 0;
  SCM_SET_CELL_TYPE (port,
		     (scm_tc16_mu_debug_port | SCM_OPN | SCM_WRTNG |
		      SCM_BUF0));
  SCM_SETSTREAM (port, dp);
  return port;
}

#define MU_DEBUG_PORT(x) ((struct _mu_debug_port *) SCM_STREAM (x))

static SCM
_mu_debug_port_mark (SCM port)
{
  return SCM_BOOL_F;
}

static void
_mu_debug_port_flush (SCM port)
{
  /* struct _mu_debug_port *dp = MU_DEBUG_PORT (port); */

  /* FIXME: */
}

static int
_mu_debug_port_close (SCM port)
{
  struct _mu_debug_port *dp = MU_DEBUG_PORT (port);

  if (dp)
    {
      _mu_debug_port_flush (port);
      SCM_SETSTREAM (port, NULL);
      scm_gc_free (dp, sizeof (struct _mu_debug_port), "mu-debug-port");
    }
  return 0;
}

static scm_sizet
_mu_debug_port_free (SCM port)
{
  _mu_debug_port_close (port);
  return 0;
}

static int
_mu_debug_port_fill_input (SCM port)
{
  return EOF;
}

static void
_mu_debug_port_write (SCM port, const void *data, size_t size)
{
  struct _mu_debug_port *dp = MU_DEBUG_PORT (port);

  mu_debug_printf (dp->debug, dp->level, "%.*s", size, (const char *)data);
}

static int
_mu_debug_port_print (SCM exp, SCM port, scm_print_state * pstate)
{
  scm_puts ("#<Mailutis debug port>", port);
  return 1;
}

void
mu_scm_debug_port_init ()
{
  scm_tc16_mu_debug_port = scm_make_port_type ("mu-debug-port",
					       _mu_debug_port_fill_input,
					       _mu_debug_port_write);
  scm_set_port_mark (scm_tc16_mu_debug_port, _mu_debug_port_mark);
  scm_set_port_free (scm_tc16_mu_debug_port, _mu_debug_port_free);
  scm_set_port_print (scm_tc16_mu_debug_port, _mu_debug_port_print);
  scm_set_port_flush (scm_tc16_mu_debug_port, _mu_debug_port_flush);
  scm_set_port_close (scm_tc16_mu_debug_port, _mu_debug_port_close);
}
