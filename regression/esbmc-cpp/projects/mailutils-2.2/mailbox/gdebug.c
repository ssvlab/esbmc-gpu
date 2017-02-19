/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2007, 2008, 2009, 2010
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

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mailutils/cctype.h>
#include <mailutils/assoc.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/argcv.h>
#include <mailutils/debug.h>
#include <mailutils/cfg.h>
#include <mailutils/nls.h>

int mu_debug_line_info = 0;

struct debug_level
{
  unsigned level;
};

static mu_assoc_t debug_table;

mu_log_level_t
mu_global_debug_level (const char *object_name)
{
  struct debug_level *p = mu_assoc_ref (debug_table, object_name);
  if (p)
    return p->level;
  return 0;
}

int
mu_global_debug_set_level (const char *object_name, mu_log_level_t level)
{
  int rc;
  struct debug_level *dbg;
  
  if (!debug_table)
    {
      rc = mu_assoc_create (&debug_table, sizeof(struct debug_level), 0);
      if (rc)
	return rc;
    }

  rc = mu_assoc_ref_install (debug_table, object_name, (void**) &dbg);
  if (rc == 0 || rc == MU_ERR_EXISTS)
    dbg->level = level;
  return rc;
}

int
mu_global_debug_clear_level (const char *object_name)
{
  int rc = 0;
  
  if (!object_name)
    mu_assoc_clear (debug_table);
  else
    rc = mu_assoc_remove (debug_table, object_name);
  return rc;
}

int
decode_debug_level (const char *p, int *lev)
{
  if (strcmp (p, "error") == 0)
    *lev = MU_DEBUG_ERROR;
  else if (strncmp (p, "trace", 5) == 0 && mu_isdigit (p[5]) && p[6] == 0)
    *lev = MU_DEBUG_TRACE0 + atoi (p + 5);
  else if (strcmp (p, "proto") == 0)
    *lev = MU_DEBUG_PROT;
  else
    return 1;
  return 0;
}

int
mu_debug_level_from_string (const char *string, mu_log_level_t *plev,
			    mu_debug_t debug)
{
  char *q;
  unsigned level = MU_DEBUG_INHERIT;
  
  if (mu_isdigit (*string))
    {
      level = strtoul (string, &q, 0);
      if (*q)
	{
	  mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			       _("invalid debugging specification `%s': "
				 "expected levels or number after `=', "
				 "but found `%s'"),
			       string, string);
	  return MU_ERR_FAILURE;
	}
    }
  else
    {
      char *p = strdup (string);
      size_t len = strlen (p);
      if (len > 0 && p[len-1] == '\n')
	p[len-1] = 0;
      for (q = strtok (p, ","); q; q = strtok (NULL, ","))
	{
	  int flag;
	  int revert = 0;
	  int upto = 0;
	  
	  if (*q == '!')
	    {
	      q++;
	      revert = 1;
	    }
	  if (*q == '<')
	    {
	      q++;
	      upto = 1;
	    }
	  
	  if (decode_debug_level (q, &flag))
	    mu_cfg_format_error (debug, MU_DEBUG_ERROR,
				 _("invalid debugging level `%s'"),
				 q);
	  else if (revert)
	    {
	      if (upto)
		level &= ~MU_DEBUG_LEVEL_UPTO (flag);
	      else
		level &= ~MU_DEBUG_LEVEL_MASK (flag);
	    }
	  else
	    {
	      if (upto)
		level |= MU_DEBUG_LEVEL_UPTO (flag);
	      else
		level |= MU_DEBUG_LEVEL_MASK (flag);
	    }
	}
      free (p);
    }
  *plev = level;
  return 0;
}

int
mu_global_debug_from_string (const char *string, const char *errpfx)
{
  int rc;
  int argc;
  char **argv;
  int i;
  
  rc = mu_argcv_get (string, ";", NULL, &argc, &argv);
  if (rc)
    return rc;

  for (i = 0; i < argc; i++)
    {
      char *p;
      mu_log_level_t level = MU_DEBUG_INHERIT;
      char *object_name = argv[i];
      
      for (p = object_name; *p && *p != '='; p++)
	;

      if (*p == '=')
	{
	  /* FIXME: Use mu_debug_level_from_string */
	  char *q;
	  
	  *p++ = 0;
	  if (mu_isdigit (*p))
	    {
	      level = strtoul (p, &q, 0);
	      if (*q)
		{
		  mu_error ("%s: invalid debugging specification `%s': "
			    "expected levels or number after `=', "
			    "but found `%s'",
			    errpfx, argv[i], p);
		  break;
		}
	    }
	  else
	    {
	      char *q;
	      for (q = strtok (p, ","); q; q = strtok (NULL, ","))
		{
		  int flag;
		  int revert = 0;
		  int upto = 0;
		  
		  if (*q == '!')
		    {
		      q++;
		      revert = 1;
		    }
		  if (*q == '<')
		    {
		      q++;
		      upto = 1;
		    }
		  
		  if (decode_debug_level (q, &flag))
		    mu_error ("%s: invalid debugging level `%s'",
			      errpfx, q);
		  else if (revert)
		    {
		      if (upto)
			level &= ~MU_DEBUG_LEVEL_UPTO (flag);
		      else
			level &= ~MU_DEBUG_LEVEL_MASK (flag);
		    }
		  else
		    {
		      if (upto)
			level |= MU_DEBUG_LEVEL_UPTO (flag);
		      else
			level |= MU_DEBUG_LEVEL_MASK (flag);
		    }
		}
	    }   
	}	  
      else
	level |= MU_DEBUG_LEVEL_UPTO (MU_DEBUG_PROT);
      
      if (p[-1] == ':')
	{
	  p[-1] = 0;
	  level &= ~MU_DEBUG_INHERIT;
	}
      
      mu_global_debug_set_level (object_name, level);
    }
  
  mu_argcv_free (argc, argv);
  return 0;
}


