/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

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
#include "cmdline.h"
#include <unistd.h>
#include <stdlib.h>
#include <mailutils/stream.h>
#include "xalloc.h"
#include <string.h>

struct mu_cfg_tree *mu_argp_tree;

void
mu_argp_init (const char *vers, const char *bugaddr)
{
  argp_program_version = vers ? vers : PACKAGE_STRING;
  argp_program_bug_address = bugaddr ? bugaddr : "<" PACKAGE_BUGREPORT ">";
}

static char *
get_canonical_name ()
{
  char *name;
  size_t len;
  char *p = strchr (argp_program_version, ' ');
  if (!p)
    return strdup (mu_program_name);
  len = p - argp_program_version;
  name = malloc (len + 1);
  if (!name)
    abort ();
  memcpy (name, argp_program_version, len);
  name[len] = 0;
  return name;
}

int mu_help_config_mode;
int mu_rcfile_lint;

int (*mu_app_cfg_verifier) (void) = NULL;

int
mu_app_init (struct argp *myargp, const char **capa,
	     struct mu_cfg_param *cfg_param,
	     int argc, char **argv, int flags, int *pindex, void *data)
{
  int rc, i;
  struct argp *argp;
  struct argp argpnull = { 0 };
  char **excapa;
  struct mu_cfg_tree *parse_tree = NULL;
  
  mu_set_program_name (argv[0]);
  mu_libargp_init ();
  if (capa)
    for (i = 0; capa[i]; i++)
      mu_gocs_register_std (capa[i]); /*FIXME*/
  if (!myargp)
    myargp = &argpnull;
  argp = mu_argp_build (myargp, &excapa);

  mu_cfg_tree_create (&mu_argp_tree);
  rc = argp_parse (argp, argc, argv, flags, pindex, data);
  mu_argp_done (argp);
  if (rc)
    return rc;

  /* Reset program name, it may have been changed using the `--program-name'
     option. */
  mu_set_program_name (program_invocation_name);
  
  mu_libcfg_init (excapa);
  free (excapa);

  if (mu_help_config_mode)
    {
      char *comment;
      char *canonical_name = get_canonical_name ();
      mu_stream_t stream;
      mu_stdio_stream_create (&stream, stdout,
			      MU_STREAM_NO_CHECK|MU_STREAM_NO_CLOSE);
      mu_stream_open (stream);
      asprintf (&comment,
		"Configuration file structure for %s utility.",
		mu_program_name);
      mu_cfg_format_docstring (stream, comment, 0);
      free (comment);
      asprintf (&comment,
		"For use in global configuration file (%s), enclose it "
		"in `program %s { ... };",
		MU_CONFIG_FILE,
		mu_program_name);		   
      mu_cfg_format_docstring (stream, comment, 0);
      free (comment);
      asprintf (&comment, "For more information, use `info %s'.",
		canonical_name);
      mu_cfg_format_docstring (stream, comment, 0);
      free (comment);
      
      mu_format_config_tree (stream, mu_program_name, cfg_param, 0);
      mu_stream_destroy (&stream, NULL);
      exit (0);
    }

  rc = mu_libcfg_parse_config (&parse_tree);
  if (rc == 0)
    {
      int cfgflags = MU_PARSE_CONFIG_PLAIN;

      if (mu_cfg_parser_verbose)
	cfgflags |= MU_PARSE_CONFIG_VERBOSE;
      if (mu_cfg_parser_verbose > 1)
	cfgflags |= MU_PARSE_CONFIG_DUMP;
      mu_cfg_tree_postprocess (mu_argp_tree, cfgflags);
      mu_cfg_tree_union (&parse_tree, &mu_argp_tree);
      rc = mu_cfg_tree_reduce (parse_tree, mu_program_name, cfg_param,
			       cfgflags, data);
    }
  
  if (mu_rcfile_lint)
    {
      if (rc || mu_cfg_error_count)
	exit (1);
      if (mu_app_cfg_verifier)
	rc = mu_app_cfg_verifier ();
      exit (rc ? 1 : 0);
    }
  
  mu_gocs_flush ();
  mu_cfg_destroy_tree (&mu_argp_tree);

  return !!(rc || mu_cfg_error_count);
}

