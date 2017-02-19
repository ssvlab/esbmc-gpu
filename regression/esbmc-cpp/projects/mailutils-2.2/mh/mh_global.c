/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2007, 2010 Free Software Foundation,
   Inc.

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

/* Global MH state. */

#include <mh.h>

static const char *current_folder = NULL;
size_t current_message = 0;
mh_context_t *context;
mh_context_t *profile;
mh_context_t *sequences;
int rcpt_mask = RCPT_DEFAULT;
int mh_auto_install = 1;

/* Global profile */

const char *
mh_global_profile_get (const char *name, const char *defval)
{
  return mh_context_get_value (profile, name, defval);
}

int
mh_global_profile_set (const char *name, const char *value)
{
  return mh_context_set_value (profile, name, value);
}

int
mh_global_profile_iterate (mh_context_iterator fp, void *data)
{
  return mh_context_iterate (profile, fp, data);
}

void
mh_read_profile ()
{
  char *p;
  const char *fallback;
  
  p = getenv ("MH");
  if (p)
    p = mu_tilde_expansion (p, "/", NULL);
  else
    {
      char *home = mu_get_homedir ();
      if (!home)
	abort (); /* shouldn't happen */
      asprintf (&p, "%s/%s", home, MH_USER_PROFILE);
      free (home);
    }

  if (mh_auto_install && access (p, R_OK))
    mh_install (p, 1);

  profile = mh_context_create (p, 1);
  mh_context_read (profile);

  mu_set_folder_directory (mh_get_dir ());

  mh_set_reply_regex (mh_global_profile_get ("Reply-Regex", NULL));
  fallback = mh_global_profile_get ("Decode-Fallback", NULL);
  if (fallback && mu_set_default_fallback (fallback))
    mu_error (_("Incorrect value for decode-fallback"));
}

/* Global context */

void
_mh_init_global_context ()
{
  char *p, *ctx_name;
  
  if (context)
    return;
  p = getenv ("CONTEXT");
  if (!p)
    p = MH_CONTEXT_FILE;
  ctx_name = mh_expand_name (NULL, p, 0);
  context = mh_context_create (ctx_name, 1);
  mh_context_read (context);
  
  if (!current_folder)
    current_folder = mh_context_get_value (context, "Current-Folder",
					   mh_global_profile_get ("Inbox",
							          "inbox"));
}

const char *
mh_global_context_get (const char *name, const char *defval)
{
  _mh_init_global_context ();
  return mh_context_get_value (context, name, defval);
}

int
mh_global_context_set (const char *name, const char *value)
{
  _mh_init_global_context ();
  return mh_context_set_value (context, name, value);
}

int
mh_global_context_iterate (mh_context_iterator fp, void *data)
{
  _mh_init_global_context ();
  return mh_context_iterate (context, fp, data);
}

const char *
mh_current_folder ()
{
  _mh_init_global_context ();
  return mh_global_context_get ("Current-Folder",
				mh_global_profile_get ("Inbox", "inbox"));
}

const char *
mh_set_current_folder (const char *val)
{
  mh_global_context_set ("Current-Folder", val);
  current_folder = mh_current_folder ();
  return current_folder;
}

/* Global sequences */

void
_mh_init_global_sequences ()
{
  const char *name;
  char *p, *seq_name;

  if (sequences)
    return;
  
  _mh_init_global_context ();
  name = mh_global_profile_get ("mh-sequences", MH_SEQUENCES_FILE);
  p = mh_expand_name (NULL, current_folder, 0);
  asprintf (&seq_name, "%s/%s", p, name);
  free (p);
  sequences = mh_context_create (seq_name, 1);
  if (mh_context_read (sequences) == 0)
    current_message = strtoul (mh_context_get_value (sequences, "cur", "0"),
			       NULL, 10);
}

void
mh_global_sequences_drop ()
{
  sequences = NULL;
}

const char *
mh_global_sequences_get (const char *name, const char *defval)
{
  _mh_init_global_sequences ();
  return mh_context_get_value (sequences, name, defval);
}

int
mh_global_sequences_set (const char *name, const char *value)
{
  _mh_init_global_sequences ();
  return mh_context_set_value (sequences, name, value);
}

int
mh_global_sequences_iterate (mh_context_iterator fp, void *data)
{
  _mh_init_global_context ();
  return mh_context_iterate (sequences, fp, data);
}

/* Global state */

void
mh_global_save_state ()
{
  mh_context_set_value (sequences, "cur", mu_umaxtostr (0, current_message));
  mh_context_write (sequences);

  mh_context_set_value (context, "Current-Folder", current_folder);
  mh_context_write (context);
}
