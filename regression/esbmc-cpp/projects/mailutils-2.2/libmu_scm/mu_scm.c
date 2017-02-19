/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2006, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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

#ifndef PATH_SENDMAIL
# define PATH_SENDMAIL "/usr/lib/sendmail"
#endif

void
mu_scm_error (const char *func_name, int status,
	      const char *fmt, SCM args)
{
  scm_error_scm (scm_from_locale_symbol ("mailutils-error"),
		 func_name ? scm_from_locale_string (func_name) : SCM_BOOL_F,
		 scm_from_locale_string (fmt),
		 args,
		 scm_list_1 (scm_from_int (status)));
}

SCM _mu_scm_package_string; /* STRING: PACKAGE_STRING */
SCM _mu_scm_package;        /* STRING: PACKAGE */
SCM _mu_scm_version;        /* STRING: VERSION */
SCM _mu_scm_mailer;         /* STRING: Default mailer path. */
SCM _mu_scm_debug;          /* NUM: Default debug level. */

struct format_record {
  char *name;
  mu_record_t *record;
};

static struct format_record format_table[] = {
  { "mbox", &mu_mbox_record },
  { "mh",   &mu_mh_record },
  { "maildir", &mu_maildir_record },
  { "pop",  &mu_pop_record },
  { "imap", &mu_imap_record },
  { "sendmail", &mu_sendmail_record },
  { "smtp", &mu_smtp_record },
  { NULL, NULL },
};

static mu_record_t *
find_format (const struct format_record *table, const char *name)
{
  for (; table->name; table++)
    if (strcmp (table->name, name) == 0)
      break;
  return table->record;
}
		
static int
register_format (const char *name)
{
  int status = 0;
  
  if (!name)
    {
      struct format_record *table;
      for (table = format_table; table->name; table++)
	mu_registrar_record (*table->record);
    }
  else
    {
      mu_record_t *record = find_format (format_table, name);
      if (record)
	status = mu_registrar_record (*record);
      else
	status = EINVAL;
    }
  return status;
}
    

SCM_DEFINE_PUBLIC (scm_mu_register_format, "mu-register-format", 0, 0, 1,
		   (SCM rest),
"Registers desired mailutils formats.  Any number of arguments can be given.\n"
"Each argument must be one of the following strings:\n\n"
"@multitable @columnfractions 0.3 0.6\n"
"@headitem Argument @tab Meaning\n"
"@item @samp{mbox}  @tab Regular UNIX mbox format\n"
"@item @samp{mh}    @tab @acronym{MH} mailbox format\n"
"@item @samp{maildir} @tab @i{Maildir} mailbox format\n"
"@item @samp{pop}   @tab @acronym{POP} mailbox format\n"
"@item @samp{imap}  @tab @acronym{IMAP} mailbox format\n"
"@item @samp{sendmail} @tab @i{sendmail} mailer format\n"
"@item @samp{smtp}  @tab @acronym{SMTP} mailer format\n"
"@end multitable\n"
"\n"
"If called without arguments, the function registers all available formats\n")
#define FUNC_NAME s_scm_mu_register_format
{
  int status;

  if (scm_is_null (rest))
    {
      status = register_format (NULL);
      if (status)
	mu_scm_error (FUNC_NAME, status,
		      "Cannot register formats",
		      SCM_BOOL_F);
    }
  else
    {
      for (; !scm_is_null (rest); rest = SCM_CDR (rest))
	{
	  char *s;
	  SCM scm = SCM_CAR (rest);
	  SCM_ASSERT (scm_is_string (scm), scm, SCM_ARGn, FUNC_NAME);
	  s = scm_to_locale_string (scm);
	  status = register_format (s);
	  free (scm);
	  if (status)
	    mu_scm_error (FUNC_NAME, status,
			  "Cannot register format ~A",
			  scm_list_1 (scm));
	}
    }
  return SCM_UNSPECIFIED;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_strerror, "mu-strerror", 1, 0, 0,
		   (SCM err),
"Return the error message corresponding to @var{err}, which must be\n"
"an integer value.\n")
#define FUNC_NAME s_scm_mu_strerror
{
  SCM_ASSERT (scm_is_integer (err), err, SCM_ARG1, FUNC_NAME);
  return scm_from_locale_string (mu_strerror (scm_to_int (err)));
}
#undef FUNC_NAME

static struct
{
  char *name;
  int value;
} attr_kw[] = {
  { "MU-ATTRIBUTE-ANSWERED",  MU_ATTRIBUTE_ANSWERED },
  { "MU-ATTRIBUTE-FLAGGED",   MU_ATTRIBUTE_FLAGGED },
  { "MU-ATTRIBUTE-DELETED",   MU_ATTRIBUTE_DELETED }, 
  { "MU-ATTRIBUTE-DRAFT",     MU_ATTRIBUTE_DRAFT },   
  { "MU-ATTRIBUTE-SEEN",      MU_ATTRIBUTE_SEEN },    
  { "MU-ATTRIBUTE-READ",      MU_ATTRIBUTE_READ },    
  { "MU-ATTRIBUTE-MODIFIED",  MU_ATTRIBUTE_MODIFIED },
  { "MU-ATTRIBUTE-RECENT",    MU_ATTRIBUTE_RECENT },
  { NULL, 0 }
};

/* Initialize the library */
void
mu_scm_init ()
{
  int i;

  _mu_scm_mailer = scm_from_locale_string ("sendmail:" PATH_SENDMAIL);
  scm_c_define ("mu-mailer", _mu_scm_mailer);

  _mu_scm_debug = scm_from_int (0);
  scm_c_define ("mu-debug", _mu_scm_debug);

  _mu_scm_package = scm_from_locale_string (PACKAGE);
  scm_c_define ("mu-package", _mu_scm_package);
  scm_c_export ("mu-package", NULL);
  
  _mu_scm_version = scm_from_locale_string (VERSION);
  scm_c_define ("mu-version", _mu_scm_version);
  scm_c_export ("mu-version", NULL);
  
  _mu_scm_package_string = scm_from_locale_string (PACKAGE_STRING);
  scm_c_define ("mu-package-string", _mu_scm_package_string);
  scm_c_export ("mu-package-string", NULL);
  
  /* Create MU- attribute names */
  for (i = 0; attr_kw[i].name; i++)
    {
      scm_c_define (attr_kw[i].name, scm_from_int(attr_kw[i].value));
      scm_c_export (attr_kw[i].name, NULL);
    }
  
  mu_scm_mutil_init ();
  mu_scm_mailbox_init ();
  mu_scm_message_init ();
  mu_scm_address_init ();
  mu_scm_body_init ();
  mu_scm_logger_init ();
  mu_scm_port_init ();
  mu_scm_mime_init ();
  mu_scm_debug_port_init ();
  
#include "mu_scm.x"

  mu_registrar_record (MU_DEFAULT_RECORD);
  mu_registrar_set_default_record (MU_DEFAULT_RECORD);
}
