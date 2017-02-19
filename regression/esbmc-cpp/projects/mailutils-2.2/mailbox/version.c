/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2009, 2010 Free Software Foundation, Inc.

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

#include <mailutils/nls.h>
#include <mailutils/version.h>
#include <mailutils/cstr.h>
#include <stdio.h>
#include <string.h>

#include <confpaths.h>

char *mu_license_text =
 N_("   GNU Mailutils is free software; you can redistribute it and/or modify\n"
    "   it under the terms of the GNU General Public License as published by\n"
    "   the Free Software Foundation; either version 3 of the License, or\n"
    "   (at your option) any later version.\n"
    "\n"
    "   GNU Mailutils is distributed in the hope that it will be useful,\n"
    "   but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
    "   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
    "   GNU General Public License for more details.\n"
    "\n"
    "   You should have received a copy of the GNU General Public License along\n"
    "   with GNU Mailutils; if not, write to the Free Software Foundation,\n"
    "   Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA\n"
    "\n"
    "\n"
);

static struct mu_conf_option mu_conf_option[] = {
  { "VERSION=" VERSION, N_("Version of this package") },
  { "SYSCONFDIR=" SYSCONFDIR, N_("System configuration directory") },
  { "MAILSPOOLDIR=" MU_PATH_MAILDIR, N_("Default mail spool directory") },
  { "SCHEME=" MU_DEFAULT_SCHEME, N_("Default mailbox type") },
  { "LOG_FACILITY=" LOG_FACILITY_STRING, N_("Default syslog facility") },
#ifdef USE_LIBPAM
  { "USE_LIBPAM", N_("PAM support") },
#endif
#ifdef HAVE_LIBLTDL
  { "HAVE_LIBLTDL", N_("a portable `dlopen' wrapper library") },
#endif
#ifdef WITH_BDB2
  { "WITH_BDB2", N_("Berkeley DB v. 2") },
#endif
#ifdef WITH_NDBM
  { "WITH_NDBM", },
#endif
#ifdef WITH_OLD_DBM
  { "WITH_OLD_DBM", N_("Old DBM support") },
#endif
#ifdef WITH_GDBM
  { "WITH_GDBM", N_("GNU DBM") },
#endif
#ifdef WITH_TOKYOCABINET
  { "WITH_TOKYOCABINET", N_("Tokyo Cabinet DBM") },
#endif
#ifdef WITH_GNUTLS
  { "WITH_GNUTLS", N_("TLS support using GNU TLS") },
#endif
#ifdef WITH_GSASL
  { "WITH_GSASL", N_("SASL support using GNU SASL") },
#endif
#ifdef WITH_GSSAPI
  { "WITH_GSSAPI", N_("GSSAPI support") },
#endif
#ifdef WITH_GUILE
  { "WITH_GUILE", N_("Support for Guile as extension language") },
#endif
#ifdef WITH_PYTHON
  { "WITH_PYTHON", N_("Support for Python as extension language") },
#endif
#ifdef WITH_PTHREAD
  { "WITH_PTHREAD", N_("Support for POSIX threads") },
#endif
#ifdef WITH_READLINE
  { "WITH_READLINE", N_("GNU Readline") },
#endif
#ifdef HAVE_MYSQL
  { "HAVE_MYSQL", N_("MySQL") },
#endif
#ifdef HAVE_PGSQL
  { "HAVE_PGSQL", N_("PostgreSQL") },
#endif
#ifdef WITH_LDAP
  { "WITH_LDAP", },
#endif
#ifdef WITH_LIBWRAP
  { "WITH_LIBWRAP", N_("Support for TCP wrappers") },
#endif
#ifdef ENABLE_VIRTUAL_DOMAINS
  { "ENABLE_VIRTUAL_DOMAINS", N_("Support for virtual mail domains") },
#endif
#ifdef ENABLE_IMAP
  { "ENABLE_IMAP", N_("IMAP4 protocol support") },
#endif
#ifdef ENABLE_POP
  { "ENABLE_POP",  N_("POP3 protocol support") },
#endif
#ifdef ENABLE_MH
  { "ENABLE_MH", N_("MH mail storage support") },
#endif
#ifdef ENABLE_MAILDIR
  { "ENABLE_MAILDIR", N_("Maildir mail storage support") },
#endif
#ifdef ENABLE_SMTP
  { "ENABLE_SMTP", N_("SMTP protocol support") },
#endif
#ifdef ENABLE_SENDMAIL
  { "ENABLE_SENDMAIL", N_("Sendmail command line interface support")},
#endif
#ifdef ENABLE_NNTP
  { "ENABLE_NNTP", N_("NNTP protocol support") },
#endif
#ifdef ENABLE_RADIUS
  { "ENABLE_RADIUS", N_("RADIUS protocol support") },
#endif
#ifdef WITH_INCLUDED_LIBINTL
  { "WITH_INCLUDED_LIBINTL", N_("GNU libintl compiled in") },
#endif
  { NULL }
};

void
mu_fprint_conf_option (FILE *fp, const struct mu_conf_option *opt, int verbose)
{
  fprintf (fp, "%s", opt->name);
  if (verbose && opt->descr)
    fprintf (fp, " \t- %s", _(opt->descr));
  fputc('\n', fp);
}

void
mu_fprint_options (FILE *fp, int verbose)
{
  int i;
  
  for (i = 0; mu_conf_option[i].name; i++)
    mu_fprint_conf_option (fp, mu_conf_option + i, verbose);
}

void
mu_print_options ()
{
  mu_fprint_options (stdout, 1);
}

const struct mu_conf_option *
mu_check_option (char *name)
{
  int i;
  
  for (i = 0; mu_conf_option[i].name; i++)
    {
      int len;
      char *q, *p = strchr (mu_conf_option[i].name, '=');
      if (p)
	len = p - mu_conf_option[i].name;
      else
	len = strlen (mu_conf_option[i].name);

      if (mu_c_strncasecmp (mu_conf_option[i].name, name, len) == 0)
	return &mu_conf_option[i];
      else if ((q = strchr (mu_conf_option[i].name, '_')) != NULL
	       && mu_c_strncasecmp (q + 1, name,
			       len - (q - mu_conf_option[i].name) - 1) == 0)
	return &mu_conf_option[i];
    }
  return NULL;
}  

