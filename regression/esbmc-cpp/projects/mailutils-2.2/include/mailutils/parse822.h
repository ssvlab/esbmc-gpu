/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2009, 2010 Free Software
   Foundation, Inc.

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

/**
* Parses syntatic elements defined in RFC 822.
*/

#ifndef _MAILUTILS_PARSE822_H
#define _MAILUTILS_PARSE822_H

#include <mailutils/types.h>
#include <mailutils/mutil.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
* Reads an RFC822 defined lexical token from an input. All names are
* as close as possible to those used in the extended BNF of the RFC.
*/

/* From RFC 822, 3.3 Lexical Tokens */

extern int mu_parse822_is_char        (char c);
extern int mu_parse822_is_digit       (char c);
extern int mu_parse822_is_ctl         (char c);
extern int mu_parse822_is_space       (char c);
extern int mu_parse822_is_htab        (char c);
extern int mu_parse822_is_lwsp_char   (char c);
extern int mu_parse822_is_special     (char c);
extern int mu_parse822_is_atom_char   (char c);
extern int mu_parse822_is_q_text      (char c);
extern int mu_parse822_is_d_text      (char c);
extern int mu_parse822_is_smtp_q      (char c);

extern int mu_parse822_skip_crlf      (const char **p, const char *e);
extern int mu_parse822_skip_lwsp_char (const char **p, const char *e);
extern int mu_parse822_skip_lwsp      (const char **p, const char *e);
extern int mu_parse822_skip_comments  (const char **p, const char *e);
extern int mu_parse822_skip_nl        (const char **p, const char *e);

extern int mu_parse822_digits         (const char **p, const char *e,
				       int min, int max, int *digits);
extern int mu_parse822_special        (const char **p, const char *e, char c);
extern int mu_parse822_comment        (const char **p, const char *e,
				       char **comment);
extern int mu_parse822_atom           (const char **p, const char *e,
				       char **atom);
extern int mu_parse822_quoted_pair    (const char **p, const char *e,
				       char **qpair);
extern int mu_parse822_quoted_string  (const char **p, const char *e,
				       char **qstr);
extern int mu_parse822_word           (const char **p, const char *e,
				       char **word);
extern int mu_parse822_phrase         (const char **p, const char *e,
				       char **phrase);
extern int mu_parse822_d_text         (const char **p, const char *e,
				       char **dtext);

/* From RFC 822, 6.1 Address Specification Syntax */

extern int mu_parse822_address_list   (mu_address_t *a, const char *s,
				       mu_address_t hint, int hflags);
extern int mu_parse822_mail_box       (const char **p, const char *e,
				       mu_address_t *a,
				       mu_address_t hint, int hflags);
extern int mu_parse822_group          (const char **p, const char *e,
				       mu_address_t *a,
				       mu_address_t hint, int hflags);
extern int mu_parse822_address        (const char **p, const char *e,
				       mu_address_t *a,
				       mu_address_t hint, int hflags);
extern int mu_parse822_route_addr     (const char **p, const char *e,
				       mu_address_t *a,
				       mu_address_t hint, int hflags);
extern int mu_parse822_route          (const char **p, const char *e,
				       char **route);
extern int mu_parse822_addr_spec      (const char **p, const char *e,
				       mu_address_t *a,
				       mu_address_t hint, int hflags);
extern int mu_parse822_unix_mbox      (const char **p, const char *e,
				       mu_address_t *a,
				       mu_address_t hint, int hflags);
extern int mu_parse822_local_part     (const char **p, const char *e,
				       char **local_part);
extern int mu_parse822_domain         (const char **p, const char *e,
				       char **domain);
extern int mu_parse822_sub_domain     (const char **p, const char *e,
				       char **sub_domain);
extern int mu_parse822_domain_ref     (const char **p, const char *e,
				       char **domain_ref);
extern int mu_parse822_domain_literal (const char **p, const char *e,
				       char **domain_literal);

/* RFC 822 Quoting Functions
 * Various elements must be quoted if then contain non-safe characters. What
 * characters are allowed depend on the element. The following functions will
 * allocate a quoted version of the raw element, it may not actually be
 * quoted if no unsafe characters were in the raw string.
 */

extern int mu_parse822_quote_string     (char **quoted, const char *raw);
extern int mu_parse822_quote_local_part (char **quoted, const char *raw);

extern int mu_parse822_field_body       (const char **p, const char *e,
					 char **fieldbody);
extern int mu_parse822_field_name       (const char **p, const char *e,
					 char **fieldname);

/***** From RFC 822, 5.1 Date and Time Specification Syntax *****/

extern int mu_parse822_day       (const char **p, const char *e, int *day);
extern int mu_parse822_date      (const char **p, const char *e, int *day,
				  int *mon, int *year);
extern int mu_parse822_time      (const char **p, const char *e, int *h,
				  int *m, int *s, int *tz, const char **tz_name);
extern int mu_parse822_date_time (const char **p, const char *e,
				  struct tm *tm, mu_timezone *tz);


#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_PARSE822_H */

