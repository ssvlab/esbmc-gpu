/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2006, 2007, 2009, 2010
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

#ifndef _MAILUTILS_MUTIL_H
#define _MAILUTILS_MUTIL_H

/*
   Collection of useful utility routines that are worth sharing,
   but don't have a natural home somewhere else.
*/

#include <time.h>

#include <mailutils/list.h>
#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned long mu_hex2ul (char hex);
extern size_t mu_hexstr2ul (unsigned long* ul, const char* hex, size_t len);

struct mu_timezone
{
  int utc_offset;
    /* Seconds east of UTC. */

  const char *tz_name;
    /* Nickname for this timezone, if known. It is always considered
     * to be a pointer to static string, so will never be freed. */
};

typedef struct mu_timezone mu_timezone;

int mu_parse_date (const char *p, time_t *rettime, const time_t *now);
extern int mu_parse_imap_date_time (const char **p, struct tm * tm,
				    mu_timezone * tz);
extern int mu_parse_ctime_date_time (const char **p, struct tm * tm,
				     mu_timezone * tz);

extern time_t mu_utc_offset (void);
extern time_t mu_tm2time (struct tm * timeptr, mu_timezone * tz);
extern char * mu_get_homedir (void);
extern char * mu_tilde_expansion (const char *ref, const char *delim, const char *homedir);

extern size_t mu_cpystr (char *dst, const char *src, size_t size);

/* Get the host name, doing a gethostbyname() if possible.
 *  
 * It is the caller's responsibility to free host.
 */
extern int mu_get_host_name (char **host);

/* Set the default user email address.
 *  
 * Subsequent calls to mu_get_user_email() with a NULL name will return this
 * email address.  email is parsed to determine that it consists of a a valid
 * rfc822 address, with one valid addr-spec, i.e, the address must be
 * qualified.
 */
extern int mu_set_user_email (const char *email);

/* Set the default user email address domain.
 *  
 * Subsequent calls to mu_get_user_email() with a non-null name will return
 * email addresses in this domain (name@domain). It should be fully
 * qualified, but this isn't (and can't) be enforced.
 */
extern int mu_set_user_email_domain (const char *domain);

/* Return the currently set user email domain, or NULL if not set. */
extern int mu_get_user_email_domain (const char** domain);

/* Same, but allocates memory */
extern int mu_aget_user_email_domain (char **pdomain);

/*
 * Get the default email address for user name. A NULL name is taken
 * to mean the current user.
 *  
 * The result must be freed by the caller after use.
 */
extern char *mu_get_user_email (const char *name);

extern char *mu_make_file_name (const char *dir, const char *file);  
extern char *mu_normalize_path (char *path);
extern int mu_tempfile (const char *tmpdir, char **namep);
extern char *mu_tempname (const char *tmpdir);

extern char * mu_get_full_path (const char *file);
extern char * mu_getcwd (void);
  
extern int mu_spawnvp(const char *prog, char *av[], int *stat);

extern int mu_unroll_symlink (char *out, size_t outsz, const char *in);

extern char * mu_expand_path_pattern (const char *pattern, const char *username);

extern int mu_rfc2822_msg_id (int subpart, char **pstr);
extern int mu_rfc2822_references (mu_message_t msg, char **pstr);
extern int mu_rfc2822_in_reply_to (mu_message_t msg, char **pstr);

/* Find NEEDLE in the HAYSTACK. Case insensitive comparison */
extern char *mu_strcasestr (const char *haystack, const char *needle);

extern int mu_string_unfold (char *text, size_t *plen);

extern int mu_unre_set_regex (const char *str, int caseflag, char **errp);
extern int mu_unre_subject  (const char *subject, const char **new_subject);

extern const char *mu_charset_lookup (char *lang, char *terr);

extern int mu_true_answer_p (const char *p);
extern int mu_scheme_autodetect_p (mu_url_t);

struct timeval; 
  
extern int mu_fd_wait (int fd, int *pflags, struct timeval *tvp);

extern int mu_decode_filter (mu_stream_t *pfilter, mu_stream_t input,
			     const char *filter_type,
			     const char *fromcode, const char *tocode);

extern enum mu_iconv_fallback_mode mu_default_fallback_mode;
extern int mu_set_default_fallback (const char *str);

extern int mu_is_proto (const char *p);

extern int mu_mh_delim (const char *str);

extern size_t mu_strftime (char *s, size_t max, const char *format,
			   const struct tm *tm);
  

extern int mutil_parse_field_map (const char *map, mu_assoc_t *passoc_tab,
				  int *perr);

extern int mu_stream_flags_to_mode (int flags, int isdir);

extern int mu_parse_stream_perm_string (int *pmode, const char *str,
					const char **endp);
  

#ifdef __cplusplus
}
#endif

#endif

