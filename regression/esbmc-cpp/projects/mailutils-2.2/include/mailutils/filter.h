/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2010 Free Software
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

#ifndef _MAILUTILS_FILTER_H
#define _MAILUTILS_FILTER_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Type.  */
#define MU_FILTER_DECODE 0
#define MU_FILTER_ENCODE 1

/* Direction.  */
#define MU_FILTER_READ  MU_STREAM_READ
#define MU_FILTER_WRITE MU_STREAM_WRITE
#define MU_FILTER_RDWR  MU_STREAM_RDWR

struct _mu_filter_record
{
  const char *name;
  int  (*_mu_filter)     (mu_filter_t);
  void *data;

  /* Stub function return the fields.  */
  int (*_is_filter)  (mu_filter_record_t, const char *);
  int (*_get_filter) (mu_filter_record_t, int (*(*_mu_filter)) (mu_filter_t));
};


extern int mu_filter_create   (mu_stream_t *, mu_stream_t, const char*, int, int);
extern int mu_filter_get_list (mu_list_t *);

/* List of defaults.  */
extern mu_filter_record_t mu_rfc822_filter;
extern mu_filter_record_t mu_qp_filter; /* quoted-printable.  */
extern mu_filter_record_t mu_base64_filter;
extern mu_filter_record_t mu_binary_filter;
extern mu_filter_record_t mu_bit8_filter;
extern mu_filter_record_t mu_bit7_filter;
extern mu_filter_record_t mu_rfc_2047_Q_filter;
extern mu_filter_record_t mu_rfc_2047_B_filter;
  
enum mu_iconv_fallback_mode {
  mu_fallback_none,
  mu_fallback_copy_pass,
  mu_fallback_copy_octal
};

extern int mu_filter_iconv_create (mu_stream_t *s, mu_stream_t transport,
				const char *fromcode, const char *tocode,
				int flags,
				enum mu_iconv_fallback_mode fallback_mode);

  
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_FILTER_H */
