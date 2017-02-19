/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2005, 2007, 2010 Free Software
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

#ifndef _MAILUTILS_MIME_H
#define _MAILUTILS_MIME_H

#include <mailutils/types.h>

/* mime flags */
#define MU_MIME_MULTIPART_MIXED	    0x1
#define MU_MIME_MULTIPART_ALT       0x2

#ifdef __cplusplus
extern "C" {
#endif

int mu_mime_create	(mu_mime_t *pmime, mu_message_t msg, int flags);
void mu_mime_destroy	(mu_mime_t *pmime);
int mu_mime_is_multipart	(mu_mime_t mime);
int mu_mime_get_num_parts	(mu_mime_t mime, size_t *nparts);

int mu_mime_get_part	(mu_mime_t mime, size_t part, mu_message_t *msg);

int mu_mime_add_part	(mu_mime_t mime, mu_message_t msg);

int mu_mime_get_message	(mu_mime_t mime, mu_message_t *msg);

int mu_rfc2047_decode   (const char *tocode, const char *fromstr, 
                         char **ptostr);

int mu_rfc2047_encode   (const char *charset, const char *encoding, 
			 const char *text, char **result);

int mu_base64_encode    (const unsigned char *input, size_t input_len,
			 unsigned char **output, size_t * output_len);

int mu_base64_decode    (const unsigned char *input, size_t input_len,
			 unsigned char **output, size_t * output_len);

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_MIME_H */
