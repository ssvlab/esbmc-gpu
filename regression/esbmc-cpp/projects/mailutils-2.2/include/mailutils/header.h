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

#ifndef _MAILUTILS_HEADER_H
#define _MAILUTILS_HEADER_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MU_HEADER_UNIX_FROM                 "From "
#define MU_HEADER_RETURN_PATH               "Return-Path"
#define MU_HEADER_RECEIVED                  "Received"
#define MU_HEADER_DATE                      "Date"
#define MU_HEADER_DCC                       "Dcc"  
#define MU_HEADER_FROM                      "From"
#define MU_HEADER_SENDER                    "Sender"
#define MU_HEADER_RESENT_FROM               "Resent-From"
#define MU_HEADER_SUBJECT                   "Subject"
#define MU_HEADER_SENDER                    "Sender"
#define MU_HEADER_RESENT_SENDER             "Resent-SENDER"
#define MU_HEADER_TO                        "To"
#define MU_HEADER_RESENT_TO                 "Resent-To"
#define MU_HEADER_CC                        "Cc"
#define MU_HEADER_RESENT_CC                 "Resent-Cc"
#define MU_HEADER_BCC                       "Bcc"
#define MU_HEADER_RESENT_BCC                "Resent-Bcc"
#define MU_HEADER_REPLY_TO                  "Reply-To"
#define MU_HEADER_RESENT_REPLY_TO           "Resent-Reply-To"
#define MU_HEADER_MESSAGE_ID                "Message-ID"
#define MU_HEADER_RESENT_MESSAGE_ID         "Resent-Message-ID"
#define MU_HEADER_IN_REPLY_TO               "In-Reply-To"
#define MU_HEADER_REFERENCE                 "Reference"
#define MU_HEADER_REFERENCES                "References"
#define MU_HEADER_ENCRYPTED                 "Encrypted"
#define MU_HEADER_PRECEDENCE                "Precedence"
#define MU_HEADER_STATUS                    "Status"
#define MU_HEADER_CONTENT_LENGTH            "Content-Length"
#define MU_HEADER_CONTENT_LANGUAGE          "Content-Language"
#define MU_HEADER_CONTENT_TRANSFER_ENCODING "Content-transfer-encoding"
#define MU_HEADER_CONTENT_ID                "Content-ID"
#define MU_HEADER_CONTENT_TYPE              "Content-Type"
#define MU_HEADER_CONTENT_DESCRIPTION       "Content-Description"
#define MU_HEADER_CONTENT_DISPOSITION       "Content-Disposition"
#define MU_HEADER_CONTENT_MD5               "Content-MD5"
#define MU_HEADER_MIME_VERSION              "MIME-Version"
#define MU_HEADER_X_MAILER                  "X-Mailer"
#define MU_HEADER_X_UIDL                    "X-UIDL"
#define MU_HEADER_X_UID                     "X-UID"
#define MU_HEADER_X_IMAPBASE                "X-IMAPbase"
#define MU_HEADER_ENV_SENDER                "X-Envelope-Sender"
#define MU_HEADER_ENV_DATE                  "X-Envelope-Date"
#define MU_HEADER_FCC                       "Fcc"
#define MU_HEADER_DELIVERY_DATE             "Delivery-date"
#define MU_HEADER_ENVELOPE_TO               "Envelope-to"
#define MU_HEADER_X_EXPIRE_TIMESTAMP        "X-Expire-Timestamp"
  
#define MU_HEADER_REPLACE 0x01
#define MU_HEADER_BEFORE  0x02

extern int mu_header_create (mu_header_t *, const char *, size_t, void *);
extern void mu_header_destroy (mu_header_t *, void *);
extern void *mu_header_get_owner (mu_header_t);

extern int mu_header_is_modified (mu_header_t);
extern int mu_header_clear_modified (mu_header_t);

/* Set and get field values by field name. */
extern int mu_header_set_value (mu_header_t, const char *, const char *, int);
extern int mu_header_remove (mu_header_t, const char *, int);
extern int mu_header_append (mu_header_t header, const char *fn,
			     const char *fv);
extern int mu_header_prepend (mu_header_t header, const char *fn,
			      const char *fv);
extern int mu_header_insert (mu_header_t, const char *, const char *, 
			     const char *, int, int);
  
extern int mu_header_sget_value_n (mu_header_t, const char *, int,
				   const char **);
#define mu_header_sget_value(header, name, pval) \
  mu_header_sget_value_n (header, name, 1, pval)

extern int mu_header_get_value_n (mu_header_t, const char *, int, char *,
				  size_t, size_t *);
#define mu_header_get_value(header, name, buffer, buflen, pn) \
  mu_header_get_value_n (header, name, 1, buffer, buflen, pn)
  
extern int mu_header_aget_value_n (mu_header_t, const char *, int, char **);
#define mu_header_aget_value(header, name, pptr) \
  mu_header_aget_value_n (header, name, 1, pptr)
  
/* Get field values as an mu_address_t. */
extern int mu_header_get_address_n (mu_header_t, const char *,
				    int, mu_address_t *);
#define mu_header_get_address(header, name, addr) \
  mu_header_get_address_n (header, name, 1, addr)
  
/* Set and get field values by field index (1-based). */
extern int mu_header_get_field_count (mu_header_t, size_t *count);

extern int mu_header_sget_field_name (mu_header_t, size_t index,
				      const char **);
  
extern int mu_header_get_field_name (mu_header_t, size_t index,
				     char *, size_t, size_t *);
extern int mu_header_aget_field_name (mu_header_t, size_t index,
				      char **);
  
extern int mu_header_sget_field_value (mu_header_t, size_t index,
				       const char **);
extern int mu_header_get_field_value (mu_header_t, size_t index,
				      char *, size_t, size_t *);
extern int mu_header_aget_field_value (mu_header_t, size_t index, char **);

extern int mu_header_get_value_unfold_n (mu_header_t header,
					 const char *name, int n,
					 char *buffer, size_t buflen,
					 size_t *pn);
#define mu_header_get_value_unfold(header, name, buffer, buflen, pn) \
  mu_header_get_value_unfold_n (header, name, 1, buffer, buflen, pn)
  
extern int mu_header_aget_value_unfold_n (mu_header_t header,
					  const char *name, int n,
					  char **pvalue);
#define mu_header_aget_value_unfold(header, name, pvalue) \
  mu_header_aget_value_unfold_n (header, name, 1, pvalue)

extern int mu_header_get_field_value_unfold (mu_header_t header, size_t num,
					     char *buf, size_t buflen,
					     size_t *nwritten);
extern int mu_header_aget_field_value_unfold (mu_header_t header, size_t num,
					      char **pvalue);

extern int mu_header_get_stream (mu_header_t, mu_stream_t *);
/* FIXME: This function does not exist:
   extern int mu_header_set_stream (mu_header_t, mu_stream_t, void *);
*/
extern int mu_header_size (mu_header_t, size_t *);
extern int mu_header_lines (mu_header_t, size_t *);

extern int mu_header_get_iterator (mu_header_t, mu_iterator_t *);
  

extern int mu_header_set_fill (mu_header_t,
      int (*_fill) (mu_header_t, char *, size_t, mu_off_t, size_t *), void *owner);

  
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_HEADER_H */
