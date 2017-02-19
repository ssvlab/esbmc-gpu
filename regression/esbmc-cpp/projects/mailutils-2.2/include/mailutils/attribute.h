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

#ifndef _MAILUTILS_ATTRIBUTE_H
#define _MAILUTILS_ATTRIBUTE_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MU_ATTRIBUTE_ANSWERED 0x01
#define MU_ATTRIBUTE_FLAGGED  0x02
#define MU_ATTRIBUTE_DELETED  0x04
#define MU_ATTRIBUTE_DRAFT    0x08
#define MU_ATTRIBUTE_SEEN     0x10
#define MU_ATTRIBUTE_READ     0x20
#define MU_ATTRIBUTE_MODIFIED 0x40

/* A message is recent if the current session is the first session
   to have been notified about it. Practically, a message is considered
   "recent" if it does not have MU_ATTRIBUTE_SEEN set. For consistency
   a pseudo-attribute is provided: */
#define MU_ATTRIBUTE_RECENT   0 

#define MU_ATTRIBUTE_IS_UNSEEN(f) \
      ((f) == 0 || ! ((f) & MU_ATTRIBUTE_SEEN))

#define MU_ATTRIBUTE_IS_UNREAD(f) \
      ((f) == 0 || ! ((f) & MU_ATTRIBUTE_READ))

extern int mu_attribute_create          (mu_attribute_t *, void *);
extern void mu_attribute_destroy        (mu_attribute_t *, void *);
extern void * mu_attribute_get_owner    (mu_attribute_t);
extern int mu_attribute_is_modified     (mu_attribute_t);
extern int mu_attribute_clear_modified  (mu_attribute_t);
extern int mu_attribute_set_modified    (mu_attribute_t attr);

extern int mu_attribute_is_userflag     (mu_attribute_t, int);
extern int mu_attribute_is_seen         (mu_attribute_t);
extern int mu_attribute_is_answered     (mu_attribute_t);
extern int mu_attribute_is_flagged      (mu_attribute_t);
extern int mu_attribute_is_deleted      (mu_attribute_t);
extern int mu_attribute_is_draft        (mu_attribute_t);
extern int mu_attribute_is_recent       (mu_attribute_t);
extern int mu_attribute_is_read         (mu_attribute_t);

extern int mu_attribute_set_userflag    (mu_attribute_t, int);
extern int mu_attribute_set_seen        (mu_attribute_t);
extern int mu_attribute_set_answered    (mu_attribute_t);
extern int mu_attribute_set_flagged     (mu_attribute_t);
extern int mu_attribute_set_deleted     (mu_attribute_t);
extern int mu_attribute_set_draft       (mu_attribute_t);
extern int mu_attribute_set_recent      (mu_attribute_t);
extern int mu_attribute_set_read        (mu_attribute_t);

extern int mu_attribute_unset_userflag  (mu_attribute_t, int);
extern int mu_attribute_unset_seen      (mu_attribute_t);
extern int mu_attribute_unset_answered  (mu_attribute_t);
extern int mu_attribute_unset_flagged   (mu_attribute_t);
extern int mu_attribute_unset_deleted   (mu_attribute_t);
extern int mu_attribute_unset_draft     (mu_attribute_t);
extern int mu_attribute_unset_recent    (mu_attribute_t);
extern int mu_attribute_unset_read      (mu_attribute_t);

extern int mu_attribute_get_flags       (mu_attribute_t, int *);
extern int mu_attribute_set_flags       (mu_attribute_t, int);
extern int mu_attribute_unset_flags     (mu_attribute_t, int);

extern int mu_attribute_set_set_flags   (mu_attribute_t,
				      int (*_set_flags) (mu_attribute_t, int),
				      void *);
extern int mu_attribute_set_unset_flags (mu_attribute_t,
				      int (*_unset_flags) (mu_attribute_t, int),
				      void *);
extern int mu_attribute_set_get_flags   (mu_attribute_t,
				      int (*_get_flags) (mu_attribute_t, int *),
				      void *);
extern int mu_attribute_is_equal        (mu_attribute_t, mu_attribute_t att2);

extern int mu_attribute_copy            (mu_attribute_t, mu_attribute_t);

/* Maximum size of buffer for mu_attribute_to_string call, including nul */
#define MU_STATUS_BUF_SIZE sizeof("OAFRd")
  
extern int mu_attribute_to_string       (mu_attribute_t, char *, size_t, size_t *);
extern int mu_string_to_flags           (const char *, int *);

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_ATTRIBUTE_H */
