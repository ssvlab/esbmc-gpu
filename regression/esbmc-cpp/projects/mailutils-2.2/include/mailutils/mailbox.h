/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2006, 2007, 2010 Free Software
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

#ifndef _MAILUTILS_MAILBOX_H
#define _MAILUTILS_MAILBOX_H

#include <sys/types.h>

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

extern char *mu_ticket_file;
  
int mu_set_mail_directory (const char *p);
int mu_set_mailbox_pattern (const char *pat);
void mu_set_folder_directory (const char *p);
const char *mu_mailbox_url (void);
const char *mu_folder_directory (void);
int mu_construct_user_mailbox_url (char **pout, const char *name);

/* Constructor/destructor and possible types.  */
extern int  mu_mailbox_create          (mu_mailbox_t *, const char *);
extern int  mu_mailbox_create_from_url (mu_mailbox_t *, mu_url_t);

extern void mu_mailbox_destroy         (mu_mailbox_t *);
extern int  mu_mailbox_create_default  (mu_mailbox_t *, const char *);

extern int  mu_mailbox_open            (mu_mailbox_t, int flag);
extern int  mu_mailbox_close           (mu_mailbox_t);
extern int  mu_mailbox_flush           (mu_mailbox_t mbox, int expunge);
extern int  mu_mailbox_get_folder      (mu_mailbox_t, mu_folder_t *);
extern int  mu_mailbox_set_folder      (mu_mailbox_t, mu_folder_t);
extern int  mu_mailbox_uidvalidity     (mu_mailbox_t, unsigned long *);
extern int  mu_mailbox_uidnext         (mu_mailbox_t, size_t *);

/* Messages.  */
extern int  mu_mailbox_get_message     (mu_mailbox_t, size_t msgno,
					mu_message_t *);
extern int  mu_mailbox_quick_get_message(mu_mailbox_t, mu_message_qid_t,
					 mu_message_t *); 
extern int  mu_mailbox_append_message  (mu_mailbox_t, mu_message_t);
extern int  mu_mailbox_messages_count  (mu_mailbox_t, size_t *);
extern int  mu_mailbox_messages_recent (mu_mailbox_t, size_t *);
extern int  mu_mailbox_message_unseen  (mu_mailbox_t, size_t *);
extern int  mu_mailbox_expunge         (mu_mailbox_t);
extern int  mu_mailbox_sync            (mu_mailbox_t);  
extern int  mu_mailbox_save_attributes (mu_mailbox_t)
                                       __attribute__ ((deprecated));

#define MU_UIDL_LENGTH 70
#define MU_UIDL_BUFFER_SIZE (MU_UIDL_LENGTH+1)
				       
struct mu_uidl
{
  size_t msgno;
  char uidl[MU_UIDL_BUFFER_SIZE];
};
extern int  mu_mailbox_get_uidls       (mu_mailbox_t, mu_list_t *);

/* Update and scanning.  */
extern int  mu_mailbox_get_size        (mu_mailbox_t, mu_off_t *size);
extern int  mu_mailbox_is_updated      (mu_mailbox_t);
extern int  mu_mailbox_scan            (mu_mailbox_t, size_t no, size_t *count);

/* Mailbox Stream.  */
extern int  mu_mailbox_set_stream      (mu_mailbox_t, mu_stream_t);
extern int  mu_mailbox_get_stream      (mu_mailbox_t, mu_stream_t *);

/* Lock settings.  */
extern int  mu_mailbox_get_locker      (mu_mailbox_t, mu_locker_t *);
extern int  mu_mailbox_set_locker      (mu_mailbox_t, mu_locker_t);

/* Property.  */
extern int  mu_mailbox_get_flags       (mu_mailbox_t, int *);
extern int  mu_mailbox_get_property    (mu_mailbox_t, mu_property_t *);

/* URL.  */
extern int  mu_mailbox_get_url         (mu_mailbox_t, mu_url_t *);

/* For any debuging */
extern int  mu_mailbox_has_debug       (mu_mailbox_t);
extern int  mu_mailbox_get_debug       (mu_mailbox_t, mu_debug_t *);
extern int  mu_mailbox_set_debug       (mu_mailbox_t, mu_debug_t);

/* Events.  */
extern int  mu_mailbox_get_observable  (mu_mailbox_t, mu_observable_t *);

/* Locking */  
extern int mu_mailbox_lock (mu_mailbox_t mbox);
extern int mu_mailbox_unlock (mu_mailbox_t mbox);

extern int mu_mailbox_get_iterator (mu_mailbox_t mbx,
				    mu_iterator_t *piterator);
  
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_MAILBOX_H */
