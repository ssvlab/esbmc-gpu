/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2004, 2005, 2006, 2007, 2008, 2009, 2010
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

#ifndef _MAILUTILS_REGISTRAR_H
#define _MAILUTILS_REGISTRAR_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Public Interface, to allow static initialization.  */
struct _mu_record
{
  int priority;    /* Higher priority records are scanned first */
  const char *scheme;
  int (*_url) (mu_url_t);
  int (*_mailbox) (mu_mailbox_t);
  int (*_mailer) (mu_mailer_t);
  int (*_folder) (mu_folder_t);
  void *data; /* back pointer.  */

  /* Stub functions to override. The default is to return the fields.  */
  int (*_is_scheme) (mu_record_t, mu_url_t, int);
  int (*_get_url) (mu_record_t, int (*(*_mu_url)) (mu_url_t));
  int (*_get_mailbox) (mu_record_t, int (*(*_mu_mailbox)) (mu_mailbox_t));
  int (*_get_mailer) (mu_record_t, int (*(*_mu_mailer)) (mu_mailer_t));
  int (*_get_folder) (mu_record_t, int (*(*_mu_folder)) (mu_folder_t));

  int (*_list_p) (mu_record_t, const char *, int);
};

/* Defaults */  
extern int mu_registrar_set_default_scheme (const char *scheme);
extern const char *mu_registrar_get_default_scheme (void);
extern int mu_registrar_get_default_record (mu_record_t *prec);
extern void mu_registrar_set_default_record (mu_record_t record);
  
/* Registration.  */
extern int mu_registrar_get_iterator (mu_iterator_t *);
extern int mu_registrar_get_list (mu_list_t *) __attribute__ ((deprecated));

extern int mu_registrar_lookup_scheme (const char *scheme,
				       mu_record_t *precord);
  
extern int mu_registrar_lookup (const char *name, int flags, 
                                mu_record_t *precord, int *pflags);
extern int mu_registrar_lookup_url (mu_url_t url, int flags,
				    mu_record_t *precord, int *pflags);
extern int mu_registrar_record       (mu_record_t);
extern int mu_unregistrar_record     (mu_record_t);

/* Scheme.  */
extern int mu_record_is_scheme       (mu_record_t, mu_url_t, int flags);
extern int mu_record_set_scheme      (mu_record_t, const char *);
extern int mu_record_set_is_scheme   (mu_record_t, 
                      int (*_is_scheme) (mu_record_t, mu_url_t, int));

/* Url.  */
extern int mu_record_get_url         (mu_record_t, int (*(*)) (mu_url_t));
extern int mu_record_set_url         (mu_record_t, int (*) (mu_url_t));
extern int mu_record_set_get_url     (mu_record_t, 
                      int (*_get_url) (mu_record_t, int (*(*)) (mu_url_t)));
/*  Mailbox. */
extern int mu_record_get_mailbox     (mu_record_t, int (*(*)) (mu_mailbox_t));
extern int mu_record_set_mailbox     (mu_record_t, int (*) (mu_mailbox_t));
extern int mu_record_set_get_mailbox (mu_record_t, 
                 int (*_get_mailbox) (mu_record_t, int (*(*)) (mu_mailbox_t)));
/* Mailer.  */
extern int mu_record_get_mailer      (mu_record_t,
				   int (*(*)) (mu_mailer_t));
extern int mu_record_set_mailer      (mu_record_t, int (*) (mu_mailer_t));
extern int mu_record_set_get_mailer  (mu_record_t, 
        int (*_get_mailer) (mu_record_t, int (*(*)) (mu_mailer_t)));
/* Folder.  */
extern int mu_record_get_folder      (mu_record_t, int (*(*)) (mu_folder_t));
extern int mu_record_set_folder      (mu_record_t, int (*) (mu_folder_t));
extern int mu_record_set_get_folder  (mu_record_t, 
         int (*_get_folder) (mu_record_t, int (*(*)) (mu_folder_t)));

extern int mu_record_list_p (mu_record_t record, const char *name, int);
  
/* Records provided by the library.  */

/* Remote Folder "imap://"  */
extern mu_record_t mu_imap_record;
extern mu_record_t mu_imaps_record;
/* Remote Mailbox POP3, pop://  */
extern mu_record_t mu_pop_record;
extern mu_record_t mu_pops_record;
/* Remote newsgroup NNTP, nntp://  */
extern mu_record_t mu_nntp_record;

/* Local Mailbox Unix Mailbox, "mbox:"  */
extern mu_record_t mu_mbox_record;
/* Local MH, "mh:" */
extern mu_record_t mu_mh_record;
/* Maildir, "maildir:" */
extern mu_record_t mu_maildir_record;
  
#define MU_IMAP_PRIO        100
#define MU_POP_PRIO         200
#define MU_MBOX_PRIO        300 
#define MU_MH_PRIO          400
#define MU_MAILDIR_PRIO     500 
#define MU_NNTP_PRIO        600
#define MU_PATH_PRIO        1000
  
#define MU_SMTP_PRIO        10000
#define MU_SENDMAIL_PRIO    10000
#define MU_PROG_PRIO        10000
  
/* SMTP mailer, "smtp://"  */
extern mu_record_t mu_smtp_record;
/* Sendmail, "sendmail:"  */
extern mu_record_t mu_sendmail_record;
/* Program mailer, "prog://", "|" */
extern mu_record_t mu_prog_record;
  
#define mu_register_all_mbox_formats() do {\
  mu_registrar_record (mu_mbox_record);\
  mu_registrar_record (mu_pop_record);\
  mu_registrar_record (mu_pops_record);\
  mu_registrar_record (mu_imap_record);\
  mu_registrar_record (mu_imaps_record);\
  mu_registrar_record (mu_mh_record);\
  mu_registrar_record (mu_maildir_record);\
  mu_registrar_set_default_record (MU_DEFAULT_RECORD);\
} while (0)

#define mu_register_local_mbox_formats() do {\
  mu_registrar_record (mu_mbox_record);\
  mu_registrar_record (mu_mh_record);\
  mu_registrar_record (mu_maildir_record);\
  mu_registrar_set_default_record (MU_DEFAULT_RECORD);\
} while (0)

#define mu_register_remote_mbox_formats() do {\
  mu_registrar_record (mu_pop_record);\
  mu_registrar_record (mu_pops_record);\
  mu_registrar_record (mu_imap_record);\
  mu_registrar_record (mu_imaps_record);\
  mu_registrar_record (mu_nntp_record);\
} while (0)

#define mu_register_all_mailer_formats() do {\
  mu_registrar_record (mu_sendmail_record);\
  mu_registrar_record (mu_smtp_record);\
  mu_registrar_record (mu_prog_record);\
} while (0)

#define mu_register_extra_formats() do {\
  mu_registrar_record (mu_nntp_record);\
} while (0)

#define mu_register_all_formats() do {\
  mu_register_all_mbox_formats ();\
  mu_register_all_mailer_formats ();\
  mu_register_extra_formats();\
} while (0)

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_REGISTRAR_H */
