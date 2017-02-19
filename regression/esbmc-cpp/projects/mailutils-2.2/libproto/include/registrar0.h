/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2004, 2005, 2007, 2008, 2009, 2010 Free
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

#ifndef _REGISTRAR0_H
#define _REGISTRAR0_H

#include <mailutils/registrar.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The pop and imap defines are all wrong now, since they seem intertwined
   with the old url parsing code. Also, "pop://" is not the POP scheme,
   at least not as a scheme is described in the RFCs.

   Perhaps they can be changed?
*/
#define MU_POP_PORT 110
#define MU_POP_SCHEME "pop"
#define MU_POP_SCHEME_LEN (sizeof (MU_POP_SCHEME) - 1)
extern int _url_pop_init          (mu_url_t);
extern int _mailbox_pop_init      (mu_mailbox_t);
extern int _folder_pop_init       (mu_folder_t);

#define MU_POPS_PORT 995
#define MU_POPS_SCHEME "pops"
#define MU_POPS_SCHEME_LEN (sizeof (MU_POPS_SCHEME) - 1)
extern int _url_pops_init         (mu_url_t);
extern int _mailbox_pops_init     (mu_mailbox_t);

#define MU_IMAP_PORT 143
#define MU_IMAP_SCHEME "imap"
#define MU_IMAP_SCHEME_LEN (sizeof (MU_IMAP_SCHEME) - 1)
extern int _url_imap_init         (mu_url_t);
extern int _mailbox_imap_init     (mu_mailbox_t);
extern int _folder_imap_init      (mu_folder_t);

#define MU_IMAPS_PORT 993
#define MU_IMAPS_SCHEME "imaps"
#define MU_IMAPS_SCHEME_LEN (sizeof (MU_IMAPS_SCHEME) - 1)
extern int _url_imaps_init        (mu_url_t);
extern int _mailbox_imaps_init    (mu_mailbox_t);

#define MU_MBOX_SCHEME "mbox"
#define MU_MBOX_SCHEME_LEN (sizeof (MU_MBOX_SCHEME) - 1)
extern int _mailbox_mbox_init     (mu_mailbox_t);
extern int _folder_mbox_init      (mu_folder_t);

#define MU_FILE_SCHEME "file"
#define MU_FILE_SCHEME_LEN (sizeof (MU_FILE_SCHEME) - 1)

#define MU_PATH_SCHEME "/"
#define MU_PATH_SCHEME_LEN (sizeof (MU_PATH_SCHEME) - 1)
extern int _mailbox_path_init     (mu_mailbox_t);
extern int _folder_path_init      (mu_folder_t);

#define MU_SMTP_SCHEME "smtp"
#define MU_SMTP_SCHEME_LEN (sizeof (MU_SMTP_SCHEME) - 1)
#define MU_SMTP_PORT 25

#define MU_SENDMAIL_SCHEME "sendmail"
#define MU_SENDMAIL_SCHEME_LEN (sizeof (MU_SENDMAIL_SCHEME) - 1)
extern int _mu_mailer_sendmail_init (mu_mailer_t mailer);
  
#define MU_PROG_SCHEME "prog"
#define MU_PROG_SCHEME_LEN (sizeof (MU_PROG_SCHEME) - 1)
  extern int _mu_mailer_prog_init  (mu_mailer_t);  
  
#define MU_MH_SCHEME "mh"
#define MU_MH_SCHEME_LEN (sizeof (MU_MH_SCHEME) - 1)
extern int _mailbox_mh_init (mu_mailbox_t mailbox);

#define MU_MAILDIR_SCHEME "maildir"
#define MU_MAILDIR_SCHEME_LEN (sizeof (MU_MAILDIR_SCHEME) - 1)
extern int _mailbox_maildir_init (mu_mailbox_t mailbox);

#ifdef __cplusplus
}
#endif

#endif /* _REGISTRAR0_H */
