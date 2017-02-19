/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#ifndef _POP3D_H
#define _POP3D_H	1

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <mu_dbm.h>
#include <mu_asprintf.h>
#include <mu_umaxtostr.h>
#include <muaux.h>

/* The implementation */
#define	IMPL		"GNU POP3 Daemon"

/* APOP password file, without .db or .passwd, which are added based on file
   type automatically */
#define APOP_PASSFILE_NAME "apop"

#ifdef USE_DBM
# define APOP_PASSFILE SYSCONFDIR "/" APOP_PASSFILE_NAME
# define ENABLE_LOGIN_DELAY
#else
# define APOP_PASSFILE SYSCONFDIR "/" APOP_PASSFILE_NAME ".passwd"
# undef ENABLE_LOGIN_DELAY
#endif

#ifdef ENABLE_LOGIN_DELAY
# define LOGIN_STAT_FILE "/var/run/pop3-login"
extern time_t login_delay;
extern char *login_stat_file;
extern int check_login_delay (char *username);
extern void update_login_delay (char *username);
extern void login_delay_capa (void);
#else
# define check_login_delay(u) 0
# define update_login_delay(u)
# define login_delay_capa()
#endif

/* Minimum advertise retention time for messages.  */
extern unsigned expire;
extern int expire_on_exit;

#define EXPIRE_NEVER ((unsigned)-1)

/* Size of the MD5 digest for APOP */
#define APOP_DIGEST	70

/* Longest legal POP command */
#define POP_MAXCMDLEN	255

/* Buffer size to use for output */
#define BUFFERSIZE	1024

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#define _QNX_SOURCE
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <pwd.h>
#include <grp.h>
#include <syslog.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#include <sysexits.h>

#include <mailutils/alloc.h>
#include <mailutils/cctype.h>
#include <mailutils/cstr.h>
#include <mailutils/attribute.h>
#include <mailutils/body.h>
#include <mailutils/daemon.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/header.h>
#include <mailutils/list.h>
#include <mailutils/locker.h>
#include <mailutils/mailbox.h>
#include <mailutils/message.h>
#include <mailutils/mutil.h>
#include <mailutils/mu_auth.h>
#include <mailutils/nls.h>
#include <mailutils/registrar.h>
#include <mailutils/tls.h>
#include <mailutils/url.h>
#include <mailutils/md5.h>
#include <mailutils/acl.h>
#include <mailutils/server.h>

/* For Berkley DB2 APOP password file */
#ifdef HAVE_DB_H
#include <db.h>
#endif

#ifdef HAVE_SECURITY_PAM_APPL_H
#include <security/pam_appl.h>
#endif

#ifdef HAVE_SHADOW_H
#include <shadow.h>
#endif

#ifndef MAXHOSTNAMELEN
/* Maximum length of a hostname (is this defined somewhere else?).  */
/* MAXHOSTNAMELEN is already defined on Solaris.  */
# define MAXHOSTNAMELEN	64
#endif

#define POP3_ATTRIBUTE_DELE 0x0001
#define POP3_ATTRIBUTE_RETR 0x0010

#define INITIAL         -1
#define AUTHORIZATION	0
#define TRANSACTION	1
#define UPDATE		2
#define ABORT           3

#define OK		0
#define ERR_WRONG_STATE	1
#define ERR_BAD_ARGS	2
#define ERR_BAD_LOGIN	3
#define ERR_NO_MESG	4
#define ERR_MESG_DELE   5
#define ERR_NOT_IMPL	6
#define ERR_BAD_CMD	7
#define ERR_MBOX_LOCK	8
#define ERR_TOO_LONG	9
#define ERR_NO_MEM	10
#define ERR_SIGNAL	11
#define ERR_FILE        12
#define ERR_NO_IFILE    13
#define ERR_NO_OFILE    14
#define ERR_IO          15
#define ERR_PROTO       16
#define ERR_TIMEOUT	17
#define ERR_UNKNOWN	18
#define ERR_MBOX_SYNC   19
#define ERR_TLS_ACTIVE  20
#define ERR_TLS_IO      21
#define ERR_LOGIN_DELAY 22
#define ERR_TERMINATE   23

typedef struct mu_pop_server *mu_pop_server_t;
typedef int (*pop3d_command_handler_t) (char *);

struct pop3d_command
{
  const char *name;
  pop3d_command_handler_t handler;
};

extern mu_pop_server_t pop3srv;
extern mu_mailbox_t mbox;
extern int state;
extern int initial_state;
extern char *username;
extern char *maildir;
extern char *md5shared;
extern size_t children;
extern struct daemon_param daemon_param;
extern int debug_mode;
#ifdef WITH_TLS
extern int tls_available;
extern int tls_done;
#endif /* WITH_TLS */
extern int undelete_on_startup;
extern struct mu_auth_data *auth_data;
extern unsigned int idle_timeout;
extern int pop3d_transcript;

extern pop3d_command_handler_t pop3d_find_command (const char *name);

extern int pop3d_stat           (char *);
extern int pop3d_top            (char *);
extern int pop3d_uidl           (char *);
extern int pop3d_user           (char *);
extern int pop3d_apop           (char *);
extern int pop3d_auth           (char *);
extern int pop3d_capa           (char *);
extern int pop3d_dele           (char *);
extern int pop3d_list           (char *);
extern int pop3d_noop           (char *);
extern int pop3d_quit           (char *);
extern int pop3d_retr           (char *);
extern int pop3d_rset           (char *);

extern void pop3d_bye           (void);
extern int pop3d_abquit         (int);
extern char *pop3d_apopuser     (const char *);
extern int pop3d_lock           (void);
extern void process_cleanup     (void);

extern void pop3d_parse_command (char *cmd, char **pcmd, char **parg);

extern RETSIGTYPE pop3d_master_signal  (int);
extern RETSIGTYPE pop3d_child_signal  (int);

#ifdef WITH_TLS
extern int pop3d_stls           (char *);
extern void enable_stls (void);
#endif /* WITH_TLS */
extern int pop3d_touchlock      (void);
extern int pop3d_unlock         (void);
extern void pop3d_outf          (const char *fmt, ...) MU_PRINTFLIKE(1,2);

extern void pop3d_setio         (FILE *in, FILE *out);
extern char *pop3d_readline     (char *, size_t);
extern void pop3d_flush_output  (void);

extern int pop3d_is_master      (void);

extern void pop3d_mark_deleted (mu_attribute_t attr);
extern int pop3d_is_deleted (mu_attribute_t attr);
extern void pop3d_unset_deleted (mu_attribute_t attr);
void pop3d_undelete_all (void);

#ifdef WITH_TLS
extern int pop3d_init_tls_server    (void);
extern void pop3d_deinit_tls_server (void);
#endif /* WITH_TLS */

extern void pop3d_mark_retr (mu_attribute_t attr);
extern int pop3d_is_retr (mu_attribute_t attr);
extern void pop3d_unmark_retr (mu_attribute_t attr);

extern void expire_mark_message (mu_message_t msg, char **value);

extern void deliver_pending_bulletins (void);
extern void set_bulletin_db (const char *file);
extern int set_bulletin_source (const char *source);
extern int pop3d_begin_session (void);
extern const char *pop3d_error_string (int code);

#endif /* _POP3D_H */
