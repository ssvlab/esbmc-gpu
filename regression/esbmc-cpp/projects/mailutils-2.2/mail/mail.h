/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2006, 2007, 2009, 2010
   Free Software Foundation, Inc.

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

#ifndef _MAIL_H
#define _MAIL_H 1

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef DMALLOC
# include <dmalloc.h>
#endif

#ifdef HAVE_ALLOCA_H
# include <alloca.h>
#endif
#include <errno.h>
#include <limits.h>
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#include <sys/wait.h>
#include <sys/types.h>
#include <stdarg.h>
#include <signal.h>

#include <confpaths.h>

#include <xalloc.h>

#ifdef HAVE_READLINE_READLINE_H
# include <readline/readline.h>
# include <readline/history.h>
#endif

#include <mailutils/address.h>
#include <mailutils/assoc.h>
#include <mailutils/attribute.h>
#include <mailutils/body.h>
#include <mailutils/debug.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/envelope.h>
#include <mailutils/filter.h>
#include <mailutils/header.h>
#include <mailutils/iterator.h>
#include <mailutils/list.h>
#include <mailutils/mailbox.h>
#include <mailutils/mailer.h>
#include <mailutils/message.h>
#include <mailutils/mutil.h>
#include <mailutils/registrar.h>
#include <mailutils/stream.h>
#include <mailutils/url.h>
#include <mailutils/nls.h>
#include <mailutils/tls.h>
#include <mailutils/argcv.h>
#include <mailutils/cctype.h>
#include <mailutils/cstr.h>
#include <mu_asprintf.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Type definitions */
#ifndef function_t
typedef int function_t (int, char **);
#endif

/* Values for mail_command_entry.flags */
#define EF_REG    0x00    /* Regular command */
#define EF_FLOW   0x01    /* Flow control command */
#define EF_SEND   0x02    /* Send command */
#define EF_HIDDEN 0x04    /* Hiddent command */
  
typedef struct compose_env
{
  mu_header_t header;   /* The message headers */
  char *filename;    /* Name of the temporary compose file */
  FILE *file;        /* Temporary compose file */
  FILE *ofile;       /* Diagnostics output channel */
  char **outfiles;   /* Names of the output files. The message is to be
		        saved in each of these. */
  int nfiles;        /* Number of output files */
} compose_env_t;

#define MAIL_COMMAND_COMMON_MEMBERS \
  const char *shortname;\
  const char *longname;\
  const char *synopsis
  
struct mail_command
{
  MAIL_COMMAND_COMMON_MEMBERS;
};

struct mail_command_entry
{
  MAIL_COMMAND_COMMON_MEMBERS;
  int flags;
  int (*func) (int, char **);
  char **(*command_completion) (int argc, char **argv, int ws);
};

struct mail_escape_entry
{
  const char *shortname;
  const char *longname;
  const char *synopsis;
  int (*escfunc) (int, char **, compose_env_t *);
};

enum mailvar_type
  {
    mailvar_type_whatever,
    mailvar_type_number,
    mailvar_type_string,
    mailvar_type_boolean
  };

union mailvar_value
{
  char *string;
  int number;
  int bool;
};

struct mailvar_variable
{
  char *name;
  enum mailvar_type type;
  int set;
  union mailvar_value value;
};

typedef struct message_set msgset_t;

struct message_set
{
  msgset_t *next;       /* Link to the next message set */
  unsigned int npart;   /* Number of parts in this set */
  size_t *msg_part;     /* Array of part numbers: msg_part[0] is the 
                           message number */
};

typedef int (*msg_handler_t) (msgset_t *mp, mu_message_t mesg, void *data);

/* Global variables and constants*/
extern mu_mailbox_t mbox;
extern size_t total;
extern FILE *ofile;
extern int interactive;
extern const char *program_version;

/* Functions */
extern int mail_alias (int argc, char **argv);
extern int mail_alt (int argc, char **argv);	/* command alternates */
extern int mail_cd (int argc, char **argv);
extern int mail_copy (int argc, char **argv);
extern int mail_decode (int argc, char **argv);
extern int mail_delete (int argc, char **argv);
extern int mail_discard (int argc, char **argv);
extern int mail_dp (int argc, char **argv);
extern int mail_echo (int argc, char **argv);
extern int mail_edit (int argc, char **argv);
extern int mail_else (int argc, char **argv);
extern int mail_endif (int argc, char **argv);
extern int mail_exit (int argc, char **argv);
extern int mail_file (int argc, char **argv);
extern int mail_folders (int argc, char **argv);
extern int mail_followup (int argc, char **argv);
extern int mail_from (int argc, char **argv);
extern int mail_from0 (msgset_t *mspec, mu_message_t msg, void *data);
extern void mail_compile_headline (struct mailvar_variable *var);

extern int mail_headers (int argc, char **argv);
extern int mail_hold (int argc, char **argv);
extern int mail_help (int argc, char **argv);
extern int mail_if (int argc, char **argv);
extern int mail_inc (int argc, char **argv);
extern int mail_list (int argc, char **argv);
extern int mail_send (int argc, char **argv);	/* command mail */
extern int mail_mbox (int argc, char **argv);
extern int mail_next (int argc, char **argv);
extern int mail_nounfold (int argc, char **argv);
extern int mail_variable (int argc, char **argv);
extern int mail_pipe (int argc, char **argv);
extern int mail_previous (int argc, char **argv);
extern int mail_print (int argc, char **argv);
extern int mail_quit (int argc, char **argv);
extern int mail_reply (int argc, char **argv);
extern int mail_retain (int argc, char **argv);
extern int mail_save (int argc, char **argv);
extern int mail_sendheader (int argc, char **argv);
extern int mail_set (int argc, char **argv);
extern int mail_shell (int argc, char **argv);
extern int mail_execute (int shell, int argc, char **argv);
extern int mail_size (int argc, char **argv);
extern int mail_source (int argc, char **argv);
extern int mail_summary (int argc, char **argv);
extern int mail_tag (int argc, char **argv);
extern int mail_top (int argc, char **argv);
extern int mail_touch (int argc, char **argv);
extern int mail_unalias (int argc, char **argv);
extern int mail_undelete (int argc, char **argv);
extern int mail_unfold (int argc, char **argv);
extern int mail_unset (int argc, char **argv);
extern int mail_version (int argc, char **argv);
extern int mail_visual (int argc, char **argv);
extern int mail_warranty (int argc, char **argv);
extern int mail_write (int argc, char **argv);
extern int mail_z (int argc, char **argv);
extern int mail_eq (int argc, char **argv);	/* command = */
extern int mail_setenv (int argc, char **argv);
extern int mail_envelope (int argc, char **argv);
extern int print_envelope (msgset_t *mspec, mu_message_t msg, void *data);
extern int mail_struct (int argc, char **argv);

extern int if_cond (void);

extern void mail_mainloop (char *(*input) (void *, int), void *closure, int do_history);
extern int mail_copy0 (int argc, char **argv, int mark);
extern int mail_send0 (compose_env_t *env, int save_to);
extern void free_env_headers (compose_env_t *env);

/*extern void print_message (mu_message_t mesg, char *prefix, int all_headers, FILE *file);*/

extern int mail_mbox_commit (void);
extern int mail_is_my_name (const char *name);
extern void mail_set_my_name (char *name);
extern char *mail_whoami (void);
extern int mail_header_is_visible (const char *str);
extern int mail_header_is_unfoldable (const char *str);
extern int mail_mbox_close (void);
extern char *mail_expand_name (const char *name);

extern void send_append_header (char *text);
extern void send_append_header2 (char *name, char *value, int mode);

extern int escape_shell (int argc, char **argv, compose_env_t *env);
extern int escape_command (int argc, char **argv, compose_env_t *env);
extern int escape_help (int argc, char **argv, compose_env_t *env);
extern int escape_sign (int argc, char **argv, compose_env_t *env);
extern int escape_bcc (int argc, char **argv, compose_env_t *env);
extern int escape_cc (int argc, char **argv, compose_env_t *env);
extern int escape_deadletter (int argc, char **argv, compose_env_t *env);
extern int escape_editor (int argc, char **argv, compose_env_t *env);
extern int escape_print (int argc, char **argv, compose_env_t *env);
extern int escape_headers (int argc, char **argv, compose_env_t *env);
extern int escape_insert (int argc, char **argv, compose_env_t *env);
extern int escape_quote (int argc, char **argv, compose_env_t *env);
extern int escape_type_input (int argc, char **argv, compose_env_t *env);
extern int escape_read (int argc, char **argv, compose_env_t *env);
extern int escape_subj (int argc, char **argv, compose_env_t *env);
extern int escape_to (int argc, char **argv, compose_env_t *env);
extern int escape_visual (int argc, char **argv, compose_env_t *env);
extern int escape_write (int argc, char **argv, compose_env_t *env);
extern int escape_exit (int argc, char **argv, compose_env_t *env);
extern int escape_pipe (int argc, char **argv, compose_env_t *env);

/* Cursor */
extern void set_cursor (unsigned value);
extern size_t get_cursor (void);
extern void page_invalidate (int hard);
extern void cond_page_invalidate (size_t value);
extern void page_do (msg_handler_t func, void *data);
extern size_t page_move (off_t offset);
extern int is_current_message (size_t n);

/* msgsets */
extern void msgset_free (msgset_t *msg_set);
extern msgset_t *msgset_make_1 (size_t number);
extern msgset_t *msgset_append (msgset_t *one, msgset_t *two);
extern msgset_t *msgset_range (int low, int high);
extern msgset_t *msgset_expand (msgset_t *set, msgset_t *expand_by);
extern msgset_t *msgset_dup (const msgset_t *set);
extern int msgset_parse (const int argc, char **argv,
			 int flags, msgset_t **mset);
extern int msgset_member (msgset_t *set, size_t n);
extern msgset_t *msgset_negate (msgset_t *set);
extern size_t msgset_count (msgset_t *set);


#define MDHINT_SELECTED_HEADERS 0x1

struct mime_descend_closure
{
  int hints;
  const msgset_t *msgset;
  mu_message_t message;
  const char *type;
  const char *encoding;
  const struct mime_descend_closure *parent;
};

typedef int (*mime_descend_fn) (struct mime_descend_closure *closure,
				void *data);

extern int mime_descend (struct mime_descend_closure *closure,
			 mime_descend_fn fun, void *data);



extern int util_do_command (const char *cmd, ...) MU_PRINTFLIKE(1,2);

extern int util_foreach_msg (int argc, char **argv, int flags,
			     msg_handler_t func, void *data);
extern size_t util_range_msg (size_t low, size_t high, int flags, 
			      msg_handler_t func, void *data);

extern function_t* util_command_get (const char *cmd);

extern void *util_find_entry (void *table, size_t nmemb, size_t size,
			      const char *cmd);
extern int util_help (void *table, size_t nmemb, size_t size, const char *word);
extern int util_command_list (void *table, size_t nmemb, size_t size);

extern const struct mail_command_entry *mail_find_command (const char *cmd);
extern const struct mail_escape_entry *mail_find_escape (const char *cmd);
extern int mail_command_help (const char *command);
extern int mail_escape_help (const char *command);
extern void mail_command_list (void);
extern const struct mail_command *mail_command_name (int i);

extern int util_getcols (void);
extern int util_getlines (void);
extern int util_screen_lines (void);
extern int util_screen_columns (void);
extern int util_get_crt (void);
extern struct mailvar_variable *mailvar_find_variable (const char *var, int create);
extern int mailvar_get (void *ptr, const char *variable,
			enum mailvar_type type, int warn);

extern void mailvar_print (int set);
extern void mailvar_variable_format (FILE *fp,
				     const struct mailvar_variable *,
				     const char *defval);

#define MOPTF_OVERWRITE 0x001
#define MOPTF_QUIET     0x002
#define MOPTF_UNSET     0x004
extern int mailvar_set (const char *name, void *value,
		      enum mailvar_type type, int flags);
extern int util_isdeleted (size_t msgno);
extern char *util_get_homedir (void);
extern char *util_fullpath (const char *inpath);
extern char *util_folder_path (const char *name);
extern char *util_get_sender (int msgno, int strip);

extern void util_slist_print (mu_list_t list, int nl);
extern int util_slist_lookup (mu_list_t list, const char *str);
extern void util_slist_add (mu_list_t *list, char *value);
extern void util_slist_remove (mu_list_t *list, char *value);
extern void util_slist_destroy (mu_list_t *list);
extern char *util_slist_to_string (mu_list_t list, const char *delim);
extern void util_strcat (char **dest, const char *str);
extern char *util_outfolder_name (char *str);
extern void util_save_outgoing (mu_message_t msg, char *savefile);
extern void util_error (const char *format, ...) MU_PRINTFLIKE(1,2);
extern int util_error_range (size_t msgno);
extern void util_noapp (void);
extern int util_tempfile (char **namep);
extern void util_msgset_iterate (msgset_t *msgset, 
                                 int (*fun) (mu_message_t, msgset_t *, void *), 
                                 void *closure);
extern int util_get_content_type (mu_header_t hdr, char **value, char **args);
extern int util_get_hdr_value (mu_header_t hdr, const char *name, char **value);
extern int util_merge_addresses (char **addr_str, const char *value);
extern int util_header_expand (mu_header_t *hdr);
extern int util_get_message (mu_mailbox_t mbox, size_t msgno, mu_message_t *msg);
void util_cache_command (mu_list_t *list, const char *fmt, ...) MU_PRINTFLIKE(2,3);
void util_run_cached_commands (mu_list_t *list);
const char *util_reply_prefix (void);
void util_rfc2047_decode (char **value);

void util_mark_read (mu_message_t msg);

const char *util_url_to_string (mu_url_t url);

size_t fprint_msgset (FILE *fp, const msgset_t *msgset);

int is_address_field (const char *name);

extern int ml_got_interrupt (void);
extern void ml_clear_interrupt (void);
extern void ml_readline_init (void);
extern int ml_reread (const char *prompt, char **text);
extern char *ml_readline (const char *prompt);
extern char *ml_readline_with_intr (const char *prompt);

extern char *alias_expand (const char *name);
extern void alias_destroy (const char *name);

typedef struct alias_iterator *alias_iterator_t;
extern char *alias_find_first (const char *prefix, alias_iterator_t *itr);
extern const char *alias_iterate_next (alias_iterator_t itr);
extern const char *alias_iterate_first (const char *p, alias_iterator_t *itr);
extern void alias_iterate_end (alias_iterator_t *itr);

extern int mail_sender    (int argc, char **argv);
extern int mail_nosender  (int argc, char **argv);
extern mu_address_t get_sender_address (mu_message_t msg);

#define COMPOSE_APPEND      0
#define COMPOSE_REPLACE     1
#define COMPOSE_SINGLE_LINE 2

void compose_init (compose_env_t *env);
int compose_header_set (compose_env_t *env, const char *name,
		        const char *value, int replace);
char *compose_header_get (compose_env_t *env, char *name, char *defval);
void compose_destroy (compose_env_t *env);

#ifndef HAVE_READLINE_READLINE_H
extern char *readline (char *prompt);
#endif

/* Flags for util_get_message */
#define MSG_ALL       0
#define MSG_NODELETED 0x0001
#define MSG_SILENT    0x0002
#define MSG_COUNT     0x0004

/* Message attributes */
#define MAIL_ATTRIBUTE_MBOXED   0x0001
#define MAIL_ATTRIBUTE_PRESERVED 0x0002
#define MAIL_ATTRIBUTE_SAVED    0x0004
#define MAIL_ATTRIBUTE_TAGGED   0x0008
#define MAIL_ATTRIBUTE_SHOWN    0x0010
#define MAIL_ATTRIBUTE_TOUCHED  0x0020

extern void ml_attempted_completion_over (void);

#ifdef WITH_READLINE
extern char **file_compl (int argc, char **argv, int ws);
extern char **no_compl (int argc, char **argv, int ws);
extern char **msglist_compl (int argc, char **argv, int ws);
extern char **msglist_file_compl (int argc, char **argv, int ws);
extern char **dir_compl (int argc, char **argv, int ws);
extern char **command_compl (int argc, char **argv, int ws);
extern char **alias_compl (int argc, char **argv, int ws);
extern char **mailvar_set_compl (int argc, char **argv, int ws);
extern char **exec_compl (int argc, char **argv, int ws);
#else
# define file_compl NULL
# define no_compl NULL
# define msglist_compl NULL
# define msglist_file_compl NULL
# define dir_compl NULL
# define command_compl NULL
# define alias_compl NULL
# define var_compl NULL
# define exec_compl NULL     
# define mailvar_set_compl NULL
#endif

#ifdef __cplusplus
}
#endif

#endif /* _MAIL_H */
