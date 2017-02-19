/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2006, 2007, 2008, 2009,
   2010 Free Software Foundation, Inc.

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

#ifndef _MAILUTILS_LIBSIEVE_H
#define _MAILUTILS_LIBSIEVE_H

#include <sys/types.h>
#include <stdarg.h>
#include <mailutils/mailutils.h>

#ifdef __cplusplus
extern "C" {
#endif

#define __s_cat3__(a,b,c) a ## b ## c
#define SIEVE_EXPORT(module,name) __s_cat3__(module,_LTX_,name)

typedef struct mu_sieve_machine *mu_sieve_machine_t;

typedef struct
{
  char *source_file;
  size_t source_line;
}
mu_sieve_locus_t;

typedef int (*mu_sieve_handler_t) (mu_sieve_machine_t mach,
				   mu_list_t args, mu_list_t tags);
typedef int (*mu_sieve_printf_t) (void *data, const char *fmt, va_list ap);
typedef int (*mu_sieve_parse_error_t) (void *data,
				       const char *filename, int lineno,
				       const char *fmt, va_list ap);
typedef void (*mu_sieve_action_log_t) (void *data,
				       const mu_sieve_locus_t * locus,
				       size_t msgno, mu_message_t msg,
				       const char *action,
				       const char *fmt, va_list ap);

typedef int (*mu_sieve_relcmp_t) (int, int);
typedef int (*mu_sieve_relcmpn_t) (size_t, size_t);
typedef int (*mu_sieve_comparator_t) (const char *, const char *);
typedef int (*mu_sieve_retrieve_t) (void *item, void *data, int idx,
				    char **pval);
typedef void (*mu_sieve_destructor_t) (void *data);
typedef int (*mu_sieve_tag_checker_t) (const char *name,
				       mu_list_t tags, mu_list_t args);

typedef enum
{
  SVT_VOID,
  SVT_NUMBER,
  SVT_STRING,
  SVT_STRING_LIST,
  SVT_TAG,
  SVT_IDENT,
  SVT_VALUE_LIST,
  SVT_POINTER
}
mu_sieve_data_type;

typedef struct mu_sieve_runtime_tag mu_sieve_runtime_tag_t;

typedef struct
{
  mu_sieve_data_type type;
  union
  {
    char *string;
    size_t number;
    mu_list_t list;
    mu_sieve_runtime_tag_t *tag;
    void *ptr;
  } v;
} mu_sieve_value_t;

typedef struct
{
  char *name;
  mu_sieve_data_type argtype;
} mu_sieve_tag_def_t;

typedef struct
{
  mu_sieve_tag_def_t *tags;
  mu_sieve_tag_checker_t checker;
} mu_sieve_tag_group_t;

struct mu_sieve_runtime_tag
{
  char *tag;
  mu_sieve_value_t *arg;
};

typedef struct
{
  const char *name;
  int required;
  mu_sieve_handler_t handler;
  mu_sieve_data_type *req_args;
  mu_sieve_tag_group_t *tags;
} mu_sieve_register_t;

#define MU_SIEVE_CHARSET "UTF-8"

#define MU_SIEVE_MATCH_IS        1
#define MU_SIEVE_MATCH_CONTAINS  2
#define MU_SIEVE_MATCH_MATCHES   3
#define MU_SIEVE_MATCH_REGEX     4
#define MU_SIEVE_MATCH_EQ        5
#define MU_SIEVE_MATCH_LAST      6

/* Debugging levels */
#define MU_SIEVE_DEBUG_TRACE  0x0001
#define MU_SIEVE_DEBUG_INSTR  0x0002
#define MU_SIEVE_DEBUG_DISAS  0x0004
#define MU_SIEVE_DRY_RUN      0x0008

extern int mu_sieve_yydebug;
extern mu_list_t mu_sieve_include_path;
extern mu_list_t mu_sieve_library_path;

/* Memory allocation functions */
void *mu_sieve_alloc (size_t size);
void *mu_sieve_palloc (mu_list_t * pool, size_t size);
void *mu_sieve_prealloc (mu_list_t * pool, void *ptr, size_t size);
void mu_sieve_pfree (mu_list_t * pool, void *ptr);
char *mu_sieve_pstrdup (mu_list_t * pool, const char *str);

void *mu_sieve_malloc (mu_sieve_machine_t mach, size_t size);
char *mu_sieve_mstrdup (mu_sieve_machine_t mach, const char *str);
void *mu_sieve_mrealloc (mu_sieve_machine_t mach, void *ptr, size_t size);
void mu_sieve_mfree (mu_sieve_machine_t mach, void *ptr);

mu_sieve_value_t *mu_sieve_value_create (mu_sieve_data_type type, void *data);
void mu_sieve_slist_destroy (mu_list_t * plist);

/* Symbol space functions */
mu_sieve_register_t *mu_sieve_test_lookup (mu_sieve_machine_t mach,
					   const char *name);
mu_sieve_register_t *mu_sieve_action_lookup (mu_sieve_machine_t mach,
					     const char *name);
int mu_sieve_register_test (mu_sieve_machine_t mach,
			    const char *name, mu_sieve_handler_t handler,
			    mu_sieve_data_type * arg_types,
			    mu_sieve_tag_group_t * tags, int required);
int mu_sieve_register_action (mu_sieve_machine_t mach,
			      const char *name, mu_sieve_handler_t handler,
			      mu_sieve_data_type * arg_types,
			      mu_sieve_tag_group_t * tags, int required);
int mu_sieve_register_comparator (mu_sieve_machine_t mach, const char *name,
				  int required, mu_sieve_comparator_t is,
				  mu_sieve_comparator_t contains,
				  mu_sieve_comparator_t matches,
				  mu_sieve_comparator_t regex,
				  mu_sieve_comparator_t eq);
int mu_sieve_require_action (mu_sieve_machine_t mach, const char *name);
int mu_sieve_require_test (mu_sieve_machine_t mach, const char *name);
int mu_sieve_require_comparator (mu_sieve_machine_t mach, const char *name);
int mu_sieve_require_relational (mu_sieve_machine_t mach, const char *name);

mu_sieve_comparator_t mu_sieve_comparator_lookup (mu_sieve_machine_t mach,
						  const char *name,
						  int matchtype);

mu_sieve_comparator_t mu_sieve_get_comparator (mu_sieve_machine_t mach,
					       mu_list_t tags);
int mu_sieve_str_to_relcmp (const char *str, mu_sieve_relcmp_t * test,
			    mu_sieve_relcmpn_t * stest);
mu_sieve_relcmp_t mu_sieve_get_relcmp (mu_sieve_machine_t mach,
				       mu_list_t tags);

void mu_sieve_require (mu_list_t slist);
int mu_sieve_tag_lookup (mu_list_t taglist, char *name,
			 mu_sieve_value_t ** arg);
int mu_sieve_load_ext (mu_sieve_machine_t mach, const char *name);
int mu_sieve_match_part_checker (const char *name, mu_list_t tags,
				 mu_list_t args);
int mu_sieve_match_part_checker (const char *name, mu_list_t tags,
				 mu_list_t args);
/* Operations in value lists */
mu_sieve_value_t *mu_sieve_value_get (mu_list_t vlist, size_t index);
int mu_sieve_vlist_do (mu_sieve_value_t * val, mu_list_action_t * ac,
		       void *data);
int mu_sieve_vlist_compare (mu_sieve_value_t * a, mu_sieve_value_t * b,
			    mu_sieve_comparator_t comp,
			    mu_sieve_relcmp_t test, mu_sieve_retrieve_t ac,
			    void *data, size_t * count);

/* Functions to create and destroy sieve machine */
int mu_sieve_machine_init (mu_sieve_machine_t * mach, void *data);
int mu_sieve_machine_dup (mu_sieve_machine_t const in,
			  mu_sieve_machine_t *out);
int mu_sieve_machine_inherit (mu_sieve_machine_t const in,
			      mu_sieve_machine_t *out);
void mu_sieve_machine_destroy (mu_sieve_machine_t * pmach);
int mu_sieve_machine_add_destructor (mu_sieve_machine_t mach,
				     mu_sieve_destructor_t destr, void *ptr);

/* Functions for accessing sieve machine internals */
void *mu_sieve_get_data (mu_sieve_machine_t mach);
mu_message_t mu_sieve_get_message (mu_sieve_machine_t mach);
size_t mu_sieve_get_message_num (mu_sieve_machine_t mach);
int mu_sieve_get_debug_level (mu_sieve_machine_t mach);
mu_mailer_t mu_sieve_get_mailer (mu_sieve_machine_t mach);
int mu_sieve_get_locus (mu_sieve_machine_t mach, mu_sieve_locus_t *);
char *mu_sieve_get_daemon_email (mu_sieve_machine_t mach);
const char *mu_sieve_get_identifier (mu_sieve_machine_t mach);

void mu_sieve_set_error (mu_sieve_machine_t mach,
			 mu_sieve_printf_t error_printer);
void mu_sieve_set_parse_error (mu_sieve_machine_t mach,
			       mu_sieve_parse_error_t p);
void mu_sieve_set_debug (mu_sieve_machine_t mach, mu_sieve_printf_t debug);
void mu_sieve_set_debug_object (mu_sieve_machine_t mach, mu_debug_t dbg);
void mu_sieve_set_debug_level (mu_sieve_machine_t mach, int level);
void mu_sieve_set_logger (mu_sieve_machine_t mach,
			  mu_sieve_action_log_t logger);
void mu_sieve_set_mailer (mu_sieve_machine_t mach, mu_mailer_t mailer);
void mu_sieve_set_daemon_email (mu_sieve_machine_t mach, const char *email);

int mu_sieve_get_message_sender (mu_message_t msg, char **ptext);

/* Logging and diagnostic functions */

void mu_sieve_error (mu_sieve_machine_t mach, const char *fmt, ...) 
                     MU_PRINTFLIKE(2,3);
void mu_sieve_debug (mu_sieve_machine_t mach, const char *fmt, ...)
                     MU_PRINTFLIKE(2,3);
void mu_sieve_log_action (mu_sieve_machine_t mach, const char *action,
			  const char *fmt, ...)
			  MU_PRINTFLIKE(3,4);
void mu_sieve_abort (mu_sieve_machine_t mach);
void mu_sieve_arg_error (mu_sieve_machine_t mach, int n);

int mu_sieve_is_dry_run (mu_sieve_machine_t mach);
const char *mu_sieve_type_str (mu_sieve_data_type type);

/* Principal entry points */

int mu_sieve_compile (mu_sieve_machine_t mach, const char *name);
int mu_sieve_compile_buffer (mu_sieve_machine_t mach,
			     const char *buf, int bufsize,
			     const char *fname, int line);
int mu_sieve_mailbox (mu_sieve_machine_t mach, mu_mailbox_t mbox);
int mu_sieve_message (mu_sieve_machine_t mach, mu_message_t message);
int mu_sieve_disass (mu_sieve_machine_t mach);

/* Configuration functions */

#define MU_SIEVE_CLEAR_INCLUDE_PATH 0x1
#define MU_SIEVE_CLEAR_LIBRARY_PATH 0x2

struct mu_gocs_sieve
{
  int clearflags;
  mu_list_t include_path;
  mu_list_t library_path;
};

int mu_sieve_module_init (enum mu_gocs_op, void *);

#ifdef __cplusplus
}
#endif

#endif
