/* cfg.h -- general-purpose configuration file parser 
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 3, or (at
   your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _MAILUTILS_CFG_H
#define _MAILUTILS_CFG_H

#include <mailutils/list.h>
#include <mailutils/debug.h>
#include <mailutils/opool.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#ifdef __cplusplus
extern "C" {
#endif  

typedef enum mu_cfg_node_type mu_cfg_node_type_t;
typedef struct mu_cfg_node mu_cfg_node_t;
typedef struct mu_cfg_locus mu_cfg_locus_t;
typedef struct mu_cfg_tree mu_cfg_tree_t;

#define MU_CFG_STRING 0
#define MU_CFG_LIST   1
#define MU_CFG_ARRAY  2

typedef struct mu_config_value mu_config_value_t;

struct mu_config_value   
{
  int type;
  union
  {
    mu_list_t list;
    const char *string;
    struct
    {
      size_t c;
      mu_config_value_t *v;
    } arg;
  } v;
};
  
enum mu_cfg_node_type
  {
    mu_cfg_node_undefined,
    mu_cfg_node_statement,
    mu_cfg_node_param
  };

struct mu_cfg_locus
{
  char *file;
  size_t line;
};

struct mu_cfg_node
{
  mu_cfg_locus_t locus;
  enum mu_cfg_node_type type;
  char *tag;
  mu_config_value_t *label;
  mu_list_t nodes;   /* a list of mu_cfg_node_t */
};

struct mu_cfg_tree
{
  mu_list_t nodes;   /* a list of mu_cfg_node_t */
  mu_debug_t debug;
  mu_opool_t pool;
};

int mu_cfg_parse (mu_cfg_tree_t **ptree);
int mu_cfg_tree_union (mu_cfg_tree_t **pa, mu_cfg_tree_t **pb);
int mu_cfg_tree_postprocess (mu_cfg_tree_t *tree, int flags);

extern mu_cfg_locus_t mu_cfg_locus;

mu_opool_t mu_cfg_lexer_pool (void);

void mu_cfg_vperror (mu_debug_t, const mu_cfg_locus_t *,
		     const char *fmt, va_list ap);
void mu_cfg_perror (mu_debug_t debug, const mu_cfg_locus_t *,
		    const char *, ...) MU_PRINTFLIKE(3,4);
void mu_cfg_parse_error (const char *, ...) MU_PRINTFLIKE(1,2);
void mu_cfg_format_error (mu_debug_t debug, size_t, const char *fmt, ...)
      MU_PRINTFLIKE(3,4);

#define MU_CFG_ITER_OK   0
#define MU_CFG_ITER_SKIP 1
#define MU_CFG_ITER_STOP 2

typedef int (*mu_cfg_iter_func_t) (const mu_cfg_node_t *node, void *data);

struct mu_cfg_iter_closure
{
  mu_cfg_iter_func_t beg;
  mu_cfg_iter_func_t end;
  void *data;
};

void mu_cfg_destroy_tree (mu_cfg_tree_t **tree);

int mu_cfg_preorder (mu_list_t nodelist, struct mu_cfg_iter_closure *);


/* Table-driven parsing */
enum mu_cfg_param_data_type
  {
    mu_cfg_string,
    mu_cfg_short,
    mu_cfg_ushort,
    mu_cfg_int,
    mu_cfg_uint,
    mu_cfg_long,
    mu_cfg_ulong,
    mu_cfg_size,
    mu_cfg_off,
    mu_cfg_time,
    mu_cfg_bool,
    mu_cfg_ipv4,
    mu_cfg_cidr,
    mu_cfg_host,
    mu_cfg_callback,
    mu_cfg_section
  };

#define MU_CFG_LIST_MASK 0x8000
#define MU_CFG_LIST_OF(t) ((t) | MU_CFG_LIST_MASK)
#define MU_CFG_TYPE(t) ((t) & ~MU_CFG_LIST_MASK)
#define MU_CFG_IS_LIST(t) ((t) & MU_CFG_LIST_MASK)
  
typedef int (*mu_cfg_callback_t) (mu_debug_t, void *, mu_config_value_t *);

struct mu_cfg_param
{
  const char *ident;
  enum mu_cfg_param_data_type type;
  void *data;
  size_t offset;
  mu_cfg_callback_t callback;
  const char *docstring;
  const char *argname;
};

#define MU_TARGET_REF(f) &f, 0
#define MU_TARGET_OFF(s,f) NULL, mu_offsetof(s,f)

enum mu_cfg_section_stage
  {
    mu_cfg_section_start,
    mu_cfg_section_end
  };

typedef int (*mu_cfg_section_fp) (enum mu_cfg_section_stage stage,
				  const mu_cfg_node_t *node,
				  const char *label,
				  void **section_data_ptr,
				  void *call_data,
				  mu_cfg_tree_t *tree);

struct mu_cfg_section
{
  const char *ident;
  char *label;
  mu_cfg_section_fp parser;
  void *target;
  size_t offset;
  mu_list_t /* of mu_cfg_cont */ children;
  char *docstring;
};

enum mu_cfg_cont_type
  {
    mu_cfg_cont_section,
    mu_cfg_cont_param
  };

struct mu_cfg_cont
{
  enum mu_cfg_cont_type type;
  mu_refcount_t refcount;
  union
  {
    const char *ident;
    struct mu_cfg_section section;
    struct mu_cfg_param param;
  } v;
};

typedef struct mu_cfg_cidr mu_cfg_cidr_t;

struct mu_cfg_cidr
{
  struct in_addr addr;
  unsigned long mask;
};

#define MU_CFG_PATH_DELIM '.'
#define MU_CFG_PATH_DELIM_STR "."
  
int mu_config_create_container (struct mu_cfg_cont **pcont,
				enum mu_cfg_cont_type type);
int mu_config_clone_container (struct mu_cfg_cont *cont);
void mu_config_destroy_container (struct mu_cfg_cont **pcont);

int mu_cfg_section_add_container (struct mu_cfg_section *sect,
				  struct mu_cfg_cont *cont);
int mu_cfg_section_add_params (struct mu_cfg_section *sect,
			       struct mu_cfg_param *param);


int mu_create_canned_section (char *name, struct mu_cfg_section **psection);
int mu_create_canned_param (char *name, struct mu_cfg_param **pparam);
struct mu_cfg_cont *mu_get_canned_container (const char *name);

int mu_cfg_create_node_list (mu_list_t *plist);
  
int mu_cfg_scan_tree (mu_cfg_tree_t *tree, struct mu_cfg_section *sections,
		      void *target, void *call_data);

int mu_cfg_find_section (struct mu_cfg_section *root_sec,
			 const char *path, struct mu_cfg_section **retval);

int mu_config_register_section (const char *parent_path,
				const char *ident,
				const char *label,
				mu_cfg_section_fp parser,
				struct mu_cfg_param *param);
int mu_config_register_plain_section (const char *parent_path,
				      const char *ident,
				      struct mu_cfg_param *params);

mu_debug_t mu_cfg_get_debug (void);

#define MU_PARSE_CONFIG_GLOBAL  0x1
#define MU_PARSE_CONFIG_VERBOSE 0x2
#define MU_PARSE_CONFIG_DUMP    0x4
#define MU_PARSE_CONFIG_PLAIN   0x8

#ifdef MU_CFG_COMPATIBILITY
# define MU_CFG_DEPRECATED
#else
# define MU_CFG_DEPRECATED __attribute__ ((deprecated))
#endif

int mu_parse_config (const char *file, const char *progname,
		     struct mu_cfg_param *progparam, int flags,
		     void *target_ptr) MU_CFG_DEPRECATED;

int mu_cfg_parse_boolean (const char *str, int *res);

extern int mu_cfg_parser_verbose;
extern size_t mu_cfg_error_count;

#define MU_CFG_FMT_LOCUS 0x01
  
void mu_cfg_format_docstring (mu_stream_t stream, const char *docstring,
			      int level);
void mu_cfg_format_parse_tree (mu_stream_t stream, struct mu_cfg_tree *tree,
			       int flags);
void mu_cfg_format_node (mu_stream_t stream, const mu_cfg_node_t *node,
			 int flags);
  
void mu_cfg_format_container (mu_stream_t stream, struct mu_cfg_cont *cont);
void mu_format_config_tree (mu_stream_t stream, const char *progname,
			    struct mu_cfg_param *progparam, int flags);
int mu_cfg_tree_reduce (mu_cfg_tree_t *parse_tree, const char *progname,
		        struct mu_cfg_param *progparam,
		        int flags, void *target_ptr);
int mu_cfg_assert_value_type (mu_config_value_t *val, int type,
			      mu_debug_t debug);
int mu_cfg_string_value_cb (mu_debug_t debug, mu_config_value_t *val,
			    int (*fun) (mu_debug_t, const char *, void *),
			    void *data);

int mu_cfg_parse_file (mu_cfg_tree_t **return_tree, const char *file,
		       int flags);
  
  
int mu_get_config (const char *file, const char *progname,
		   struct mu_cfg_param *progparam, int flags,
		   void *target_ptr) MU_CFG_DEPRECATED;

int mu_cfg_tree_create (struct mu_cfg_tree **ptree);
void mu_cfg_tree_set_debug (struct mu_cfg_tree *tree, mu_debug_t debug);
mu_cfg_node_t *mu_cfg_tree_create_node (struct mu_cfg_tree *tree,
					enum mu_cfg_node_type type,
					const mu_cfg_locus_t *loc,
					const char *tag,
					const char *label,
					mu_list_t nodelist);
void mu_cfg_tree_add_node (mu_cfg_tree_t *tree, mu_cfg_node_t *node);
void mu_cfg_tree_add_nodelist (mu_cfg_tree_t *tree, mu_list_t nodelist);

int mu_cfg_find_node (mu_cfg_tree_t *tree, const char *path,
		      mu_cfg_node_t **pnode);
int mu_cfg_create_subtree (const char *path, mu_cfg_node_t **pnode);

#ifdef __cplusplus
}
#endif

#endif
