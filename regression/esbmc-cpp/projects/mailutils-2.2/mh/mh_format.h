/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2006, 2007, 2010 Free Software Foundation, Inc.

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

#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
#include <obstack.h>

typedef struct       /* A string object type */
{
  int size;          /* Allocated size or 0 for static storage */
  char *ptr;         /* Actual data */
}
strobj_t;

#define strobj_ptr(p) ((p)->ptr ? (p)->ptr : "")
#define strobj_len(p) (strobj_is_null(p) ? 0 : strlen((p)->ptr))
#define strobj_is_null(p) ((p)->ptr == NULL)
#define strobj_is_static(p) ((p)->size == 0)

struct mh_machine
{
  strobj_t reg_str;         /* String register */
  int reg_num;              /* Numeric register */

  strobj_t arg_str;         /* String argument */
  long arg_num;             /* Numeric argument */
  
  size_t pc;                /* Program counter */
  size_t progsize;          /* Size of allocated program*/
  mh_instr_t *prog;         /* Program itself */
  int stop;                 /* Stop execution immediately */

  struct obstack stk;       /* Output buffer */
  size_t width;             /* Output buffer width */
  size_t ind;               /* Output buffer index */

  mu_list_t addrlist;       /* The list of email addresses output this far */
  int fmtflags;             /* Current formatting flags */
  
  mu_message_t message;     /* Current message */
  size_t msgno;             /* Its number */
};

void strobj_free (strobj_t *obj);
void strobj_create (strobj_t *lvalue, const char *str);
void strobj_set (strobj_t *lvalue, char *str);
void strobj_assign (strobj_t *lvalue, strobj_t *rvalue);
void strobj_copy (strobj_t *lvalue, strobj_t *rvalue);
void strobj_realloc (strobj_t *obj, size_t length);
