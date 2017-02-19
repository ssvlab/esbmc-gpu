/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2006, 2007, 2008, 2010
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif  

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sieve-priv.h>

void *
mu_sieve_alloc (size_t size)
{
  void *p = malloc (size);
  if (!p)
    {
      mu_error ("not enough memory");
      abort ();
    }
  return p;
}

void *
mu_sieve_palloc (mu_list_t *pool, size_t size)
{
  void *p = malloc (size);
  if (p)
    {
      if (!*pool && mu_list_create (pool))
	{
	  free (p);
	  return NULL;
	}
      mu_list_append (*pool, p);
    }
  return p;
}

char *
mu_sieve_pstrdup (mu_list_t *pool, const char *str)
{
  size_t len;
  char *p;
  
  if (!str)
    return NULL;
  len = strlen (str);
  p = mu_sieve_palloc (pool, len + 1);
  if (p)
    {
      memcpy (p, str, len);
      p[len] = 0;
    }
  return p;
}

void *
mu_sieve_prealloc (mu_list_t *pool, void *ptr, size_t size)
{
  void *newptr;
  
  if (*pool)
    mu_list_remove (*pool, ptr);

  newptr = realloc (ptr, size);
  if (newptr)
    {
      if (!*pool && mu_list_create (pool))
	{
	  free (newptr);
	  return NULL;
	}
      mu_list_append (*pool, newptr);
    }
  return newptr;
}

void
mu_sieve_pfree (mu_list_t *pool, void *ptr)
{
  if (*pool)
    mu_list_remove (*pool, ptr);
  free (ptr);
}  

void *
mu_sieve_malloc (mu_sieve_machine_t mach, size_t size)
{
  return mu_sieve_palloc (&mach->memory_pool, size);
}

char *
mu_sieve_mstrdup (mu_sieve_machine_t mach, const char *str)
{
  return mu_sieve_pstrdup (&mach->memory_pool, str);
}

void *
mu_sieve_mrealloc (mu_sieve_machine_t mach, void *ptr, size_t size)
{
  return mu_sieve_prealloc (&mach->memory_pool, ptr, size);
}

void
mu_sieve_mfree (mu_sieve_machine_t mach, void *ptr)
{
  mu_sieve_pfree (&mach->memory_pool, ptr);
}  

static int
_destroy_item (void *item, void *data)
{
  free (item);
  return 0;
}

void
mu_sieve_slist_destroy (mu_list_t *plist)
{
  if (!plist)
    return;
  mu_list_do (*plist, _destroy_item, NULL);
  mu_list_destroy (plist);
}

mu_sieve_value_t *
mu_sieve_value_create (mu_sieve_data_type type, void *data)
{
  mu_sieve_value_t *val = mu_sieve_alloc (sizeof (*val));

  val->type = type;
  switch (type)
    {
    case SVT_NUMBER:
      val->v.number = * (long *) data;
      break;
      
    case SVT_STRING:
      val->v.string = data;
      break;
      
    case SVT_VALUE_LIST:
    case SVT_STRING_LIST:
      val->v.list = data;
      break;
      
    case SVT_TAG:
    case SVT_IDENT:
      val->v.string = data;
      break;

    case SVT_POINTER:
      val->v.ptr = data;
      break;
	
    default:
      mu_sv_compile_error (&mu_sieve_locus, _("invalid data type"));
      abort ();
    }
  return val;
}

mu_sieve_value_t *
mu_sieve_value_get (mu_list_t vlist, size_t index)
{
  mu_sieve_value_t *val = NULL;
  mu_list_get (vlist, index, (void **)&val);
  return val;
}

void
mu_sv_compile_error (mu_sieve_locus_t *ploc, const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  mu_sieve_error_count++;
  mu_sieve_machine->parse_error_printer (mu_sieve_machine->data,
				         ploc->source_file,
				         ploc->source_line,
                                         fmt, ap);
  va_end (ap);
}

void
mu_sieve_error (mu_sieve_machine_t mach, const char *fmt, ...)
{
  va_list ap;
  
  va_start (ap, fmt);
  if (mach->identifier)
    {
      char *new_fmt = malloc (strlen (mach->identifier) + 2 +
			      strlen (fmt) + 1);
      if (new_fmt)
	{
	  strcpy (new_fmt, mach->identifier);
	  strcat (new_fmt, ": ");
	  strcat (new_fmt, fmt);
	  mach->error_printer (mach->data, new_fmt, ap);
	  free (new_fmt);
	}
      else
	mach->error_printer (mach->data, fmt, ap);
    }
  else
    mach->error_printer (mach->data, fmt, ap);
  va_end (ap);
}

void
mu_sieve_arg_error (mu_sieve_machine_t mach, int n)
{
  mu_sieve_error (mach, _("cannot retrieve argument %d"), n);
}

static void sieve_debug_internal (mu_sieve_printf_t printer, void *data,
				  const char *fmt, ...) MU_PRINTFLIKE(3,4);

static void
sieve_debug_internal (mu_sieve_printf_t printer, void *data,
                      const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  printer (data, fmt, ap);
  va_end (ap);
}

void
mu_sieve_debug (mu_sieve_machine_t mach, const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  mach->debug_printer (mach->data, fmt, ap);
  va_end (ap);
}

void
mu_sieve_log_action (mu_sieve_machine_t mach, const char *action,
		     const char *fmt, ...)
{
  va_list ap;

  if (!mach->logger)
    return;
  va_start (ap, fmt);
  mach->logger (mach->data, &mach->locus, mach->msgno, mach->msg,
		action, fmt, ap);
  va_end (ap);
}
  
const char *
mu_sieve_type_str (mu_sieve_data_type type)
{
  switch (type)
    {
    case SVT_VOID:
      return "void";
      
    case SVT_NUMBER:
      return "number";
      
    case SVT_STRING:
      return "string";

    case SVT_STRING_LIST:
      return "string-list";
      
    case SVT_TAG:
      return "tag";

    case SVT_IDENT:
      return "ident";

    case SVT_VALUE_LIST:
      return "value-list";

    case SVT_POINTER:
      return "pointer";
    }

  return "unknown";
}

struct debug_data {
  mu_sieve_printf_t printer;
  void *data;
};

static int
string_printer (char *s, struct debug_data *dbg)
{
  sieve_debug_internal (dbg->printer, dbg->data, "\"%s\" ", s);
  return 0;
}

static void sieve_print_value (mu_sieve_value_t *, mu_sieve_printf_t,
			       void *);

static int
value_printer (mu_sieve_value_t *val, struct debug_data *dbg)
{
  sieve_print_value (val, dbg->printer, dbg->data);
  sieve_debug_internal (dbg->printer, dbg->data, " ");
  return 0;
}

static void
sieve_print_value (mu_sieve_value_t *val, mu_sieve_printf_t printer,
		   void *data)
{
  struct debug_data dbg;

  dbg.printer = printer;
  dbg.data = data;

  sieve_debug_internal (printer, data, "%s(", mu_sieve_type_str (val->type));
  switch (val->type)
    {
    case SVT_VOID:
      break;
      
    case SVT_NUMBER:
      sieve_debug_internal (printer, data, "%lu",
			    (unsigned long) val->v.number);
      break;
      
    case SVT_TAG:
    case SVT_IDENT:
    case SVT_STRING:
      sieve_debug_internal (printer, data, "%s", val->v.string);
      break;
      
    case SVT_STRING_LIST:
      mu_list_do (val->v.list, (mu_list_action_t*) string_printer, &dbg);
      break;

    case SVT_VALUE_LIST:
      mu_list_do (val->v.list, (mu_list_action_t*) value_printer, &dbg);

    case SVT_POINTER:
      sieve_debug_internal (printer, data, "%p", val->v.ptr);
    }
  sieve_debug_internal (printer, data, ")");
} 

void
mu_sv_print_value_list (mu_list_t list, mu_sieve_printf_t printer, void *data)
{
  mu_sieve_value_t val;
  
  val.type = SVT_VALUE_LIST;
  val.v.list = list;
  sieve_print_value (&val, printer, data);
}

static int
tag_printer (mu_sieve_runtime_tag_t *val, struct debug_data *dbg)
{
  sieve_debug_internal (dbg->printer, dbg->data, "%s", val->tag);
  if (val->arg)
    {
      sieve_debug_internal (dbg->printer, dbg->data, "(");
      sieve_print_value (val->arg, dbg->printer, dbg->data);
      sieve_debug_internal (dbg->printer, dbg->data, ")");
    }
  sieve_debug_internal (dbg->printer, dbg->data, " ");
  return 0;
}

void
mu_sv_print_tag_list (mu_list_t list, mu_sieve_printf_t printer, void *data)
{
  struct debug_data dbg;

  dbg.printer = printer;
  dbg.data = data;
  mu_list_do (list, (mu_list_action_t*) tag_printer, &dbg);
}

static int
tag_finder (void *item, void *data)
{
  mu_sieve_runtime_tag_t *val = item;
  mu_sieve_runtime_tag_t *target = data;

  if (strcmp (val->tag, target->tag) == 0)
    {
      target->arg = val->arg;
      return 1;
    }
  return 0;
}

int
mu_sieve_tag_lookup (mu_list_t taglist, char *name, mu_sieve_value_t **arg)
{
  mu_sieve_runtime_tag_t t;

  t.tag = name;
  if (taglist && mu_list_do (taglist, tag_finder, &t))
    {
      if (arg)
	*arg = t.arg;
      return 1;
    }
  return 0;
}

int
mu_sieve_vlist_do (mu_sieve_value_t *val, mu_list_action_t *ac, void *data)
{
  switch (val->type)
    {
    case SVT_VALUE_LIST:
    case SVT_STRING_LIST:
      return mu_list_do (val->v.list, ac, data);
      
    default:
      return -1;
    }
}

struct comp_data {
  mu_sieve_value_t *val;
  mu_sieve_comparator_t comp;
  mu_sieve_relcmp_t test;
  mu_sieve_retrieve_t retr;
  void *data;
  size_t count;
};

struct comp_data2 {
  char *sample;
  mu_sieve_comparator_t comp;
  mu_sieve_relcmp_t test;
};

int
_comp_action2 (void *item, void *data)
{
  struct comp_data2 *cp = data;
  return cp->test (cp->comp (item, cp->sample), 0);
}

int
_comp_action (void *item, void *data)
{
  struct comp_data *cp = data;
  struct comp_data2 d;
  int rc = 0;
  int i;

  d.comp = cp->comp;
  d.test = cp->test;
  for (i = 0; rc == 0 && cp->retr (item, cp->data, i, &d.sample) == 0; i++)
    if (d.sample)
      {
	cp->count++;
        rc = mu_sieve_vlist_do (cp->val, _comp_action2, &d);
        free (d.sample);
      }
  return rc;
}

int
mu_sieve_vlist_compare (mu_sieve_value_t *a, mu_sieve_value_t *b,
		     mu_sieve_comparator_t comp, mu_sieve_relcmp_t test,
		     mu_sieve_retrieve_t retr,
		     void *data, size_t *count)
{
  struct comp_data d;
  int rc;
  
  d.comp = comp;
  d.test = test;
  d.retr = retr;
  d.data = data;
  d.val = b;
  d.count = 0;
  rc = mu_sieve_vlist_do (a, _comp_action, &d);
  if (count)
    *count = d.count;
  return rc;
}
