/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

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
#include <assert.h>
#include <sieve-priv.h>

int
mu_sv_code (sieve_op_t *op)
{
  if (mu_sieve_machine->pc >= mu_sieve_machine->progsize)
    {
      size_t newsize = mu_sieve_machine->progsize + SIEVE_CODE_INCR;
      sieve_op_t *newprog = mu_sieve_mrealloc (mu_sieve_machine, 
                                               mu_sieve_machine->prog,
					       newsize *
					     sizeof mu_sieve_machine->prog[0]);
      if (!newprog)
	{
	  mu_sv_compile_error (&mu_sieve_locus, _("not enough memory"));
	  return 1;
	}
      mu_sieve_machine->prog = newprog;
      mu_sieve_machine->progsize = newsize;
    }
  mu_sieve_machine->prog[mu_sieve_machine->pc++] = *op;
  return 0;
}

int
mu_sv_code_instr (sieve_instr_t instr)
{
  sieve_op_t op;

  op.instr = instr;
  return mu_sv_code (&op);
}

int
mu_sv_code_handler (mu_sieve_handler_t handler)
{
  sieve_op_t op;

  op.handler = handler;
  return mu_sv_code (&op);
}

int
mu_sv_code_list (mu_list_t list)
{
  sieve_op_t op;

  op.list = list;
  return mu_sv_code (&op);
}

int
mu_sv_code_number (long num)
{
  sieve_op_t op;

  op.number = num;
  return mu_sv_code (&op);
}

int
mu_sv_code_string (const char *string)
{
  sieve_op_t op;

  op.string = string;
  return mu_sv_code (&op);
}

mu_sieve_tag_def_t *
find_tag (mu_sieve_tag_group_t *taglist, char *tagname,
	  mu_sieve_tag_checker_t *checker)
{
  *checker = NULL;
  
  if (!taglist)
    return NULL;
  
  for (; taglist->tags; taglist++)
    {
      mu_sieve_tag_def_t *def;
      for (def = taglist->tags; def->name; def++)
	if (strcmp (def->name, tagname) == 0)
	  {
	    *checker = taglist->checker;
	    return def;
	  }
    }
  return NULL;
}

static int
_compare_ptr (void *item, void *data)
{
  return item == data;
}

struct check_arg {
  const char *name;
  mu_list_t args;
  mu_list_t tags;
};

static int
_run_checker (void *item, void *data)
{
  struct check_arg *arg = data;
  return (*(mu_sieve_tag_checker_t)item) (arg->name, arg->tags, arg->args);
}

int
mu_sv_code_command (mu_sieve_register_t *reg, mu_list_t arglist)
{
  mu_iterator_t itr;
  mu_list_t arg_list = NULL;
  mu_list_t tag_list = NULL;
  mu_list_t chk_list = NULL;
  mu_sieve_data_type *exp_arg;
  int rc, err = 0;
  static mu_sieve_data_type empty[] = { SVT_VOID };
  
  if (mu_sv_code_handler (reg->handler))
    return 1;

  exp_arg = reg->req_args ? reg->req_args : empty;

  if (arglist)
    {
      rc = mu_list_get_iterator (arglist, &itr);

      if (rc)
	{
	  mu_sv_compile_error (&mu_sieve_locus, 
                               _("cannot create iterator: %s"),
  		               mu_strerror (rc));
	  return 1;
	}
  
      for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
	{
	  mu_sieve_value_t *val;
	  mu_sieve_runtime_tag_t tagrec, *tagptr;
	  
	  mu_iterator_current (itr, (void **)&val);
	  
	  if (val->type == SVT_TAG)
	    {
	      mu_sieve_tag_checker_t cf;
	      mu_sieve_tag_def_t *tag = find_tag (reg->tags, val->v.string, &cf);
	      if (!tag)
		{
		  mu_sv_compile_error (&mu_sieve_locus, 
				       _("invalid tag name `%s' for `%s'"),
				       val->v.string, reg->name);
		  err = 1;
		  break;
		}
	      
	      if (!tag_list && (rc = mu_list_create (&tag_list)))
		{
		  mu_sv_compile_error (&mu_sieve_locus, 
                                       _("cannot create tag list: %s"),
			               mu_strerror (rc));
		  err = 1;
		  break;
		}
	      
	      tagrec.tag = tag->name;
	      if (tag->argtype != SVT_VOID)
		{
		  mu_iterator_next (itr);
		  if (mu_iterator_is_done (itr))
		    {
		      mu_sv_compile_error (&mu_sieve_locus, 
			   _("required argument for tag %s is missing"),
					   tag->name);
		      err = 1;
		      break;
		    }
		  mu_iterator_current (itr, (void **)&tagrec.arg);
		  if (tagrec.arg->type != tag->argtype)
		    {
		      mu_sv_compile_error (&mu_sieve_locus, 
					   _("type mismatch in argument to "
					     "tag `%s'"),
					   tag->name);
		      mu_sv_compile_error (&mu_sieve_locus, 
					   _("expected %s but passed %s"),
					   mu_sieve_type_str (tag->argtype),
					   mu_sieve_type_str
					                 (tagrec.arg->type));
		      err = 1;
		      break;
		    }
		}
	      else
		tagrec.arg = NULL;
	      
	      tagptr = mu_sieve_malloc (mu_sieve_machine, sizeof (*tagptr));
	      *tagptr = tagrec;
	      mu_list_append (tag_list, tagptr);

	      if (cf)
		{
		  if (!chk_list && (rc = mu_list_create (&chk_list)))
		    {
		      mu_sv_compile_error (&mu_sieve_locus, 
 			  	         _("cannot create check list: %s"),
					   mu_strerror (rc));
		      err = 1;
		      break;
		    }
		  if (mu_list_do (chk_list, _compare_ptr, cf) == 0)
		    mu_list_append (chk_list, cf);
		}
	    }
	  else if (*exp_arg == SVT_VOID)
	    {
	      mu_sv_compile_error (&mu_sieve_locus, 
                                   _("too many arguments in call to `%s'"),
 			           reg->name);
	      err = 1;
	      break;
	    }
	  else
	    {
	      if (*exp_arg != val->type)
		{
		  if (*exp_arg == SVT_STRING_LIST && val->type == SVT_STRING)
		    {
		      mu_list_t list;

		      mu_list_create (&list);
		      mu_list_append (list, val->v.string);
		      mu_sieve_mfree (mu_sieve_machine, val);
		      val = mu_sieve_value_create (SVT_STRING_LIST, list);
		    }
		  else
		    {
		      mu_sv_compile_error (&mu_sieve_locus, 
                                   _("type mismatch in argument %lu to `%s'"),
				   (unsigned long) (exp_arg - reg->req_args + 1),
					   reg->name);
		      mu_sv_compile_error (&mu_sieve_locus, 
					   _("expected %s but passed %s"),
					   mu_sieve_type_str (*exp_arg),
					   mu_sieve_type_str (val->type));
		      err = 1;
		      break;
		    }
		}

	      if (!arg_list && (rc = mu_list_create (&arg_list)))
		{
		  mu_sv_compile_error (&mu_sieve_locus, 
                                       _("cannot create arg list: %s"),
			               mu_strerror (rc));
		  err = 1;
		  break;
		}
	      
	      mu_list_append (arg_list, val);
	      exp_arg++;
	    }	    
	}
      mu_iterator_destroy (&itr);
    }

  if (!err)
    {
      if (*exp_arg != SVT_VOID)
	{
	  mu_sv_compile_error (&mu_sieve_locus, 
                               _("too few arguments in call to `%s'"),
			       reg->name);
	  err = 1;
	}

      if (chk_list)
	{
	  struct check_arg chk_arg;
      
	  chk_arg.name = reg->name;
	  chk_arg.tags = tag_list;
	  chk_arg.args = arg_list;
	  err = mu_list_do (chk_list, _run_checker, &chk_arg);
	}
    }
  
  if (!err)
    err = mu_sv_code_list (arg_list)
          || mu_sv_code_list (tag_list)
          || mu_sv_code_string (reg->name);

  if (err)
    {
      mu_list_destroy (&arg_list);
      mu_list_destroy (&tag_list);
      mu_list_destroy (&chk_list);
    }

  return err;
}

int
mu_sv_code_source (const char *name)
{
  char *s;
  
  if (mu_list_locate (mu_sieve_machine->source_list, 
                      (void*) name, (void **) &s))
    {
      s = mu_sieve_mstrdup (mu_sieve_machine, name);
      mu_list_append (mu_sieve_machine->source_list, s);
    }
  
  return mu_sv_code_instr (_mu_sv_instr_source)
	 || mu_sv_code_string (s);
}

int
mu_sv_code_line (size_t line)
{
  sieve_op_t op;

  op.line = line;
  return mu_sv_code_instr (_mu_sv_instr_line)
	 || mu_sv_code (&op);
}

static int sieve_source_changed;

void
mu_sv_change_source ()
{
  sieve_source_changed = 1;
}

static int
sieve_check_source_changed ()
{
  if (sieve_source_changed)
    {
      sieve_source_changed = 0;
      return mu_sv_code_source (mu_sieve_locus.source_file);
    }
  return 0;
}

int
mu_sv_code_action (mu_sieve_register_t *reg, mu_list_t arglist)
{
  return sieve_check_source_changed ()
         || mu_sv_code_line (mu_sieve_locus.source_line)
         || mu_sv_code_instr (_mu_sv_instr_action)
         || mu_sv_code_command (reg, arglist);
}

int
mu_sv_code_test (mu_sieve_register_t *reg, mu_list_t arglist)
{
  return sieve_check_source_changed ()
         || mu_sv_code_line (mu_sieve_locus.source_line)
         || mu_sv_code_instr (_mu_sv_instr_test)
         || mu_sv_code_command (reg, arglist);
}

void
mu_sv_code_anyof (size_t start)
{
  size_t end = mu_sieve_machine->pc;
  while (mu_sieve_machine->prog[start+1].pc != 0)
    {
      size_t next = mu_sieve_machine->prog[start+1].pc;
      mu_sieve_machine->prog[start].instr = _mu_sv_instr_brnz;
      mu_sieve_machine->prog[start+1].pc = end - start - 2;
      start = next;
    }
  mu_sieve_machine->prog[start].instr = _mu_sv_instr_nop;
  mu_sieve_machine->prog[start+1].instr = _mu_sv_instr_nop;
}

void
mu_sv_code_allof (size_t start)
{
  size_t end = mu_sieve_machine->pc;
  
  while (mu_sieve_machine->prog[start+1].pc != 0)
    {
      size_t next = mu_sieve_machine->prog[start+1].pc;
      mu_sieve_machine->prog[start+1].pc = end - start - 2;
      start = next;
    }
  mu_sieve_machine->prog[start].instr = _mu_sv_instr_nop;
  mu_sieve_machine->prog[start+1].instr = _mu_sv_instr_nop;
}
		
