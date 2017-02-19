/*******************************************************************\

Module: Race Detection for Threaded Goto Programs

Author: Daniel Kroening

Date: February 2006

\*******************************************************************/

#include <expr_util.h>
#include <std_expr.h>
#include <namespace.h>
#include <arith_tools.h>
#include <pointer-analysis/goto_program_dereference.h>

#include "rw_set.h"

/*******************************************************************\

Function: rw_sett::compute

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void rw_sett::compute(const codet &code)
{
  const irep_idt &statement=code.get_statement();
  if(statement=="assign")
  {
    assert(code.operands().size()==2);
    assign(code.op0(), code.op1());
  }
}

/*******************************************************************\

Function: rw_sett::assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void rw_sett::assign(const exprt &lhs, const exprt &rhs)
{
  read(rhs);
  read_write_rec(lhs, false, true, "", guardt());
}

/*******************************************************************\

Function: rw_sett::read_write_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void rw_sett::read_write_rec(
  const exprt &expr,
  bool r, bool w,
  const std::string &suffix,
  const guardt &guard)
{
  if(expr.id()=="symbol" && !expr.has_operands())
  {

    const symbol_exprt &symbol_expr=to_symbol_expr(expr);

    const symbolt *symbol;

    if(!ns.lookup(symbol_expr.get_identifier(), symbol))
    {

      if(!symbol->static_lifetime && !(expr.type().id()=="pointer") && !symbol->cuda_shared)
      {
        return; // ignore for now
      }

      if(symbol->name=="c::__ESBMC_alloc" ||
         symbol->name=="c::__ESBMC_alloc_size" ||
         symbol->name=="c::stdin" ||
         symbol->name=="c::stdout" ||
         symbol->name=="c::stderr" ||
         symbol->name=="c::sys_nerr")
      {
        return; // ignore for now
      }

      //improvements for CUDA features
      if(expr.type().tag()=="__dim3" ||
    	 expr.type().tag().as_string().find("arg_struct")!=std::string::npos	||
    	 symbol->name.as_string().find("c::__dim3")!=std::string::npos  	||
		 symbol->name.as_string().find("c::io")!=std::string::npos  	||
		 symbol->name.as_string().find("c::fopen")!=std::string::npos  	||
		 symbol->name.as_string().find("c::pthread_lib")!=std::string::npos  	||
		 symbol->name.as_string().find("cpp::__CPROVER")!=std::string::npos  	||
		 symbol->name.as_string().find("cpp::__cuda")!=std::string::npos  	||
		 symbol->name.as_string().find("cpp::__ESBMC")!=std::string::npos  	||
    	 symbol->name.as_string().find("cpp::cuda")!=std::string::npos  	||
		 symbol->name.as_string().find("cpp::itoa")!=std::string::npos  	||
    	 symbol->name.as_string().find("cpp::mem")!=std::string::npos  	||
		 symbol->name.as_string().find("cpp::rev")!=std::string::npos  	||
		 symbol->name.as_string().find("cpp::search")!=std::string::npos  	||
		 symbol->name.as_string().find("cpp::std")!=std::string::npos  	||
		 symbol->name.as_string().find("cpp::str")!=std::string::npos  	||
		 symbol->name.as_string().find("verify_kernel")!=std::string::npos  	||
		 symbol->name.as_string().find("virtual_table")!=std::string::npos  	||
		 symbol->name=="c::indexOfBlock" 		||
		 symbol->name=="c::warpSize" 		||
		 symbol->name=="cpp::blockGlobal"	||
		 symbol->name=="cpp::cond"	||
		 symbol->name=="cpp::count"	||
		 symbol->name=="cpp::cudaDeviceList"	||
		 symbol->name=="cpp::cudaThreadList"	||
		 symbol->name=="cpp::lastError" 		||
		 symbol->name=="cpp::mutex"	||
		 symbol->name=="cpp::n_threads"	 	||
		 symbol->name=="cpp::std::message" 	||
		 symbol->name=="cpp::threadGlobal"	||
		 symbol->name=="cpp::threads_id" )
      {
    	  return;
      }

    }

    irep_idt object=id2string(symbol_expr.get_identifier())+suffix;

    entryt &entry=entries[object];
    entry.object=object;
    entry.r=entry.r || r;
    entry.w=entry.w || w;
    entry.guard = migrate_expr_back(guard.as_expr());
  }
  else if(expr.id()=="member")
  {
    assert(expr.operands().size()==1);
    const std::string &component_name=expr.component_name().as_string();
    read_write_rec(expr.op0(), r, w, "."+component_name+suffix, guard);
  }
  else if(expr.id()=="index")
  {
    // we don't distinguish the array elements for now
    assert(expr.operands().size()==2);
    std::string tmp;

    tmp = integer2string(binary2integer(expr.op1().value().as_string(), true),10);

    read_write_rec(expr.op0(), r, w, "["+suffix+tmp+"]", guard);
    read(expr.op1(), guard);
  }
  else if(expr.id()=="dereference")
  {
    assert(expr.operands().size()==1);
    read(expr.op0(), guard);

    exprt tmp(expr.op0());
    expr2tc tmp_expr;
    migrate_expr(tmp, tmp_expr);
    dereference(target, tmp_expr, ns, value_sets);
    tmp = migrate_expr_back(tmp_expr);

    read_write_rec(tmp, r, w, suffix, guard);
  }
  else if(expr.is_address_of() ||
          expr.id()=="implicit_address_of")
  {
    assert(expr.operands().size()==1);

  }
  else if(expr.id()=="if")
  {
    assert(expr.operands().size()==3);
    read(expr.op0(), guard);

    guardt true_guard(guard);
    expr2tc tmp_expr;
    migrate_expr(expr.op0(), tmp_expr);
    true_guard.add(tmp_expr);
    read_write_rec(expr.op1(), r, w, suffix, true_guard);

    guardt false_guard(guard);
    migrate_expr(gen_not(expr.op0()), tmp_expr);
    false_guard.add(tmp_expr);
    read_write_rec(expr.op2(), r, w, suffix, false_guard);
  }
  else
  {
    forall_operands(it, expr)
      read_write_rec(*it, r, w, suffix, guard);
  }
}
