/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2008, 2010 Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <sysexits.h>
#include <mailutils/kwd.h>
#include <mailutils/nls.h>

struct mu_kwd exittab[] = {
  { N_("Normal termination"), EX_OK },
  { N_("Unspecified error"), EXIT_FAILURE },
  { N_("Usage error"), EX_USAGE },
  { N_("Incorrect input data"), EX_DATAERR },
  { N_("No input data"), EX_NOINPUT },     
  { N_("User does not exist"), EX_NOUSER },      
  { N_("Host does not exist"), EX_NOHOST },      
  { N_("Service unavailable"), EX_UNAVAILABLE }, 
  { N_("Software error"), EX_SOFTWARE },    
  { N_("Operating system error"), EX_OSERR },       
  { N_("Required system file does not exist or cannot be opened"), EX_OSFILE },
  { N_("Output file cannot be created"), EX_CANTCREAT },   
  { N_("I/O error"), EX_IOERR },       
  { N_("Temporary failure"), EX_TEMPFAIL },    
  { N_("Remote protocol error"), EX_PROTOCOL },    
  { N_("Insufficient permissions"), EX_NOPERM },      
  { N_("Configuration error"), EX_CONFIG },      
  { 0 }
};

const char *
mu_strexit (int code)
{
  const char *str;
  if (mu_kwd_xlat_tok (exittab, code, &str))
    str = N_("Unknown exit code");
  return gettext (str);
}
  

