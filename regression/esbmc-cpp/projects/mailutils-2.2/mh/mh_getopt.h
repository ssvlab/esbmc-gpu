/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2005, 2007, 2009, 2010 Free Software Foundation,
   Inc.

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

#include <mailutils/nls.h>
#include "argp.h"
#include "mailutils/libargp.h"

#define MH_OPT_BOOL 1
#define MH_OPT_ARG  2

struct mh_option
{
  char *opt;
  int match_len;
  int flags;
  char *arg;
};

struct mh_argp_data
{
  struct mh_option *mh_option;
  argp_parser_t handler;
  int errind;
  void *closure;
  char *doc;
};

enum mh_arg {
  ARG_ADD = 256,		
  ARG_AFTER,		
  ARG_ALIAS,		
  ARG_ALL,		
  ARG_AND,		
  ARG_ANNOTATE,		
  ARG_AUDIT,		
  ARG_AUTO,		
  ARG_BEFORE,		
  ARG_BELL,		
  ARG_BUILD,		
  ARG_CC,	        
  ARG_CFLAGS,		
  ARG_CHANGECUR,
  ARG_CHARSET,
  ARG_CHECK,
  ARG_CHUNKSIZE,
  ARG_CLEAR,		
  ARG_COMPAT,		
  ARG_COMPONENT,		
  ARG_COMPOSE,		
  ARG_CREATE,		
  ARG_DATE,		
  ARG_DATEFIELD,		
  ARG_DEBUG,		
  ARG_DELETE,		
  ARG_DRAFT,		
  ARG_DRAFTFOLDER,	
  ARG_DRAFTMESSAGE,	
  ARG_DRY_RUN,		
  ARG_DUMP,		
  ARG_EDITOR,		
  ARG_FAST,		
  ARG_FCC,		
  ARG_FILE,		
  ARG_FILTER,		
  ARG_FOLDER,		
  ARG_FORM,		
  ARG_FORMAT,		
  ARG_FORWARD,		
  ARG_FROM,		
  ARG_GROUP,              
  ARG_HEADER,		
  ARG_INPLACE,		
  ARG_INTERACTIVE,	
  ARG_LBRACE,		
  ARG_LENGTH,		
  ARG_LICENSE,		
  ARG_LIMIT,		
  ARG_LINK,		
  ARG_LIST,		
  ARG_MIME,		
  ARG_MOREPROC,		
  ARG_MSGID,		
  ARG_NOALIAS,            
  ARG_NOAUDIT,		
  ARG_NOAUTO,		
  ARG_NOBELL,		
  ARG_NOCC,		
  ARG_NOCHANGECUR,	
  ARG_NOCHECK,		
  ARG_NOCLEAR,		
  ARG_NOCOMPOSE,		
  ARG_NOCREATE,		
  ARG_NODATE,		
  ARG_NODATEFIELD,	
  ARG_NODRAFTFOLDER,	
  ARG_NOEDIT,		
  ARG_NOFAST,		
  ARG_NOFILTER,		
  ARG_NOFORMAT,		
  ARG_NOFORWARD,		
  ARG_NOHEADER,		
  ARG_NOHEADERS,		
  ARG_NOINTERACTIVE,      
  ARG_NOINPLACE,		
  ARG_NOLIMIT,		
  ARG_NOLIST,		
  ARG_NOMIME,		
  ARG_NOMOREPROC,		
  ARG_NOMSGID,		
  ARG_NOPAUSE,		
  ARG_NOPUBLIC,		
  ARG_NOPUSH,		
  ARG_NOQUIET,            
  ARG_NOREALSIZE,		
  ARG_NORECURSIVE,        
  ARG_NOREVERSE,	
  ARG_NORMALIZE,          
  ARG_NOSERIALONLY,	
  ARG_NOSHOW,		
  ARG_NOSTORE,		
  ARG_NOT,		
  ARG_NOTEXTFIELD,	
  ARG_NOTOTAL,		
  ARG_NOTRUNCATE,		
  ARG_NOUSE,		
  ARG_NOVERBOSE,		
  ARG_NOWATCH,		
  ARG_NOWHATNOWPROC,	
  ARG_NOZERO,		
  ARG_NUMFIELD,		
  ARG_OR,		        
  ARG_PACK,               
  ARG_PART,		
  ARG_PATTERN,		
  ARG_PAUSE,		
  ARG_POP,		
  ARG_PRESERVE,		
  ARG_PRINT,		
  ARG_PROMPT,		
  ARG_PUBLIC,		
  ARG_PUSH,		
  ARG_QUERY,		
  ARG_QUIET,		
  ARG_RBRACE,		
  ARG_REALSIZE,		
  ARG_RECURSIVE,		
  ARG_REORDER,		
  ARG_REVERSE,		
  ARG_SEQUENCE,		
  ARG_SERIALONLY,		
  ARG_SHOW,		
  ARG_SOURCE,		
  ARG_SPLIT,		
  ARG_STORE,		
  ARG_SUBJECT,		
  ARG_TEXT,		
  ARG_TEXTFIELD,		
  ARG_TO,		        
  ARG_TOTAL,		
  ARG_TRUNCATE,		
  ARG_TYPE,		
  ARG_USE,		
  ARG_USER,               
  ARG_VERBOSE,		
  ARG_WATCH,		
  ARG_WHATNOWPROC,	
  ARG_WIDTH,	
  ARG_ZERO
};

void mh_argp_init (const char *vers);
void mh_argv_preproc (int argc, char **argv, struct mh_argp_data *data);
int mh_getopt (int argc, char **argv, struct mh_option *mh_opt, const char *doc);
int mh_argp_parse (int *argc, char **argv[],
		   int flags,
		   struct argp_option *option,
		   struct mh_option *mh_option,
		   char *argp_doc, char *doc,
		   argp_parser_t handler,
		   void *closure, int *index);

void mh_help (struct mh_option *mh_option, const char *doc);
void mh_license (const char *name);

void mh_opt_notimpl (const char *name);
void mh_opt_notimpl_warning (const char *name);
