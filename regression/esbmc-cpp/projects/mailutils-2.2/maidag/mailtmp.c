/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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

#include "maidag.h"

struct mail_tmp
{
  mu_stream_t stream;
  size_t line;
  char *tempfile;
  const char *from;
  int had_nl;
};  

int
mail_tmp_begin (struct mail_tmp **pmtmp, const char *from)
{
  int status;
  struct mail_tmp *mtmp = malloc (sizeof *mtmp);
  
  if (!mtmp)
    return ENOMEM;

  memset (mtmp, 0, sizeof *mtmp);

  mtmp->tempfile = mu_tempname (NULL);
  if ((status = mu_file_stream_create (&mtmp->stream, mtmp->tempfile,
				       MU_STREAM_RDWR)))
    {
      free (mtmp);
      maidag_error (_("unable to open temporary file: %s"),
		    mu_strerror (status));
      return status;
    }

  if ((status = mu_stream_open (mtmp->stream)))
    {
      free (mtmp);
      maidag_error (_("unable to open temporary file: %s"),
		    mu_strerror (status));
      return status;
    }
  mtmp->from = from;
  *pmtmp = mtmp;
  return 0;
}

int
mail_tmp_add_line (struct mail_tmp *mtmp, char *buf, size_t buflen)
{
  int status = 0;
  
  mtmp->line++;
  if (mtmp->line == 1)
    {
      const char *from = mtmp->from;
      
      if (buflen >= 5 && memcmp (buf, "From ", 5))
	{
	  struct mu_auth_data *auth = NULL;
	  if (!from)
	    {
	      auth = mu_get_auth_by_uid (getuid ());
	      if (auth)
		from = auth->name;
	    }
	  if (from)
	    {
	      time_t t;
	      char *envs;
	      
	      time (&t);
	      asprintf (&envs, "From %s %s", from, ctime (&t));
	      status = mu_stream_sequential_write (mtmp->stream, 
						   envs,
						   strlen (envs));
	      free (envs);
	    }
	  else
	    {
	      maidag_error (_("cannot determine sender address"));
	      return EINVAL;
	    }
	  if (auth)
	    mu_auth_data_free (auth);
	}
    }
  else if (buflen >= 5 && !memcmp (buf, "From ", 5))
    {
      static char *escape = ">";
      status = mu_stream_sequential_write (mtmp->stream, escape, 1);
    }

  if (!status)
    status = mu_stream_sequential_write (mtmp->stream, buf, buflen);
      
  if (status)
    {
      maidag_error (_("error writing temporary file: %s"), 
                    mu_strerror (status));
      mu_stream_destroy (&mtmp->stream, mu_stream_get_owner (mtmp->stream));
    }
  mtmp->had_nl = buf[buflen-1] == '\n';
  return status;
}

int
mail_tmp_finish (struct mail_tmp *mtmp, mu_mailbox_t *mbox)
{
  int status;
  static char *newline = "\n";
  size_t n;
  
  if (!mtmp->had_nl)
    status = mu_stream_sequential_write (mtmp->stream, newline, 1);

  status = mu_stream_sequential_write (mtmp->stream, newline, 1);
  unlink (mtmp->tempfile);
  free (mtmp->tempfile);
  mtmp->tempfile = NULL;
  
  if (status)
    {
      errno = status;
      maidag_error (_("error writing temporary file: %s"), 
                    mu_strerror (status));
      mu_stream_destroy (&mtmp->stream, mu_stream_get_owner (mtmp->stream));
      return status;
    }

  mu_stream_flush (mtmp->stream);
  if ((status = mu_mailbox_create (mbox, "mbox:/dev/null")) 
      || (status = mu_mailbox_open (*mbox, MU_STREAM_READ))
      || (status = mu_mailbox_set_stream (*mbox, mtmp->stream)))
    {
      maidag_error (_("error opening temporary file: %s"), 
                    mu_strerror (status));
      mu_stream_destroy (&mtmp->stream, mu_stream_get_owner (mtmp->stream));
      return status;
    }

  status = mu_mailbox_messages_count (*mbox, &n);
  if (status)
    {
      errno = status;
      maidag_error (_("error creating temporary message: %s"),
		    mu_strerror (status));
      mu_stream_destroy (&mtmp->stream, mu_stream_get_owner (mtmp->stream));
      return status;
    }

  mtmp->stream = NULL;
  mtmp->line = 0;
  
  return status;
  
}

void
mail_tmp_destroy (struct mail_tmp **pmtmp)
{
  struct mail_tmp *mtmp = *pmtmp;

  if (mtmp)
    {
      if (mtmp->tempfile)
	{
	  unlink (mtmp->tempfile);
	  free (mtmp->tempfile);
	}
      mu_stream_destroy (&mtmp->stream, mu_stream_get_owner (mtmp->stream));
      free (*pmtmp);
      *pmtmp = NULL;
    }
}


  
