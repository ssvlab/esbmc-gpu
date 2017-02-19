/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2005, 2007, 2008, 2009, 2010 Free Software
   Foundation, Inc.

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

#include "imap4d.h"

enum value_type { STORE_SET, STORE_ADD, STORE_UNSET };

struct store_parse_closure
{
  enum value_type how;
  int ack;
  int type;
  int isuid;
  size_t *set;
  int count;
};
  
static int
store_thunk (imap4d_parsebuf_t p)
{
  struct store_parse_closure *pclos = imap4d_parsebuf_data (p);
  char *msgset;
  char *data;
  int status;
  
  msgset = imap4d_parsebuf_next (p, 1);
  data = imap4d_parsebuf_next (p, 1);

  if (*data == '+')
    {
      pclos->how = STORE_ADD;
      data++;
    }
  else if (*data == '-')
    {
      pclos->how = STORE_UNSET;
      data++;
    }
  else
    pclos->how = STORE_SET;
  
  if (mu_c_strcasecmp (data, "FLAGS"))
    imap4d_parsebuf_exit (p, "Bogus data item");
  data = imap4d_parsebuf_next (p, 1);

  if (*data == '.')
    {
      data = imap4d_parsebuf_next (p, 1);
      if (mu_c_strcasecmp (data, "SILENT") == 0)
	{
	  pclos->ack = 0;
	  imap4d_parsebuf_next (p, 1);
	}
      else
	imap4d_parsebuf_exit (p, "Bogus data suffix");
    }

  /* Get the message numbers in set[].  */
  status = util_msgset (msgset, &pclos->set, &pclos->count, pclos->isuid);
  switch (status)
    {
    case 0:
      break;

    case EINVAL:
      /* See RFC 3501, section 6.4.8, and a comment to the equivalent code
	 in fetch.c */
      p->err_text = "Completed";
      return RESP_OK;

    default:
      p->err_text = "Failed to parse message set";
      return RESP_NO;
    }      

  if (p->token[0] != '(')
    imap4d_parsebuf_exit (p, "Syntax error");
  imap4d_parsebuf_next (p, 1);
  
  do
    {
      int t;
      if (!util_attribute_to_type (p->token, &t))
	pclos->type |= t;
    }
  while (imap4d_parsebuf_next (p, 1) && p->token[0] != ')');
  return RESP_OK;
}

int
imap4d_store0 (imap4d_tokbuf_t tok, int isuid, char **ptext)
{
  int rc;
  struct store_parse_closure pclos;

  memset (&pclos, 0, sizeof pclos);
  pclos.ack = 1;
  pclos.isuid = isuid;
  
  rc = imap4d_with_parsebuf (tok,
			     IMAP4_ARG_1 + !!isuid,
			     ".",
			     store_thunk, &pclos,
			     ptext);
  if (rc == RESP_OK)
    {
      size_t i;
      
      for (i = 0; i < pclos.count; i++)
	{
	  mu_message_t msg = NULL;
	  mu_attribute_t attr = NULL;
	  size_t msgno = isuid ? uid_to_msgno (pclos.set[i]) : pclos.set[i];
      
	  if (msgno)
	    {
	      mu_mailbox_get_message (mbox, msgno, &msg);
	      mu_message_get_attribute (msg, &attr);
	      
	      switch (pclos.how)
		{
		case STORE_ADD:
		  mu_attribute_set_flags (attr, pclos.type);
		  break;
		  
		case STORE_UNSET:
		  mu_attribute_unset_flags (attr, pclos.type);
		  break;
      
		case STORE_SET:
		  mu_attribute_unset_flags (attr, 0xffffffff); /* FIXME */
		  mu_attribute_set_flags (attr, pclos.type);
		}
	    }
	  
	  if (pclos.ack)
	    {
	      util_send ("* %lu FETCH (", (unsigned long) msgno);
	      
	      if (isuid)
		util_send ("UID %lu ", (unsigned long) msgno);
	      util_send ("FLAGS (");
	      util_print_flags (attr);
	      util_send ("))\r\n");
	    }
	  /* Update the flags of uid table.  */
	  imap4d_sync_flags (pclos.set[i]);
	}

      *ptext = "Completed";
    }

  free (pclos.set);
  
  return rc;
}

int
imap4d_store (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  int rc;
  char *err_text;
  
  rc = imap4d_store0 (tok, 0, &err_text);
  return util_finish (command, rc, "%s", err_text);
}

