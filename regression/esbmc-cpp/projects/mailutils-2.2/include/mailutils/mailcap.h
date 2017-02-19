/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2003, 2005, 2007, 2010 Free Software
   Foundation, Inc.

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

#ifndef _MAILUTILS_MAILCAP_H
#define _MAILUTILS_MAILCAP_H

#include <mailutils/types.h>
#include <mailutils/errno.h>

/* See RFC1524 (A User Agent Configuration Mechanism).  */

#ifdef __cplusplus
extern "C" {
#endif

/* Create a mailcap from stream.  */
int mu_mailcap_create (mu_mailcap_t * mailcap, mu_stream_t stream);

/* Destroy mailcap object.  */
void mu_mailcap_destroy (mu_mailcap_t * mailcap);

/* Return the number of entries in the mailcap file.  */
int mu_mailcap_entries_count (mu_mailcap_t mailcap, size_t *pno);

/* Return the mailcap record number, no, of the mailcap file .  */
int mu_mailcap_get_entry (mu_mailcap_t mailcap, size_t no,
			  mu_mailcap_entry_t *entry);

/* Return the number of fields in a mailcap entry */  
int mu_mailcap_entry_fields_count (mu_mailcap_entry_t entry,
				   size_t *pcount);

/* Save in buffer[] the content-type of the record.  */
int mu_mailcap_entry_get_typefield (mu_mailcap_entry_t entry,
				    char *buffer, size_t buflen,
				    size_t *pn);

/* Save in buffer[] the view command of the record.  */
int mu_mailcap_entry_get_viewcommand (mu_mailcap_entry_t entry,
				      char *buffer, size_t buflen,
				      size_t *pn);

/* Save in buffer[] the field number no the record .  */
int mu_mailcap_entry_get_field (mu_mailcap_entry_t entry, size_t no,
			        char *buffer, size_t buflen, size_t *pn);

/* Save in buffer the value of a key:
 * mu_mailcap_entry_get_value (entry, "compose", buffer, buflen, pn);
 * i.e compose="lynx %s" -->  "lynx %s" will be saved in the buffer without
 * the quotes.  */
int mu_mailcap_entry_get_value (mu_mailcap_entry_t entry, const char *key,
				char *buffer, size_t buflen, size_t *pn);

/* Helper function saving in buffer, the argument of "compose" field.  */
int mu_mailcap_entry_get_compose (mu_mailcap_entry_t entry, char *buffer,
				  size_t buflen, size_t *pn);

/* Helper function saving in buffer, the argument of "composetyped" field.  */
int mu_mailcap_entry_get_composetyped (mu_mailcap_entry_t entry,
				       char *buffer, size_t buflen,
				       size_t *pn);

/* Helper function saving in buffer, the argument of "edit" field.  */
int mu_mailcap_entry_get_edit (mu_mailcap_entry_t entry, char *buffer,
			       size_t buflen, size_t *pn);

/* Helper function saving in buffer, the argument of "textualnewlines" field.  */
int mu_mailcap_entry_get_textualnewlines (mu_mailcap_entry_t entry,
					  char *buffer, size_t buflen,
					  size_t *pn);

/* Helper function saving in buffer, the argument of "test" field.  */
int mu_mailcap_entry_get_test (mu_mailcap_entry_t entry,
			       char *buffer, size_t buflen, size_t *pn);

/* Helper function saving in buffer, the argument of "x11-bitmap" field.  */
int mu_mailcap_entry_get_x11bitmap (mu_mailcap_entry_t entry,
				    char *buffer, size_t buflen, size_t *pn);

/* Helper function saving in buffer, the argument of "description" field.  */
int mu_mailcap_entry_get_description (mu_mailcap_entry_t entry,
				      char *buffer, size_t buflen,
				      size_t *pn);

/* Helper function saving in buffer, the argument of "nametemplate" field.  */
int mu_mailcap_entry_get_nametemplate (mu_mailcap_entry_t entry,
				       char *buffer, size_t buflen,
			               size_t *pn);

/* Helper function saving in buffer, the argument of "notes" field.  */
int mu_mailcap_entry_get_notes (mu_mailcap_entry_t entry, char *buffer,
			        size_t buflen, size_t *pn);

/* Helper function. Returns *on != 0 if the flag "needsterminal" is in the
   record.  */
int mu_mailcap_entry_needsterminal (mu_mailcap_entry_t entry, int *on);

/* Helper function. Returns *on != 0 if the flag "copiousoutput" is in the
   record.  */
int mu_mailcap_entry_copiousoutput (mu_mailcap_entry_t entry, int *on);

#ifdef __cplusplus
}
#endif
  
#endif
