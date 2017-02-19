@comment   GNU Mailutils -- a suite of utilities for electronic mail
@comment   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2010
@comment   Free Software Foundation, Inc.
@comment
@comment   GNU Mailutils is free software; you can redistribute it and/or modify
@comment   it under the terms of the GNU General Public License as published by
@comment   the Free Software Foundation; either version 3, or (at your option)
@comment   any later version.
@comment
@comment   GNU Mailutils is distributed in the hope that it will be useful,
@comment   but WITHOUT ANY WARRANTY; without even the implied warranty of
@comment   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
@comment   GNU General Public License for more details.
@comment
@comment   You should have received a copy of the GNU General Public License
@comment   along with GNU Mailutils; if not, write to the Free Software
@comment   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
@comment   MA 02110-1301 USA 

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mailutils/mailutils.h>

int
main (int argc, const char **argv)
@{
  char *from;
  char *subject;
  mu_mailbox_t mbox;
  size_t msgno, total = 0;
  int status;

  /* Register the formats. */
  mu_register_all_mbox_formats ();

  status = mu_mailbox_create_default (&mbox, argv[1]);
  if (status != 0)
    @{
      mu_error ("mu_mailbox_create: %s", mu_strerror (status));
      exit (EXIT_FAILURE);
    @}

  status = mu_mailbox_open (mbox, MU_STREAM_READ);
  if (status != 0)
    @{
      mu_error ("mu_mailbox_open: %s", mu_strerror (status));
      exit (EXIT_FAILURE);
    @}

  mu_mailbox_messages_count (mbox, &total);

  for (msgno = 1; msgno <= total; msgno++)
    @{
      mu_message_t msg;
      mu_header_t hdr;

      if ((status = mu_mailbox_get_message (mbox, msgno, &msg)) != 0
          || (status = mu_message_get_header (msg, &hdr)) != 0)
        @{
          mu_error ("Error message: %s", mu_strerror (status));
          exit (EXIT_FAILURE);
        @}

      if (mu_header_aget_value (hdr, MU_HEADER_FROM, &from))
        from = strdup ("(NO FROM)");

      if (mu_header_aget_value (hdr, MU_HEADER_SUBJECT, &subject))
        subject = strdup ("(NO SUBJECT)");

      printf ("%s\t%s\n", from, subject);
      free (from);
      free (subject);
    @}

  status = mu_mailbox_close (mbox);
  if (status != 0)
    @{
      mu_error ("mu_mailbox_close: %s", mu_strerror (status));
      exit (EXIT_FAILURE);
    @}

  mu_mailbox_destroy (&mbox);
  return 0;
@}
