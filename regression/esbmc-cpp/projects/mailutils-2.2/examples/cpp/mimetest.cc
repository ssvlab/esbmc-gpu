/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

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

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <mailutils/cpp/mailutils.h>

using namespace std;
using namespace mailutils;

void message_display_parts (Message& msg, int indent);

bool print_attachments = false;
int indent_level = 4;

void
print_file (const std::string& fname, int indent)
{
  char buf[128];
  FILE *fp = fopen (fname.c_str (), "r");

  if (!fp)
    {
      cerr << "can't open file " << fname << ": " << strerror (errno) << endl;
      return;
    }

  while (fgets (buf, sizeof buf, fp))
    cout << setw (indent) << setfill (' ') << buf;
  fclose (fp);
  unlink (fname.c_str ());
}

int
main (int argc, char **argv)
{
  int i = 0;
  bool debug = false;

  for (i = 1; i < argc; i++)
    {
      if (strcmp (argv[i], "-d") == 0)
        debug = true;
      else if (strcmp (argv[i], "-p") == 0)
        print_attachments = true;
      else if (strcmp (argv[i], "-i") == 0)
        indent_level = strtoul (argv[++i], NULL, 0);
      else
        break;
    }

  /* Registration. */
  registrar_record (mu_imap_record);
  registrar_record (mu_pop_record);
  registrar_record (mu_mbox_record);
  registrar_set_default_record (mu_mbox_record);

  MailboxDefault mbox (argv[i]);
  
  /* Debugging trace. */
  if (debug)
    {
      Debug debug = mbox.get_debug ();
      debug.set_level (MU_DEBUG_LEVEL_UPTO (MU_DEBUG_PROT));
    }

  /* Open the mailbox for reading only. */
  mbox.open ();

  /* Iterate through the entire message set. */
  size_t count = mbox.messages_count ();

  for (size_t i = 1; i <= count; ++i)
  {
    Message msg = mbox.get_message (i);
    Header hdr = msg.get_header ();

    cout << "Message: " << i << endl;
    cout << "From: " << hdr[MU_HEADER_FROM] << endl;
    cout << "Subject: " << hdr.get_value (MU_HEADER_SUBJECT, "[none]") << endl;
    cout << "Number of parts in message - " << msg.get_num_parts () << endl;
    cout << "Total message size - "
	 << msg.size () << "/" << msg.lines () << endl;

    try {
      message_display_parts (msg, 0);
    }
    catch (Exception& e)
    {
      cerr << e.method () << ": " << e.what () << endl;
    }
  }

  mbox.close ();
  return 0;
}

static void
print_message_part_sizes (Message& part, int indent)
{
  Header hdr = part.get_header ();
  Body body = part.get_body ();

  cout << setw (indent) << setfill (' ') << "Message part size - ";
  cout << part.size () << "/" << part.lines () << ": "
       << hdr.size  () << "/" << hdr.lines  () << ", "
       << body.size () << "/" << body.lines () << endl;
}
  
void
message_display_parts (Message& msg, int indent)
{
  /* How many parts does the message has? */
  size_t nparts = msg.get_num_parts ();

  /* Iterate through all the parts.  Treat type "message/rfc822"
     differently, since it is a message of its own that can have other
     subparts(recursive). */
  for (int j = 1; j <= nparts; j++)
    {
      Message part = msg.get_part (j);
      Header hdr = part.get_header ();

      string type ("text/plain");
      string encoding ("7bit");
      try {
	type = hdr[MU_HEADER_CONTENT_TYPE];
	encoding = hdr[MU_HEADER_CONTENT_TRANSFER_ENCODING];
      }
      catch (Exception& e)
      {
	if (e.status () != MU_ERR_NOENT) {
	  cerr << e.method () << ": " << e.what () << endl;
	  exit (1);
	}
      }

      cout << setw (indent) << setfill (' ')
	   << "Type of part " << j << " = " << type << endl;

      print_message_part_sizes (part, indent);

      bool ismulti = part.is_multipart ();
      if (type == "message/rfc822" || ismulti)
        {
          if (!ismulti)
	    part = part.unencapsulate ();

	  Header hdr = part.get_header ();
	  string from = hdr[MU_HEADER_FROM];
	  string subject = hdr.get_value (MU_HEADER_SUBJECT, "[none]");

	  cout << setw (indent) << setfill (' ')
	       << "Encapsulated message : " << from << "\t" << subject << endl;
	  cout << setw (indent) << setfill (' ') << "Begin" << endl;

	  size_t nsubparts = part.get_num_parts ();
          message_display_parts (part, indent + indent_level);
        }
      else if (strncasecmp (type.c_str (), "text/plain",
			    strlen ("text/plain")) == 0 ||
               strncasecmp (type.c_str (), "text/html",
			    strlen ("text/html")) == 0 ||
	       type.empty ())
        {
	  cout << setw (indent) << setfill (' ') << "Text Message" << endl;
	  cout << setw (indent) << setfill (' ') << "Begin" << endl;

	  Body body = part.get_body ();
	  Stream stream = body.get_stream ();

	  FilterStream filter (stream, encoding, 0, 0);
          int offset = 0;
	  char buf[2048];

          while (filter.readline (buf, sizeof (buf), offset) == 0 &&
		 filter.get_read_count ())
            {
	      cout << setw (indent) << setfill (' ') << buf;
              offset += filter.get_read_count ();
            }
        }
      else
        {
          /* Save the attachements. */
	  string fname;
	  try {
	    fname = part.get_attachment_name ();
	  }
	  catch (Exception& e) {
	    fname = mailutils::tempname ();
	  }
	  cout << setw (indent) << setfill (' ')
	       << "Attachment - saving " << "[" << fname << "]" << endl;
	  cout << setw (indent) << setfill (' ') << "Begin" << endl;

          part.save_attachment ();
          if (print_attachments)
            print_file (fname, indent);
        }
      cout << endl << setw (indent) << setfill (' ') << "End" << endl;
    }
}

