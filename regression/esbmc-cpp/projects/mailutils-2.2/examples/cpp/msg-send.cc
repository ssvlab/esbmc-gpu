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
#include <mailutils/cpp/mailutils.h>

using namespace std;
using namespace mailutils;

const char USAGE[] =
"usage: mailer [-hd] [-m mailer] [-f from] [to]..."
 ;
const char HELP[] =
"  -h    print this helpful message\n"
"  -m    a mailer URL (default is \"sendmail:\")\n"
"  -f    the envelope from address (default is from user environment)\n"
"  to    a list of envelope to addresses (default is from message)\n"
"\n"
"An RFC2822 formatted message is read from stdin and delivered using\n"
"the mailer.\n"
 ;

int
main (int argc, char *argv[])
{
  int opt;
  int optdebug = 0;
  const char *optmailer = "sendmail:";
  char *optfrom = 0;

  while ((opt = getopt (argc, argv, "hdm:f:")) != -1)
    {
      switch (opt)
        {
        case 'h':
          cout << USAGE << endl << HELP;
          return 0;
          
        case 'd':
          optdebug++;
          break;
          
        case 'm':
          optmailer = optarg;
          break;
          
        case 'f':
          optfrom = optarg;
          break;

        default:
	  cerr << USAGE << endl;
          exit (1);
        }
    }

  /* Register mailers. */
  register_all_mailer_formats ();

  Address from;
  Address to;

  if (optfrom)
    {
      from = Address (optfrom);
    }

  if (argv[optind])
    {
      char **av = argv + optind;
      to = Address ((const char **) av, -1);
    }

  try {
    StdioStream in (stdin, MU_STREAM_SEEKABLE);
    in.open ();

    Message msg;
    msg.set_stream (in);

    Mailer mailer (optmailer);
    if (optdebug)
      {
	Debug debug = mailer.get_debug ();
	debug.set_level (MU_DEBUG_LEVEL_UPTO (MU_DEBUG_PROT));
      }
    mailer.open ();
    mailer.send_message (msg, from, to);
    mailer.close ();
  }
  catch (Exception& e) {
    cerr << e.method () << ": " << e.what () << endl;
    exit (1);
  }

  return 0;
}

