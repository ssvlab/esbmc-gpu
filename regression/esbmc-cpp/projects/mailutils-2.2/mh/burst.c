/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2006, 2007, 2008, 2009, 2010 Free Software
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

/* MH burst command */

#include <mh.h>
#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
#include <obstack.h>

const char *program_version = "burst (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH burst")"\v"
N_("Options marked with `*' are not yet implemented.\n\
Use -help to obtain the list of traditional MH options.");
static char args_doc[] = "[msgs]";

/* GNU options */
static struct argp_option options[] = {
  {"folder",        ARG_FOLDER,        N_("FOLDER"), 0,
   N_("specify folder to operate upon")},
  {"inplace",      ARG_INPLACE,      N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("replace the source message with the table of contents, insert extracted messages after it") },
  {"noinplace",    ARG_NOINPLACE,    0, OPTION_HIDDEN, ""},
  {"quiet",        ARG_QUIET,        N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("be quiet about the messages that are not in digest format") },
  {"noquiet",      ARG_NOQUIET,      0, OPTION_HIDDEN, ""},
  {"verbose",      ARG_VERBOSE,      N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("verbosely list the actions taken") },
  {"noverbose",    ARG_NOVERBOSE,    0, OPTION_HIDDEN, ""},
  {"recursive",    ARG_RECURSIVE,    N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("recursively expand MIME messages") },
  {"norecursive",  ARG_NORECURSIVE,  0, OPTION_HIDDEN, ""},
  {"length",       ARG_LENGTH,       N_("NUMBER"), 0,
   N_("set minimal length of digest encapsulation boundary (default 1)") },
  {"license",      ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  { NULL }
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"inplace",    1, MH_OPT_BOOL },
  {"quiet",      1, MH_OPT_BOOL },
  {"verbose",    1, MH_OPT_BOOL },
  {NULL}
};

/* Command line switches */
int inplace; 
int quiet;
int verbose;
int recursive;
int eb_min_length = 1;  /* Minimal length of encapsulation boundary */

#define VERBOSE(c) do { if (verbose) { printf c; putchar ('\n'); } } while (0)

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_FOLDER: 
      mh_set_current_folder (arg);
      break;

    case ARG_INPLACE:
      inplace = is_true (arg);
      break;

    case ARG_NOINPLACE:
      inplace = 0;
      break;

    case ARG_LENGTH:
      eb_min_length = strtoul (arg, NULL, 0);
      if (eb_min_length == 0)
	eb_min_length = 1;
      break;
      
    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    case ARG_VERBOSE:
      verbose = is_true (arg);
      break;

    case ARG_NOVERBOSE:
      verbose = 0;
      break;

    case ARG_RECURSIVE:
      recursive = is_true (arg);
      break;
      
    case ARG_NORECURSIVE:
      recursive = 0;
      break;
      
    case ARG_QUIET:
      quiet = is_true (arg);
      break;

    case ARG_NOQUIET:
      quiet = 0;
      break;
      
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}


/* General-purpose data structures */
struct burst_map
{
  int mime;            /* Is mime? */
  size_t msgno;        /* Number of the original message */
  /* Following numbers refer to tmpbox */
  size_t first;        /* Number of the first bursted message */
  size_t count;        /* Number of bursted messages */
};


/* Global data */
struct burst_map map;        /* Currently built map */
struct burst_map *burst_map; /* Finished burst map */
size_t burst_count;          /* Number of items in burst_map */
mu_mailbox_t tmpbox;         /* Temporaty mailbox */			 
struct obstack stk;          /* Stack for building burst_map, etc. */

static int burst_or_copy (mu_message_t msg, int recursive, int copy);


/* MIME messages */
int 
burst_mime (mu_message_t msg)
{
  size_t i, nparts;
  
  mu_message_get_num_parts (msg, &nparts);

  for (i = 1; i <= nparts; i++)
    {
      mu_message_t mpart;
      if (mu_message_get_part (msg, i, &mpart) == 0)
	{
	  if (!map.first)
	    mu_mailbox_uidnext (tmpbox, &map.first);
	  burst_or_copy (mpart, recursive, 1);
	}
    }
  return 0;
}


/* Digest messages */

/* Bursting FSA states accoring to RFC 934:
   
      S1 ::   "-" S3
            | CRLF {CRLF} S1
            | c {c} S2

      S2 ::   CRLF {CRLF} S1
            | c {c} S2

      S3 ::   " " S2
            | c S4     ;; the bursting agent should consider the current
	               ;; message ended.  

      S4 ::   CRLF S5
            | c S4

      S5 ::   CRLF S5
            | c {c} S2 ;; The bursting agent should consider a new
	               ;; message started
*/

#define S1 1
#define S2 2
#define S3 3
#define S4 4
#define S5 5

/* Negative state means no write */
int transtab[][4] = {
/*          DEF    '\n'   ' '   '-' */
/* S1 */ {  S2,    S1,    S2,   -S3 },
/* S2 */ {  S2,    S1,    S2,    S2 },
/* S3 */ { -S4,   -S4,   -S2,   -S4 }, 
/* S4 */ { -S4,   -S5,   -S4,   -S4 },
/* S5 */ {  S2,   -S5,    S2,    S2 }
};

static int
token_num(int c)
{
  switch (c)
    {
    case '\n':
      return 1;
    case ' ':
      return 2;
    case '-':
      return 3;
    default:
      return 0;
    }
}

static void
finish_stream (mu_stream_t *pstr)
{
  mu_message_t msg;
  mu_stream_seek (*pstr, 0, SEEK_SET);
  msg = mh_stream_to_message (*pstr);
  if (!map.first)
    mu_mailbox_uidnext (tmpbox, &map.first);
  burst_or_copy (msg, recursive, 1);
  mu_stream_close (*pstr);
  mu_stream_destroy (pstr, mu_stream_get_owner (*pstr));
}  

static void
flush_stream (mu_stream_t *pstr, char *buf, size_t size)
{
  int rc;

  if (size == 0)
    return;
  if (!*pstr
      && ((rc = mu_temp_file_stream_create (pstr, NULL)) != 0
	  || (rc = mu_stream_open (*pstr))))
    {
      mu_error (_("Cannot open temporary file: %s"),
		mu_strerror (rc));
      exit (1);
    }
  rc = mu_stream_sequential_write (*pstr, buf, size);
  if (rc)
    {
      mu_error (_("error writing temporary stream: %s"),
		mu_strerror (rc));
      exit (1); /* FIXME: better error handling please */
    }
}

int
burst_digest (mu_message_t msg)
{
  mu_stream_t is, os = NULL;
  char *buf;
  size_t bufsize;
  size_t n;
  int state = S1;
  size_t count = 0;
  int eb_length = 0;
  
  mu_message_size (msg, &bufsize);

  for (; bufsize > 1; bufsize >>= 1)
    if ((buf = malloc (bufsize)))
      break;

  if (!buf)
    {
      mu_error (_("cannot burst message: %s"), mu_strerror (ENOMEM));
      exit (1);
    }

  mu_message_get_stream (msg, &is);

  while (mu_stream_sequential_read (is, buf, bufsize, &n) == 0
	 && n > 0)
    {
      size_t start, i;
	
      for (i = start = 0; i < n; i++)
	{
	  int newstate = transtab[state-1][token_num(buf[i])];
	  
	  if (newstate < 0)
	    {
	      newstate = -newstate;
	      flush_stream (&os, buf + start, i - start);
	      start = i + 1;
	    }

	  if (state == S1)
	    {
	      /* GNU extension: check if we have seen enough dashes to
		 constitute a valid encapsulation boundary. */
	      if (newstate == S3)
		{
		  eb_length++;
		  if (eb_length < eb_min_length)
		    continue; /* Ignore state change */
		}
	      else if (eb_length)
		while (eb_length--)
		  flush_stream (&os, "-", 1);
	      eb_length = 0;
	    }
	  else if (state == S5 && newstate == S2)
	    {
	      /* As the automaton traverses from state S5 to S2, the
		 bursting agent should consider a new message started
		 and output the first character. */
	      os = NULL;
	      count++;
	    }
	  else if (state == S3 && newstate == S4)
	    {
	      /* As the automaton traverses from state S3 to S4, the
		 bursting agent should consider the current message ended. */
	      finish_stream (&os);
	    }
	  
	  state = newstate;
	}

      flush_stream (&os, buf + start, i - start);
    }

  free (buf);
  if (os)
    {
      if (count)
	finish_stream (&os);
      else
	{
	  mu_stream_close (os);
	  mu_stream_destroy (&os, mu_stream_get_owner (os));
	}
    }
  return count > 0;
}


int
burst_or_copy (mu_message_t msg, int recursive, int copy)
{
  if (recursive)
    {
      int mime = 0;
      mu_message_is_multipart (msg, &mime);
  
      if (mime)
	{
	  if (!map.first)
	    map.mime = 1;
	  return burst_mime (msg);
	}
      else if (burst_digest (msg))
	return 0;
    }

  if (copy)
    {
      int rc;
      
      if (map.mime)
	{
	  mu_header_t hdr;
	  char *value = NULL;
	  
	  mu_message_get_header (msg, &hdr);
	  if (mu_header_aget_value (hdr, MU_HEADER_CONTENT_TYPE, &value) == 0
	      && memcmp (value, "message/rfc822", 14) == 0)
	    {
	      mu_stream_t str;
	      mu_body_t body;
	      
	      mu_message_get_body (msg, &body);
	      mu_body_get_stream (body, &str);

	      msg = mh_stream_to_message (str);
	    }
	  free (value);
	}

      /* FIXME:
	 if (verbose && !inplace)
	   printf(_("message %lu of digest %lu becomes message %s"),
		   (unsigned long) (j+1),
		   (unsigned long) burst_map[i].msgno, to));
      */     

      rc = mu_mailbox_append_message (tmpbox, msg);
      if (rc)
	{
	  mu_error (_("cannot append message: %s"), mu_strerror (rc));
	  exit (1);
	}
      map.count++;
      return 0;
    }      
   
  return 1;
}

void
burst (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  memset (&map, 0, sizeof (map));
  mh_message_number (msg, &map.msgno);
  
  if (burst_or_copy (msg, 1, 0) == 0)
    {
      VERBOSE((ngettext ("%s message exploded from digest %s",
			 "%s messages exploded from digest %s",
			 (unsigned long) map.count),
	       mu_umaxtostr (0, map.count),
	       mu_umaxtostr (1, num)));
      if (inplace)
	{
	  obstack_grow (&stk, &map, sizeof map);
	  burst_count++;
	}
    }
  else if (!quiet)
    mu_error (_("message %s not in digest format"), mu_umaxtostr (0, num));
}


/* Inplace handling */

void
burst_rename (mh_msgset_t *ms, size_t lastuid)
{
  size_t i, j;

  VERBOSE ((_("Renaming messages")));
  j = burst_count - 1;
  for (i = ms->count; i > 0; i--)
    {
      const char *from;
      const char *to;

      if (ms->list[i-1] == burst_map[j].msgno)
	{
	  lastuid -= burst_map[j].count;
	  burst_map[j].msgno = lastuid;
	  j--;
	}

      if (ms->list[i-1] == lastuid)
	continue;
      
      from = mu_umaxtostr (0, ms->list[i-1]);
      to   = mu_umaxtostr (1, lastuid);
      --lastuid;

      VERBOSE((_("message %s becomes message %s"), from, to));
	       
      if (rename (from, to))
	{
	  mu_error (_("error renaming %s to %s: %s"),
		    from, to, mu_strerror (errno));
	  exit (1);
	}
    }
}  

void
msg_copy (size_t num, const char *file)
{
  mu_message_t msg;
  mu_attribute_t attr = NULL;
  mu_stream_t istream, ostream;
  int rc;
  size_t n;
  char buf[512];
  
  if ((rc = mu_file_stream_create (&ostream,
				   file,
				   MU_STREAM_WRITE|MU_STREAM_CREAT)) != 0
      || (rc = mu_stream_open (ostream)))
    {
      mu_error (_("Cannot open output file `%s': %s"),
		file, mu_strerror (rc));
      exit (1);
    }

  mu_mailbox_get_message (tmpbox, num, &msg);
  mu_message_get_stream (msg, &istream);

  while (rc == 0
	 && mu_stream_sequential_read (istream, buf, sizeof buf, &n) == 0
	 && n > 0)
    /* FIXME: Implement RFC 934 FSA? */
    rc = mu_stream_sequential_write (ostream, buf, n);
  
  mu_stream_close (ostream);
  mu_stream_destroy (&ostream, mu_stream_get_owner (ostream));

  /* Mark message as deleted */
  mu_message_get_attribute (msg, &attr);
  mu_attribute_set_deleted (attr);
}

void
finalize_inplace (size_t lastuid)
{
  size_t i;

  VERBOSE ((_("Moving bursted out messages in place")));

  for (i = 0; i < burst_count; i++)
    {
      size_t j;

      /* FIXME: toc handling */
      for (j = 0; j < burst_map[i].count; j++)
	{
	  const char *to = mu_umaxtostr (0, burst_map[i].msgno + 1 + j);
	  VERBOSE((_("message %s of digest %s becomes message %s"),
		   mu_umaxtostr (1, j + 1),
		   mu_umaxtostr (2, burst_map[i].msgno), to));
	  msg_copy (burst_map[i].first + j, to);
	}
    }
}

int
main (int argc, char **argv)
{
  int index, rc;
  mu_mailbox_t mbox;
  mh_msgset_t msgset;
  const char *tempfolder = mh_global_profile_get ("Temp-Folder", ".temp");
  
  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  argc -= index;
  argv += index;

  VERBOSE ((_("Opening folder `%s'"), mh_current_folder ()));
  mbox = mh_open_folder (mh_current_folder (), 0);
  mh_msgset_parse (mbox, &msgset, argc, argv, "cur");

  if (inplace)
    {
      size_t i, count;
      
      VERBOSE ((_("Opening temporary folder `%s'"), tempfolder));
      tmpbox = mh_open_folder (tempfolder, 1);
      VERBOSE ((_("Cleaning up temporary folder")));
      mu_mailbox_messages_count (tmpbox, &count);
      for (i = 1; i <= count; i++)
	{
	  mu_attribute_t attr = NULL;
	  mu_message_t msg = NULL;
	  mu_mailbox_get_message (tmpbox, i, &msg);
	  mu_message_get_attribute (msg, &attr);
	  mu_attribute_set_deleted (attr);
	}
      mu_mailbox_expunge (tmpbox);
      obstack_init (&stk);
    }
  else
    tmpbox = mbox;

  rc = mh_iterate (mbox, &msgset, burst, NULL);
  if (rc)
    return rc;

  if (inplace && burst_count)
    {
      mu_url_t dst_url = NULL;
      size_t i, next_uid, last_uid;
      mh_msgset_t ms;
      char *xargv[2];
      const char *dir;
      
      burst_map = obstack_finish (&stk);

      mu_mailbox_uidnext (mbox, &next_uid);
      for (i = 0, last_uid = next_uid-1; i < burst_count; i++)
	last_uid += burst_map[i].count;
      VERBOSE ((_("Estimated last UID: %s"), mu_umaxtostr (0, last_uid)));

      asprintf (&xargv[0], "%s-last", mu_umaxtostr (0, burst_map[0].msgno));
      xargv[1] = NULL;
      mh_msgset_parse (mbox, &ms, 1, xargv, NULL);
      free (xargv[0]);
      mh_msgset_uids (mbox, &ms);
	
      mu_mailbox_get_url (mbox, &dst_url);
      dir = mu_url_to_string (dst_url);
      VERBOSE ((_("changing to `%s'"), dir + 3));
      if (chdir (dir+3))
	{
	  mu_error (_("cannot change to `%s': %s"), dir, mu_strerror (errno));
	  exit (1);
	}
      mu_mailbox_close (mbox);

      burst_rename (&ms, last_uid);
      mh_msgset_free (&ms);

      finalize_inplace (last_uid);

      VERBOSE ((_("Expunging temporary folder")));
      mu_mailbox_expunge (tmpbox);
      mu_mailbox_close (tmpbox);
      mu_mailbox_destroy (&tmpbox);
    }
  else
    mu_mailbox_close (mbox);

  mu_mailbox_destroy (&mbox);
  VERBOSE ((_("Finished bursting")));
  return rc;
}

    

