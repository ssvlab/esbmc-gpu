/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2009,
   2010 Free Software Foundation, Inc.

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

/* Initialize MH applications. */

#include <mh.h>
#include <mailutils/url.h>
#include <mailutils/tls.h>
#include <pwd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>

char mh_list_format[] = 
  "%4(msg)"
  "%<(cur)+%| %>"
  "%<{replied}-%?{encrypted}E%| %>"
  "%02(mon{date})/%02(mday{date})"
  "%<{date} %|*%>"
  "%<(mymbox{from})%<{to}To:%14(decode(friendly{to}))%>%>"
  "%<(zero)%17(decode(friendly{from}))%>"
  "  %(decode{subject})%<{body}<<%{body}>>%>";

void
mh_init ()
{
  /* Register all mailbox and mailer formats */
  mu_register_all_formats ();
#ifdef WITH_TLS
  mu_init_tls_libs ();
#endif

  /* Read user's profile */
  mh_read_profile ();
}

void
mh_init2 ()
{
  mh_current_folder ();
  mh_global_sequences_get ("cur", NULL);
}

int
mh_read_formfile (char *name, char **pformat)
{
  FILE *fp;
  struct stat st;
  char *ptr;
  size_t off = 0;
  char *format_str;

  if (stat (name, &st))
    {
      mu_error (_("cannot stat format file %s: %s"), name, strerror (errno));
      return -1;
    }
  
  fp = fopen (name, "r");
  if (!fp)
    {
      mu_error (_("cannot open format file %s: %s"), name, strerror (errno));
      return -1;
    }

  format_str = xmalloc (st.st_size+1);
  while ((ptr = fgets (format_str + off, st.st_size - off + 1, fp)) != NULL)
    {
      int len = strlen (ptr);
      if (len == 0)
	break;

      if (*ptr == '%' && ptr[1] == ';')
	continue;
      
      if (len > 0 && ptr[len-1] == '\n')
	{
	  if (ptr[len-2] == '\\')
	    {
	      len -= 2;
	      ptr[len] = 0;
	    }
	}
      off += len;
    }
  if (off > 0 && format_str[off-1] == '\n')
    off--;
  format_str[off] = 0;
  fclose (fp);
  *pformat = format_str;
  return 0;
}

void
mh_err_memory (int fatal)
{
  mu_error (_("not enough memory"));
  if (fatal)
    abort ();
}

static char *my_name;
static char *my_email;

void
mh_get_my_name (char *name)
{
  if (!name)
    {
      struct passwd *pw = getpwuid (getuid ());
      if (!pw)
	{
	  mu_error (_("cannot determine my username"));
	  return;
	}
      name = pw->pw_name;
    }

  my_name = strdup (name);
  my_email = mu_get_user_email (name);
}

int
emailcmp (char *pattern, char *name)
{
  char *p;

  p = strchr (pattern, '@');
  if (p)
    for (p++; *p; p++)
      *p = mu_toupper (*p);

  return fnmatch (pattern, name, 0);
}

int
mh_is_my_name (const char *name)
{
  char *pname, *p;
  int rc = 0;
  
  pname = strdup (name);
  p = strchr (pname, '@');
  if (p)
    for (p++; *p; p++)
      *p = mu_toupper (*p);
  
  if (!my_email)
    mh_get_my_name (NULL);
  if (emailcmp (my_email, pname) == 0)
    rc = 1;
  else
    {
      const char *nlist = mh_global_profile_get ("Alternate-Mailboxes", NULL);
      if (nlist)
	{
	  const char *end, *p;
	  char *pat;
	  int len;
	  
	  for (p = nlist; rc == 0 && *p; p = end)
	    {
	      
	      while (*p && mu_isspace (*p))
		p++;

	      end = strchr (p, ',');
	      if (end)
		{
		  len = end - p;
		  end++;
		}
	      else
		{
		  len = strlen (p);
		  end = p + len;
		}

	      while (len > 0 && mu_isspace (p[len-1]))
		len--;

	      pat = xmalloc (len + 1);
	      memcpy (pat, p, len);
	      pat[len] = 0;
	      rc = emailcmp (pat, pname) == 0;
	      free (pat);
	    }
	}
    }
  free (pname);
  return rc;
}

char *
mh_my_email ()
{
  if (!my_email)
    mh_get_my_name (NULL);
  return my_email;
}

static int
make_dir_hier (const char *p, mode_t perm)
{
  int rc = 0;
  char *dir = xstrdup (p);
  char *q = dir;

  while (!rc && (q = strchr (q + 1, '/')))
    {
      *q = 0;
      if (access (dir, X_OK))
	{
	  if (errno != ENOENT)
	    {
	      mu_error (_("cannot create directory %s: error accessing name component %s: %s"),

			p, dir, strerror (errno));
	      rc = 1;
	    }
	  else if ((rc = mkdir (dir, perm)))
	    mu_error (_("cannot create directory %s: error creating name component %s: %s"),
		      p, dir, mu_strerror (rc));
	}
      *q = '/';
    }
  free (dir);
  return rc;
}

int
mh_makedir (const char *p)
{
  int rc;
  mode_t save_umask;
  mode_t perm = 0711;
  const char *pb = mh_global_profile_get ("Folder-Protect", NULL);
  if (pb)
    perm = strtoul (pb, NULL, 8);

  save_umask = umask (0);

  if ((rc = make_dir_hier (p, perm)) == 0)
    {
      rc = mkdir (p, perm);
      if (rc)
	mu_error (_("cannot create directory %s: %s"),
		  p, strerror (errno));
    }

  umask (save_umask);
  return rc;
}

int
mh_check_folder (const char *pathname, int confirm)
{
  const char *p;
  struct stat st;
  
  if ((p = strchr (pathname, ':')) != NULL)
    p++;
  else
    p = pathname;
  
  if (stat (p, &st))
    {
      if (errno == ENOENT)
	{
	  /* TRANSLATORS: This is a question and will be followed
	     by question mark on output. */
	  if (!confirm || mh_getyn (_("Create folder \"%s\""), p))
	    return mh_makedir (p);
	  else
	    return 1;
	}
      else
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "stat", p, errno);
	  return 1;
	}
    }
  return 0;
}

int
mh_interactive_mode_p ()
{
  static int interactive = -1;

  if (interactive < 0)
    interactive = isatty (fileno (stdin)) ? 1 : 0;
  return interactive;
}

int
mh_vgetyn (const char *fmt, va_list ap)
{
  char repl[64];

  while (1)
    {
      char *p;
      int len, rc;
      
      vfprintf (stdout, fmt, ap);
      fprintf (stdout, "? ");
      p = fgets (repl, sizeof repl, stdin);
      if (!p)
	return 0;
      len = strlen (p);
      if (len > 0 && p[len-1] == '\n')
	p[len--] = 0;

      rc = mu_true_answer_p (p);

      if (rc >= 0)
	return rc;

      /* TRANSLATORS: See msgids "nN" and "yY". */
      fprintf (stdout, _("Please answer yes or no: "));
    }
  return 0; /* to pacify gcc */
}

int
mh_getyn (const char *fmt, ...)
{
  va_list ap;
  int rc;
  
  if (!mh_interactive_mode_p ())
      return 1;
  va_start (ap, fmt);
  rc = mh_vgetyn (fmt, ap);
  va_end (ap);
  return rc;
}

int
mh_getyn_interactive (const char *fmt, ...)
{
  va_list ap;
  int rc;
  
  va_start (ap, fmt);
  rc = mh_vgetyn (fmt, ap);
  va_end (ap);
  return rc;
}
	    
FILE *
mh_audit_open (char *name, mu_mailbox_t mbox)
{
  FILE *fp;
  char date[64];
  time_t t;
  struct tm *tm;
  mu_url_t url;
  char *namep;
  
  namep = mu_tilde_expansion (name, "/", NULL);
  if (strchr (namep, '/') == NULL)
    {
      char *p = NULL;

      asprintf (&p, "%s/%s", mu_folder_directory (), namep);
      if (!p)
	{
	  mu_error (_("not enough memory"));
	  exit (1);
	}
      free (namep);
      namep = p;
    }

  fp = fopen (namep, "a");
  if (!fp)
    {
      mu_error (_("cannot open audit file %s: %s"), namep, strerror (errno));
      free (namep);
      return NULL;
    }
  free (namep);
  
  time (&t);
  tm = localtime (&t);
  mu_strftime (date, sizeof date, "%a, %d %b %Y %H:%M:%S %Z", tm);
  mu_mailbox_get_url (mbox, &url);
  
  fprintf (fp, "<<%s>> %s %s\n",
	   mu_program_name,
	   date,
	   mu_url_to_string (url));
  return fp;
}

void
mh_audit_close (FILE *fp)
{
  fclose (fp);
}

int
mh_message_number (mu_message_t msg, size_t *pnum)
{
  return mu_message_get_uid (msg, pnum);	
}

mu_mailbox_t
mh_open_folder (const char *folder, int create)
{
  mu_mailbox_t mbox = NULL;
  char *name;
  int flags = MU_STREAM_RDWR;
  
  name = mh_expand_name (NULL, folder, 1);
  if (create && mh_check_folder (name, 1))
    exit (0);
    
  if (mu_mailbox_create_default (&mbox, name))
    {
      mu_error (_("cannot create mailbox %s: %s"),
		name, strerror (errno));
      exit (1);
    }

  if (create)
    flags |= MU_STREAM_CREAT;
  
  if (mu_mailbox_open (mbox, flags))
    {
      mu_error (_("cannot open mailbox %s: %s"), name, strerror (errno));
      exit (1);
    }

  free (name);

  return mbox;
}

char *
mh_get_dir ()
{
  const char *mhdir = mh_global_profile_get ("Path", "Mail");
  char *mhcopy;
  
  if (mhdir[0] != '/')
    {
      char *p = mu_get_homedir ();
      asprintf (&mhcopy, "%s/%s", p, mhdir);
      free (p);
    }
  else 
    mhcopy = strdup (mhdir);
  return mhcopy;
}

char *
mh_expand_name (const char *base, const char *name, int is_folder)
{
  char *p = NULL;
  char *namep = NULL;
  
  namep = mu_tilde_expansion (name, "/", NULL);
  if (namep[0] == '+')
    memmove (namep, namep + 1, strlen (namep)); /* copy null byte as well */
  else if (strncmp (namep, "../", 3) == 0 || strncmp (namep, "./", 2) == 0)
    {
      char *cwd = mu_getcwd ();
      char *tmp = NULL;
      asprintf (&tmp, "%s/%s", cwd, namep);
      free (cwd);
      free (namep);
      namep = tmp;
    }
  
  if (is_folder)
    {
      if (memcmp (namep, "mh:/", 4) == 0)
	return namep;
      else if (namep[0] == '/')
	asprintf (&p, "mh:%s", namep);
      else
	asprintf (&p, "mh:%s/%s", base ? base : mu_folder_directory (), namep);
    }
  else if (namep[0] != '/')
    asprintf (&p, "%s/%s", base ? base : mu_folder_directory (), namep);
  else
    return namep;
  
  free (namep);
  return p;
}

int
mh_iterate (mu_mailbox_t mbox, mh_msgset_t *msgset,
	    mh_iterator_fp itr, void *data)
{
  int rc;
  size_t i;

  for (i = 0; i < msgset->count; i++)
    {
      mu_message_t msg;
      size_t num;

      num = msgset->list[i];
      if ((rc = mu_mailbox_get_message (mbox, num, &msg)) != 0)
	{
	  mu_error (_("cannot get message %lu: %s"),
		    (unsigned long) num, mu_strerror (rc));
	  return 1;
	}

      itr (mbox, msg, num, data);
    }

  return 0;
}

int
mh_spawnp (const char *prog, const char *file)
{
  int argc, i, rc, status;
  char **argv, **xargv;

  if ((rc = mu_argcv_get (prog, "", "#", &argc, &argv)) != 0)
    {
      mu_diag_funcall (MU_DIAG_ERROR, "mu_argcv_get", prog, rc);
      mu_argcv_free (argc, argv);
      return 1;
    }

  xargv = calloc (argc + 2, sizeof (*xargv));
  if (!xargv)
    {
      mh_err_memory (0);
      mu_argcv_free (argc, argv);
      return 1;
    }

  for (i = 0; i < argc; i++)
    xargv[i] = argv[i];
  xargv[i++] = (char*) file;
  xargv[i++] = NULL;

  rc = mu_spawnvp (xargv[0], xargv, &status);

  free (xargv);
  mu_argcv_free (argc, argv);

  return rc;
}

int
mh_file_copy (const char *from, const char *to)
{
  char *buffer;
  size_t bufsize, rdsize;
  struct stat st;
  mu_stream_t in;
  mu_stream_t out;
  int rc;
  
  if (stat (from, &st))
    {
      mu_error ("mh_copy: %s", mu_strerror (errno));
      return -1;
    }

  for (bufsize = st.st_size; bufsize > 0 && (buffer = malloc (bufsize)) == 0;
       bufsize /= 2)
    ;

  if (!bufsize)
    mh_err_memory (1);

  if ((rc = mu_file_stream_create (&in, from, MU_STREAM_READ)) != 0
      || (rc = mu_stream_open (in)))
    {
      mu_error (_("cannot open input file `%s': %s"),
		from, mu_strerror (rc));
      free (buffer);
      return 1;
    }

  if ((rc = mu_file_stream_create (&out, to, MU_STREAM_RDWR|MU_STREAM_CREAT)) != 0
      || (rc = mu_stream_open (out)))
    {
      mu_error (_("cannot open output file `%s': %s"),
		to, mu_strerror (rc));
      free (buffer);
      mu_stream_close (in);
      mu_stream_destroy (&in, mu_stream_get_owner (in));
      return 1;
    }

  while (st.st_size > 0
	 && (rc = mu_stream_sequential_read (in, buffer, bufsize, &rdsize)) == 0
	 && rdsize > 0)
    {
      if ((rc = mu_stream_sequential_write (out, buffer, rdsize)) != 0)
	{
	  mu_error (_("write error on `%s': %s"),
		    to, mu_strerror (rc));
	  break;
	}
      st.st_size -= rdsize;
    }

  free (buffer);

  mu_stream_close (in);
  mu_stream_close (out);
  mu_stream_destroy (&in, mu_stream_get_owner (in));
  mu_stream_destroy (&out, mu_stream_get_owner (out));
  
  return rc;
}

static mu_message_t
_file_to_message (const char *file_name)
{
  struct stat st;
  int rc;
  mu_stream_t instream;

  if (stat (file_name, &st) < 0)
    {
      mu_diag_funcall (MU_DIAG_ERROR, "stat", file_name, errno);
      return NULL;
    }
  
  if ((rc = mu_file_stream_create (&instream, file_name, MU_STREAM_READ)))
    {
      mu_error (_("cannot create input stream (file %s): %s"),
		file_name, mu_strerror (rc));
      return NULL;
    }
  
  if ((rc = mu_stream_open (instream)))
    {
      mu_error (_("cannot open input stream (file %s): %s"),
		file_name, mu_strerror (rc));
      mu_stream_destroy (&instream, mu_stream_get_owner (instream));
      return NULL;
    }

  return mh_stream_to_message (instream);
}

mu_message_t
mh_file_to_message (const char *folder, const char *file_name)
{
  mu_message_t msg;
  char *tmp_name = NULL;
  
  if (folder)
    {
      tmp_name = mh_expand_name (folder, file_name, 0);
      msg = _file_to_message (tmp_name);
      free (tmp_name);
    }
  else
    msg = _file_to_message (file_name);
  
  return msg;
}

void
mh_install_help (char *mhdir)
{
  static char *text = N_(
"Prior to using MH, it is necessary to have a file in your login\n"
"directory (%s) named .mh_profile which contains information\n"
"to direct certain MH operations.  The only item which is required\n"
"is the path to use for all MH folder operations.  The suggested MH\n"
"path for you is %s...\n");

  printf (_(text), mu_get_homedir (), mhdir);
}

void
mh_real_install (char *name, int automode)
{
  char *home = mu_get_homedir ();
  char *mhdir;
  char *ctx;
  FILE *fp;
  
  asprintf (&mhdir, "%s/%s", home, "Mail");
  
  if (!automode)
    {
      size_t n = 0;
      
      /* TRANSLATORS: This is a question and will be followed
	 by question mark on output. */
      if (mh_getyn_interactive (_("Do you need help")))
	mh_install_help (mhdir);

      /* TRANSLATORS: This is a question and will be followed
	 by question mark on output. */
      if (!mh_getyn_interactive (_("Do you want the standard MH path \"%s\""), mhdir))
	{
	  int local;
	  char *p;
	  
	  /* TRANSLATORS: This is a question and will be followed
	     by question mark on output. */
	  local = mh_getyn_interactive (_("Do you want a path below your login directory"));
	  if (local)
	    printf (_("What is the path? "));
	  else
	    printf (_("What is the full path? "));
	  if (getline (&p, &n, stdin) <= 0)
	    exit (1);

	  n = strlen (p);
	  if (n == 0)
	    exit (1);

	  if (p[n-1] == '\n')
	    p[n-1] = 0;

	  free (mhdir);
	  if (local)
	    {
	      asprintf (&mhdir, "%s/%s", home, p);
	      free (p);
	    }
	  else
	    mhdir = p;
	}
    }

  if (mh_check_folder (mhdir, !automode))
    exit (1);

  fp = fopen (name, "w");
  if (!fp)
    {
      mu_error (_("cannot open file %s: %s"), name, mu_strerror (errno));
      exit (1);
    }
  fprintf (fp, "Path: %s\n", mhdir);
  fclose (fp);

  asprintf (&ctx, "%s/%s", mhdir, MH_CONTEXT_FILE);
  fp = fopen (ctx, "w");
  if (fp)
    {
      fprintf (fp, "Current-Folder: inbox\n");
      fclose (fp);
    }
  free (ctx);
  asprintf (&ctx, "%s/inbox", mhdir);
  if (mh_check_folder (ctx, !automode))
    exit (1);
  free (ctx);
  free (mhdir);
}  

void
mh_install (char *name, int automode)
{
  struct stat st;
  
  if (stat(name, &st))
    {
      if (errno == ENOENT)
	{
	  if (automode)
	    printf(_("I'm going to create the standard MH path for you.\n"));
	  mh_real_install (name, automode);
	}
      else
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "stat", name, errno);
	  exit (1);
	}
    }
  else if ((st.st_mode & S_IFREG) || (st.st_mode & S_IFLNK)) 
    {
      mu_error(_("You already have an MH profile, use an editor to modify it"));
      exit (0);
    }
  else
    {
      mu_error(_("You already have file %s which is not a regular file or a symbolic link.\n"
		 "Please remove it and try again"),
	       name);
      exit (1);
    }
}
        
void
mh_annotate (mu_message_t msg, char *field, char *text, int date)
{
  mu_header_t hdr;
  mu_attribute_t attr;
  
  if (mu_message_get_header (msg, &hdr))
    return;

  if (date)
    {
      time_t t;
      struct tm *tm;
      char datebuf[80];
      t = time (NULL);
      tm = localtime (&t);
      mu_strftime (datebuf, sizeof datebuf, "%a, %d %b %Y %H:%M:%S %Z", tm);

      mu_header_set_value (hdr, field, datebuf, 0);
    }

  if (text)
    mu_header_set_value (hdr, field, text, 0);
  mu_message_get_attribute (msg, &attr);
  mu_attribute_set_modified (attr);
}

char *
mh_draft_name ()
{
  return mh_expand_name (mh_global_profile_get ("Draft-Folder",
						mu_folder_directory ()),
			 "draft", 0);
}

char *
mh_create_message_id (int subpart)
{
  char *p;
  mu_rfc2822_msg_id (subpart, &p);
  return p;
}

void
mh_set_reply_regex (const char *str)
{
  char *err;
  int rc = mu_unre_set_regex (str, 0, &err);
  if (rc)
    mu_error ("reply_regex: %s%s%s", mu_strerror (rc),
	      err ? ": " : "",
	      err ? err : "");
}

const char *
mh_charset (const char *dfl)
{
  const char *charset = mh_global_profile_get ("Charset", dfl);

  if (!charset)
    return NULL;
  if (mu_c_strcasecmp (charset, "auto") == 0)
    {
      /* Try to deduce the charset from LC_ALL variable */
      
      char *lc_all = getenv ("LC_ALL");
      if (lc_all)
	{
	  char *sp;
	  char *lang;
	  char *terr;

	  char *tmp = strdup (lc_all);
	  lang = strtok_r (tmp, "_", &sp);
	  terr = strtok_r (NULL, ".", &sp);
	  charset = strtok_r (NULL, "@", &sp);

	  if (!charset)
	    charset = mu_charset_lookup (lang, terr);
	  
	  free (tmp);
	}
    }
  return charset;
}

int
mh_decode_2047 (char *text, char **decoded_text)
{
  const char *charset = mh_charset (NULL);
  if (!charset)
    return 1;
  
  return mu_rfc2047_decode (charset, text, decoded_text);
}

void
mh_quote (const char *in, char **out)
{
  size_t len = strlen (in);
  if (len && in[0] == '"' && in[len - 1] == '"')
    {
      const char *p;
      char *q;
      
      for (p = in + 1; p < in + len - 1; p++)
        if (*p == '\\' || *p == '"')
	  len++;

      *out = xmalloc (len + 1);
      q = *out;
      p = in;
      *q++ = *p++;
      while (p[1])
	{
	  if (*p == '\\' || *p == '"')
	    *q++ = '\\';
	  *q++ = *p++;
	}
      *q++ = *p++;
      *q = 0;
    }
  else
    *out = xstrdup (in);
}

void
mh_expand_aliases (mu_message_t msg,
		   mu_address_t *addr_to,
		   mu_address_t *addr_cc,
		   mu_address_t *addr_bcc)
{
  mu_header_t hdr;
  size_t i, num;
  const char *buf;
  
  mu_message_get_header (msg, &hdr);
  mu_header_get_field_count (hdr, &num);
  for (i = 1; i <= num; i++)
    {
      if (mu_header_sget_field_name (hdr, i, &buf) == 0)
	{
	  if (mu_c_strcasecmp (buf, MU_HEADER_TO) == 0
	      || mu_c_strcasecmp (buf, MU_HEADER_CC) == 0
	      || mu_c_strcasecmp (buf, MU_HEADER_BCC) == 0)
	    {
	      char *value;
	      mu_address_t addr = NULL;
	      int incl;
	      
	      mu_header_aget_field_value_unfold (hdr, i, &value);
	      
	      mh_alias_expand (value, &addr, &incl);
	      free (value);
	      if (mu_c_strcasecmp (buf, MU_HEADER_TO) == 0)
		mu_address_union (addr_to, addr);
	      else if (mu_c_strcasecmp (buf, MU_HEADER_CC) == 0)
		mu_address_union (addr_cc ? addr_cc : addr_to, addr);
	      else if (mu_c_strcasecmp (buf, MU_HEADER_BCC) == 0)
		mu_address_union (addr_bcc ? addr_bcc : addr_to, addr);
	    }
	}
    }
}

int
mh_draft_message (const char *name, const char *msgspec, char **pname)
{
  mu_url_t url;
  size_t uid;
  int rc;
  const char *urlstr;
  mu_mailbox_t mbox;

  mbox = mh_open_folder (name, 0);
  if (!mbox)
    return 1;
  
  mu_mailbox_get_url (mbox, &url);
  urlstr = mu_url_to_string (url);

  if (strcmp (msgspec, "new") == 0)
    {
      rc = mu_mailbox_uidnext (mbox, &uid);
      if (rc)
	mu_error (_("cannot obtain sequence number for the new message: %s"),
		  mu_strerror (rc));
    }
  else
    {
      char *argv[2];
      mh_msgset_t msgset;
      
      argv[0] = (char*) msgspec;
      argv[1] = NULL;
      rc = mh_msgset_parse (mbox, &msgset, 1, argv, "cur");
      if (rc)
	mu_error (_("invalid message number: %s"), msgspec);
      else if (msgset.count > 1)
	mu_error (_("only one message at a time!"));
      else
	uid = msgset.list[0];

      mh_msgset_free (&msgset);
    }
  
  if (rc == 0)
    {
      const char *dir;
      const char *msg;
      size_t len;
      
      dir = urlstr + 3; /* FIXME */
      
      msg = mu_umaxtostr (0, uid);
      len = strlen (dir) + 1 + strlen (msg) + 1;
      *pname = xmalloc (len);
      strcpy (*pname, dir);
      strcat (*pname, "/");
      strcat (*pname, msg);
    }
  mu_mailbox_close (mbox);
  mu_mailbox_destroy (&mbox);
  return rc;
}
