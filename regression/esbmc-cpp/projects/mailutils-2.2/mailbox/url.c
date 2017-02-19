/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2007, 2008, 2009, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/mutil.h>
#include <mailutils/errno.h>
#include <mailutils/argcv.h>
#include <mailutils/secret.h>
#include <mailutils/cctype.h>
#include <mailutils/cstr.h>
#include <url0.h>

#define AC2(a,b) a ## b
#define AC4(a,b,c,d) a ## b ## c ## d

static int url_parse0 (mu_url_t, char *, size_t *poff);

static int
parse_query (const char *query,
	     char *delim,
	     int *pargc, char ***pargv, const char **pend)
{
  size_t count, i;
  char **v;
  const char *p;

  for (p = query, count = 0; ; count++)
    {
      size_t len = strcspn (p, delim);
      p += len;
      if (!*p || *p == delim[1])
	break;
      p++;
    }

  if (pend)
    *pend = p;
  if (p == query)
    return 0;
  count++;

  v = calloc (count + 1, sizeof (v[0]));
  for (i = 0, p = query; i < count; i++)
    {
      size_t len = strcspn (p, delim);
      v[i] = mu_url_decode_len (p, len);
      if (v[i] == NULL)
	{
	  mu_argcv_free (i, v);
	  return 1;
	}
      p += len + 1;
    }
  v[i] = NULL;

  *pargc = count;
  *pargv = v;
  return 0;
}

int
mu_url_create (mu_url_t *purl, const char *name)
{
  mu_url_t url = calloc (1, sizeof (*url));
  if (url == NULL)
    return ENOMEM;

  url->name = strdup (name);
  if (url->name == NULL)
    {
      free (url);
      return ENOMEM;
    }
  *purl = url;
  return 0;
}

static char **
argcv_copy (size_t argc, char **argv)
{
  size_t i;
  char **nv = calloc (argc + 1, sizeof (nv[0]));
  if (!nv)
    return NULL;
  for (i = 0; i < argc; i++)
    if ((nv[i] = strdup (argv[i])) == NULL)
      {
	mu_argcv_free (i, nv);
	free (nv);
	return NULL;
      }
  return nv;
}

static int
mu_url_copy0 (mu_url_t old_url, mu_url_t new_url)
{
  const char *str;
  size_t argc;
  char **argv;
  int rc;
  mu_secret_t sec;

#define URLCOPY(what)						\
  do								\
    {								\
      rc = AC2(mu_url_sget_,what) (old_url, &str);		\
      if (rc == 0)						\
	{							\
	  if ((new_url->what = strdup (str)) == NULL)		\
	    return ENOMEM;					\
	}							\
      else if (rc != MU_ERR_NOENT)				\
	return rc;						\
    }								\
  while (0);

  URLCOPY (scheme);
  URLCOPY (user);

  rc = mu_url_get_secret (old_url, &sec);
  if (rc == MU_ERR_NOENT)
    new_url->secret = NULL;
  else if (rc)
    return rc;
  else
    {
      rc = mu_secret_dup (sec, &new_url->secret);
      if (rc)
	return rc;
    }
  
  URLCOPY (auth);
  URLCOPY (host);
  new_url->port = old_url->port;
  URLCOPY (path);

  rc = mu_url_sget_fvpairs (old_url, &argc, &argv);
  if (rc == 0 && argc)
    {
      if ((new_url->fvpairs = argcv_copy (argc, argv)) == NULL)
	return ENOMEM;
      new_url->fvcount = argc;
    }

  rc = mu_url_sget_query (old_url, &argc, &argv);
  if (rc == 0 && argc)
    {
      if ((new_url->qargv = argcv_copy (argc, argv)) == NULL)
	return ENOMEM;
      new_url->qargc = argc;
    }
  return 0;
#undef URLCOPY
}

int
mu_url_dup (mu_url_t old_url, mu_url_t *new_url)
{
  mu_url_t url;
  int rc = mu_url_create (&url, mu_url_to_string (old_url));

  if (rc)
    return rc;

  rc = mu_url_copy0 (old_url, url);
  if (rc == 0)
    *new_url = url;
  else
    mu_url_destroy (&url);
  return rc;
}

int
mu_url_uplevel (mu_url_t url, mu_url_t *upurl)
{
  int rc;
  char *p;
  mu_url_t new_url;

  if (url->_uplevel)
    return url->_uplevel (url, upurl);

  if (!url->path)
    return MU_ERR_NOENT;
  p = strrchr (url->path, '/');

  rc = mu_url_dup (url, &new_url);
  if (rc == 0)
    {
      if (!p || p == url->path)
	{
	  free (new_url->path);
	  new_url->path = NULL;
	}
      else
	{
	  size_t size = p - url->path;
	  new_url->path = realloc (new_url->path, size + 1);
	  if (!new_url->path)
	    {
	      mu_url_destroy (&new_url);
	      return ENOMEM;
	    }
	  memcpy (new_url->path, url->path, size);
	  new_url->path[size] = 0;
	}
      *upurl = new_url;
    }
  return rc;
}

void
mu_url_destroy (mu_url_t * purl)
{
  if (purl && *purl)
    {
      mu_url_t url = (*purl);

      if (url->_destroy)
	url->_destroy (url);

      if (url->name)
	free (url->name);

      if (url->scheme)
	free (url->scheme);

      if (url->user)
	free (url->user);

      mu_secret_destroy (&url->secret);

      if (url->auth)
	free (url->auth);

      if (url->host)
	free (url->host);

      if (url->path)
	free (url->path);

      if (url->fvcount)
	mu_argcv_free (url->fvcount, url->fvpairs);

      mu_argcv_free (url->qargc, url->qargv);

      free (url);

      *purl = NULL;
    }
}

int
mu_url_parse (mu_url_t url)
{
  int err = 0;
  char *n = NULL;
  struct _mu_url u;
  size_t pstart;
  mu_secret_t newsec;

  if (!url || !url->name)
    return EINVAL;

  memset (&u, 0, sizeof u);
  /* can't have been parsed already */
  if (url->scheme || url->user || url->secret || url->auth ||
      url->host || url->path || url->qargc)
    return EINVAL;

  n = strdup (url->name);

  if (!n)
    return ENOMEM;

  err = url_parse0 (&u, n, &pstart);

  if (!err)
    {
      if (u.secret)
	{
	  /* Obfuscate the password */
#define PASS_REPL "***"
#define PASS_REPL_LEN (sizeof (PASS_REPL) - 1)
	  size_t plen = mu_secret_length (u.secret);
	  size_t nlen = strlen (url->name);
	  size_t len = nlen - plen + PASS_REPL_LEN + 1;
	  char *newname;

	  memset (url->name + pstart, 0, plen);
	  newname = realloc (url->name, len);
	  if (!newname)
	    goto CLEANUP;
	  memmove (newname + pstart + PASS_REPL_LEN, newname + pstart + plen,
		   nlen - (pstart + plen) + 1);
	  memcpy (newname + pstart, PASS_REPL, PASS_REPL_LEN);
	  url->name = newname;
	}

      /* Dup the strings we found. We wouldn't have to do this
	 if we did a single alloc of the source url name, and
	 kept it around. It's also a good time to do hex decoding,
	 though.
       */

#define UALLOC(X)                                          \
  if (u.X && u.X[0] && (url->X = mu_url_decode(u.X)) == 0) \
    {                                                      \
       err = ENOMEM;                                       \
       goto CLEANUP;                                       \
    }                                                      \
  else                                                     \
    {                                                      \
       /* Set zero-length strings to NULL. */              \
	u.X = NULL; \
    }

      UALLOC (scheme);
      UALLOC (user);

      if (u.secret)
	{
	  char *pass = mu_url_decode (mu_secret_password (u.secret));
	  err = mu_secret_create (&newsec, pass, strlen (pass));
	  memset (pass, 0, strlen (pass));
	  mu_secret_destroy (&u.secret);
	  if (err)
	    goto CLEANUP;

	  url->secret = newsec;
	}

      UALLOC (auth);
      UALLOC (host);
      UALLOC (path);

#undef UALLOC
      url->fvcount = u.fvcount;
      url->fvpairs = u.fvpairs;

      url->qargc = u.qargc;
      url->qargv = u.qargv;

      url->port = u.port;
    }

CLEANUP:
  memset (n, 0, strlen (n));
  free (n);

  if (err)
    {
#define UFREE(X) if (X) { free(X); X = 0; }

      UFREE (url->scheme);
      UFREE (url->user);
      mu_secret_destroy (&u.secret);
      UFREE (url->auth);
      UFREE (url->host);
      UFREE (url->path);
      mu_argcv_free (url->fvcount, url->fvpairs);
      mu_argcv_free (url->qargc, url->qargv);
#undef UFREE
    }

  return err;
}

/*

Syntax, condensed from RFC 1738, and extended with the ;auth=
of RFC 2384 (for POP) and RFC 2192 (for IMAP):

url =
    scheme ":" [ "//"

    [ user [ ( ":" password ) | ( ";auth=" auth ) ] "@" ]

    host [ ":" port ]

    [ ( "/" urlpath ) | ( "?" query ) ] ]

All hell will break loose in this parser if the user/pass/auth
portion is missing, and the urlpath has any @ or : characters
in it. A imap mailbox, say, named after the email address of
the person the mail is from:

  imap://imap.uniserve.com/alain@qnx.com

Is this required to be % quoted, though? I hope so!

*/

static int
url_parse0 (mu_url_t u, char *name, size_t *poff)
{
  char *start = name;
  char *p;			/* pointer into name */

  /* reject the obvious */
  if (name == NULL)
    return EINVAL;

  if (name[0] == '/')
    {
      u->scheme = "file";
    }
  else if (name[0] == '|')
    {
      int rc;
      u->scheme = "prog";
      rc = mu_argcv_get (name + 1, NULL, NULL, &u->qargc, &u->qargv);
      if (rc == 0)
	{
	  u->path = strdup (u->qargv[0]);
	  if (!u->path)
	    rc = ENOMEM;
	}
      return rc;
    }
  else
    {
      /* Parse out the SCHEME. */
      p = strchr (name, ':');
      if (p == NULL)
	return MU_ERR_PARSE;

      *p++ = 0;

      u->scheme = name;

      /* RFC 1738, section 2.1, lower the scheme case */
      for (; name < p; name++)
	*name = mu_tolower (*name);

      name = p;
    }

  /* Check for nothing following the scheme. */
  if (!*name)
    return 0;

  if (strncmp (name, "//", 2) == 0)
    {
      name += 2;

      if (name[0] == '/')
	{
	  u->path = name;
	  p = u->path + strcspn (u->path, ";?");
	}
      else
	{
	  /* Split into LHS and RHS of the '@', and then parse each side. */
	  u->host = strchr (name, '@');
	  if (u->host == NULL)
	    u->host = name;
	  else
	    {
	      char *pass = NULL;

	      /* Parse the LHS into an identification/authentication pair. */
	      *u->host++ = 0;

	      u->user = name;

	      /* Try to split the user into a:
		 <user>:<password>
		 or
		 <user>:<password>;AUTH=<auth>
	      */

	      for (; *name; name++)
		{
		  if (*name == ':')
		    {
		      *name++ = 0;
		      pass = name;
		      *poff = pass - start;
		    }
		  else if (*name == ';')
		    {
		      /* Make sure it's the auth token. */
		      if (mu_c_strncasecmp (name + 1, "auth=", 5) == 0)
			{
			  *name++ = 0;
			  name += 5;
			  u->auth = name;
			  break;
			}
		    }
		}

	      if (pass)
		{
		  if (mu_secret_create (&u->secret, pass, strlen (pass)))
		    return ENOMEM;
		  else
		    /* Obfuscate password */
		    memset (pass, 0, strlen (pass));
		}
	    }

	  /* Parse the host and port from the RHS. */
	  p = strchr (u->host, ':');
	  if (p)
	    {
	      *p++ = 0;
	      u->port = strtol (p, &p, 10);

	      /* Check for garbage after the port: we should be on the start
		 of a path, a query, or at the end of the string. */
	      if (*p && strcspn (p, "/?") != 0)
		return MU_ERR_PARSE;
	    }
	  else
	    p = u->host + strcspn (u->host, ";/?");
	}
    }
  else
    {
      u->path = name;
      p = u->path + strcspn (u->path, ";?");
    }

  /* Either way, if we're not at a nul, we're at a path or query. */
  if (u->path == NULL && *p == '/')
    {
      /* found a path */
      *p++ = 0;
      u->path = p;
      p = u->path + strcspn (u->path, ";?");
    }

  if (*p == ';')
    {
      *p++ = 0;
      if (parse_query (p, ";?", &u->fvcount, &u->fvpairs, (const char **)&p))
	return ENOMEM;
    }

  if (*p == '?')
    {
      /* found a query */
      *p++ = 0;
      if (parse_query (p, "&", &u->qargc, &u->qargv, NULL))
	return ENOMEM;
    }

  return 0;
}


/* General accessors: */
#define ACCESSOR(action,field) AC4(mu_url_,action,_,field)

#define DECL_SGET(field)						  \
int									  \
ACCESSOR(sget,field) (mu_url_t url, char const **sptr)	                  \
{									  \
  if (url == NULL)							  \
    return EINVAL;							  \
  if (!url->field)							  \
    {									  \
      if (url->AC2(_get_,field))					  \
	{								  \
	  size_t n;							  \
	  char *buf;							  \
									  \
	  int status = url->AC2(_get_,field) (url, NULL, 0, &n);	  \
	  if (status)							  \
	    return status;						  \
									  \
	  buf = malloc (n + 1);						  \
	  if (!buf)							  \
	    return ENOMEM;						  \
									  \
	  status = url->AC2(_get_,field) (url, buf, n + 1, NULL);	  \
	  if (status)				          \
	    return status;						  \
									  \
	  if (buf[0])                                                     \
	    {                                                             \
	       url->field = mu_url_decode (buf);			  \
	       free (buf);						  \
	    }                                                             \
	  else                                                            \
	    url->field = buf;                                             \
	  if (!url->field)						  \
	    return ENOMEM;						  \
	}								  \
      else								  \
	return MU_ERR_NOENT;			                          \
    }									  \
  *sptr = url->field;							  \
  return 0;								  \
}

#define DECL_GET(field)							  \
int									  \
ACCESSOR(get,field) (mu_url_t url, char *buf, size_t len, size_t *n)      \
{									  \
  size_t i;								  \
  const char *str;							  \
  int status = ACCESSOR(sget, field) (url, &str);			  \
									  \
  if (status)								  \
    return status;							  \
									  \
  i = mu_cpystr (buf, str, len);					  \
  if (n)								  \
    *n = i;								  \
  return 0;								  \
}

#define DECL_AGET(field)						  \
int									  \
ACCESSOR(aget, field) (mu_url_t url, char **buf)	                  \
{									  \
  const char *str;							  \
  int status = ACCESSOR(sget, field) (url, &str);			  \
									  \
  if (status)								  \
    return status;							  \
									  \
  if (str)								  \
    {									  \
      *buf = strdup (str);						  \
      if (!*buf)							  \
	status = ENOMEM;						  \
    }									  \
  else									  \
    *buf = NULL;							  \
  return status;							  \
}

#define DECL_CMP(field)							  \
int									  \
ACCESSOR(is_same,field) (mu_url_t url1, mu_url_t url2)		          \
{									  \
  const char *s1, *s2;							  \
  int status1, status2;							  \
									  \
  status1 = ACCESSOR(sget, field) (url1, &s1);				  \
  if (status1 && status1 != MU_ERR_NOENT)				  \
    return 0;								  \
  status2 = ACCESSOR(sget, field) (url2, &s2);				  \
  if (status2 && status2 != MU_ERR_NOENT)				  \
    return 0;								  \
									  \
  if (status1 && status1 == status2) /* Both fields are missing */	  \
    return 1;								  \
  return mu_c_strcasecmp (s1, s2) == 0;					  \
}

#define DECL_ACCESSORS(field)			                          \
DECL_SGET(field)				                          \
DECL_GET(field)					                          \
DECL_AGET(field)                                                          \
DECL_CMP(field)


/* Declare particular accessors */
DECL_ACCESSORS (scheme)
DECL_ACCESSORS (user)
DECL_ACCESSORS (auth)
DECL_ACCESSORS (host)
DECL_ACCESSORS (path)

int
mu_url_get_secret (const mu_url_t url, mu_secret_t *psecret)
{
  if (url->_get_secret)
    return url->_get_secret (url, psecret);
  if (url->secret == NULL)
    return MU_ERR_NOENT;
  mu_secret_ref (url->secret);
  *psecret = url->secret;
  return 0;
}

int
mu_url_sget_query (const mu_url_t url, size_t *qc, char ***qv)
{
  if (url == NULL)
    return EINVAL;
  /* See FIXME below */
  *qc = url->qargc;
  *qv = url->qargv;
  return 0;
}

int
mu_url_aget_query (const mu_url_t url, size_t *qc, char ***qv)
{
  size_t qargc, i;
  char **qargv;
  char **qcopy;

  int rc = mu_url_sget_fvpairs (url, &qargc, &qargv);
  if (rc)
    return rc;

  qcopy = calloc (qargc + 1, sizeof (qcopy[0]));
  if (!qcopy)
    return errno;
  for (i = 0; i < qargc; i++)
    {
      if (!(qcopy[i] = strdup (qargv[i])))
	{
	  mu_argcv_free (i, qcopy);
	  return errno;
	}
    }
  qcopy[i] = NULL;
  *qc = qargc;
  *qv = qcopy;
  return 0;
}

/* field-value pairs accessors */
int
mu_url_sget_fvpairs (const mu_url_t url, size_t *fvc, char ***fvp)
{
  if (url == NULL)
    return EINVAL;
  /* FIXME: no _get_fvpairs method, but the method stuff needs to be rewritten
     anyway */
  *fvc = url->fvcount;
  *fvp = url->fvpairs;
  return 0;
}

int
mu_url_aget_fvpairs (const mu_url_t url, size_t *pfvc, char ***pfvp)
{
  size_t fvc, i;
  char **fvp;
  char **fvcopy;

  int rc = mu_url_sget_fvpairs (url, &fvc, &fvp);
  if (rc)
    return rc;

  fvcopy = calloc (fvc + 1, sizeof (fvcopy[0]));
  if (!fvcopy)
    return errno;
  for (i = 0; i < fvc; i++)
    {
      if (!(fvcopy[i] = strdup (fvp[i])))
	{
	  mu_argcv_free (i, fvcopy);
	  return errno;
	}
    }
  fvcopy[i] = NULL;
  *pfvc = fvc;
  *pfvp = fvcopy;
  return 0;
}

int
mu_url_get_port (const mu_url_t url, long *pport)
{
  if (url == NULL)
    return EINVAL;
  if (url->_get_port)
    return url->_get_port (url, pport);
  *pport = url->port;
  return 0;
}

const char *
mu_url_to_string (const mu_url_t url)
{
  if (url == NULL || url->name == NULL)
    return "";
  return url->name;
}

int
mu_url_set_scheme (mu_url_t url, const char *scheme)
{
  char *p;
  if (!url || !scheme)
    return EINVAL;
  p = realloc (url->scheme, strlen (scheme) + 1);
  if (!p)
    return ENOMEM;
  strcpy (url->scheme, scheme);
  return 0;
}

int
mu_url_is_scheme (mu_url_t url, const char *scheme)
{
  if (url && scheme && url->scheme 
      && mu_c_strcasecmp (url->scheme, scheme) == 0)
    return 1;

  return 0;
}

int
mu_url_is_same_port (mu_url_t url1, mu_url_t url2)
{
  long p1 = 0, p2 = 0;

  mu_url_get_port (url1, &p1);
  mu_url_get_port (url2, &p2);
  return (p1 == p2);
}

/* From RFC 1738, section 2.2 */
char *
mu_url_decode_len (const char *s, size_t len)
{
  char *d;
  const char *eos = s + len;
  int i;

  d = malloc (len + 1);
  if (!d)
    return NULL;

  for (i = 0; s < eos; i++)
    {
      if (*s != '%')
	{
	  d[i] = *s;
	  s++;
	}
      else
	{
	  unsigned long ul = 0;

	  s++;

	  /* don't check return value, it's correctly coded, or it's not,
	     in which case we just skip the garbage, this is a decoder,
	     not an AI project */

	  mu_hexstr2ul (&ul, s, 2);

	  s += 2;

	  d[i] = (char) ul;
	}
    }

  d[i] = 0;

  return d;
}

char *
mu_url_decode (const char *s)
{
  return mu_url_decode_len (s, strlen (s));
}

static int
defined (const char *s)
{
  if (s && strcmp ("*", s) != 0)
    return 1;
  return 0;
}

int
mu_url_is_ticket (mu_url_t ticket, mu_url_t url)
{
  if (!ticket || !url)
    return 0;

  /* If ticket has a scheme, host, port, or path, then the queries
     equivalent must be defined and match. */
  if (defined (ticket->scheme))
    {
      if (!url->scheme || mu_c_strcasecmp (ticket->scheme, url->scheme) != 0)
	return 0;
    }
  if (defined (ticket->host))
    {
      if (!url->host || mu_c_strcasecmp (ticket->host, url->host) != 0)
	return 0;
    }
  if (ticket->port && ticket->port != url->port)
    return 0;
  /* If ticket has a user or pass, but url doesn't, that's OK, we were
     urling for this info. But if url does have a user/pass, it
     must match the ticket. */
  if (url->user)
    {
      if (defined (ticket->user) && strcmp (ticket->user, url->user) != 0)
	return 0;
    }

  /* Guess it matches. */
  return 1;
}

int
mu_url_init (mu_url_t url, int port, const char *scheme)
{
  int status = 0;

  url->_destroy = NULL;

  status = mu_url_parse (url);
  if (status)
    return status;

  if (!mu_url_is_scheme (url, scheme))
    return EINVAL;

  if (url->port == 0)
    url->port = port;

  return status;
}

/* Default mailbox path generator */
static char *
_url_path_default (const char *spooldir, const char *user, int unused)
{
  char *mbox = malloc (strlen (spooldir) + strlen (user) + 2);
  if (!mbox)
    errno = ENOMEM;
  else
    sprintf (mbox, "%s/%s", spooldir, user);
  return mbox;
}

/* Hashed indexing */
static char *
_url_path_hashed (const char *spooldir, const char *user, int param)
{
  int i;
  int ulen = strlen (user);
  char *mbox;
  unsigned hash;

  if (param > ulen)
    param = ulen;
  for (i = 0, hash = 0; i < param; i++)
    hash += user[i];

  mbox = malloc (ulen + strlen (spooldir) + 5);
  sprintf (mbox, "%s/%02X/%s", spooldir, hash % 256, user);
  return mbox;
}

static int transtab[] = {
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
  'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
  'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
  'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f',
  'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
  'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
  'w', 'x', 'y', 'z', 'a', 'b', 'c', 'd',
  'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
  'm', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
  'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
  'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
  'x', 'y', 'z', 'b', 'c', 'd', 'e', 'f',
  'g', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
  'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
  'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
  'x', 'y', 'z', 'b', 'c', 'd', 'e', 'f',
  'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
  'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
  'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
  'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
  'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
  'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
  'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e',
  'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
  'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
  'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
  'y', 'z', 'b', 'c', 'd', 'e', 'f', 'g',
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
  'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
  'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
  'y', 'z', 'b', 'c', 'd', 'e', 'f', 'g'
};

/* Forward Indexing */
static char *
_url_path_index (const char *spooldir, const char *iuser, int index_depth)
{
  const unsigned char* user = (const unsigned char*) iuser;
  int i, ulen = strlen (iuser);
  char *mbox, *p;

  if (ulen == 0)
    return NULL;

  mbox = malloc (ulen + strlen (spooldir) + 2*index_depth + 2);
  strcpy (mbox, spooldir);
  p = mbox + strlen (mbox);
  for (i = 0; i < index_depth && i < ulen; i++)
    {
      *p++ = '/';
      *p++ = transtab[ user[i] ];
    }
  for (; i < index_depth; i++)
    {
      *p++ = '/';
      *p++ = transtab[ user[ulen-1] ];
    }
  *p++ = '/';
  strcpy (p, iuser);
  return mbox;
}

/* Reverse Indexing */
static char *
_url_path_rev_index (const char *spooldir, const char *iuser, int index_depth)
{
  const unsigned char* user = (const unsigned char*) iuser;
  int i, ulen = strlen (iuser);
  char *mbox, *p;

  if (ulen == 0)
    return NULL;

  mbox = malloc (ulen + strlen (spooldir) + 2*index_depth + 1);
  strcpy (mbox, spooldir);
  p = mbox + strlen (mbox);
  for (i = 0; i < index_depth && i < ulen; i++)
    {
      *p++ = '/';
      *p++ = transtab[ user[ulen - i - 1] ];
    }
  for (; i < index_depth; i++)
    {
      *p++ = '/';
      *p++ = transtab[ user[0] ];
    }
  *p++ = '/';
  strcpy (p, iuser);
  return mbox;
}

static int
rmselector (const char *p, void *data MU_ARG_UNUSED)
{
  return strncmp (p, "type=", 5) == 0
	 || strncmp (p, "user=", 5) == 0
	 || strncmp (p, "param=", 6) == 0;
}

int
mu_url_expand_path (mu_url_t url)
{
  size_t i;
  char *user = NULL;
  int param = 0;
  char *p;
  char *(*fun) (const char *, const char *, int) = _url_path_default;

  if (url->fvcount == 0)
    return 0;

  for (i = 0; i < url->fvcount; i++)
    {
      p = url->fvpairs[i];
      if (strncmp (p, "type=", 5) == 0)
	{
	  char *type = p + 5;

	  if (strcmp (type, "hash") == 0)
	    fun = _url_path_hashed;
	  else if (strcmp (type, "index") == 0)
	    fun = _url_path_index;
	  else if (strcmp (type, "rev-index") == 0)
	    fun = _url_path_rev_index;
	  else
	    return MU_ERR_NOENT;
	}
      else if (strncmp (p, "user=", 5) == 0)
	{
	  user = p + 5;
	}
      else if (strncmp (p, "param=", 6) == 0)
	{
	  param = strtoul (p + 6, NULL, 0);
	}
    }

  if (user)
    {
      char *p = fun (url->path, user, param);
      if (p)
	{
	  free (url->path);
	  url->path = p;
	}
      mu_argcv_remove (&url->fvcount, &url->fvpairs, rmselector, NULL);
    }
  else
    return MU_ERR_NOENT;

  return 0;
}
