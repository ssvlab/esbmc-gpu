/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2009, 2010 Free Software
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

/*

FIXME: what is the status of this TODO list?

Things to consider:

  - When parsing phrase, should I ignore non-ascii, or replace with a
    '?' character? Right now parsing fails.

    --> Should ignore non-ascii, it is unicode or iso8892-1.

  - Are comments allowed in domain-literals?

  - Need a way to mark the *end* of a group. Maybe add a field to _mu_address,
    int group_end;, so if you care, you can search for the end of
    a group with address_is_group_end();

    --> Groups no longer show up in the mu_address_t list.

  - Need a way to parse ",,,", it's a valid address-list, it just doesn't
    have any addresses.

  - The personal for ""Sam"" <sam@here> is "Sam", and for "'s@b'" <s@b>
    is 's@b', should I strip those outside parentheses, or is that
    too intrusive? Maybe an apps business if it wants to?

  - Should we do best effort parsing, so parsing "sam@locahost, foo@"
    gets one address, or just say it is or it isn't in RFC format?
    Right now we're strict, we'll see how it goes.

  - parse Received: field?

  - test for memory leaks on malloc failure

  - fix the realloc, try a struct _string { char* b, size_t sz };

The lexer finds consecutive sequences of characters, so it should
define:

struct parse822_token_t {
    const char* b;  // beginning of token
    const char* e;  // one past end of token
}
typedef struct parse822_token_t TOK;

Then I can have str_append_token(), and the lexer functions can
look like:

int mu_parse822_atom(const char** p, const char* e, TOK* atom);

Just a quick thought, I'll have to see how many functions that will
actually help.

  - get example addresses from rfc2822, and from the perl code.
*/

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/cctype.h>
#include <mailutils/cstr.h>
#include <mailutils/errno.h>
#include <mailutils/parse822.h>
#include <mailutils/address.h>

#ifdef EOK
# undef EOK
#endif

#define EOK	0
#define EPARSE	MU_ERR_BAD_822_FORMAT

/*
 * Some convenience functions for dealing with dynamically re-sized
 * strings.
 */

static int
str_append_n (char **to, const char *from, size_t n)
{
  size_t l = 0;

  /* if not to, then silently discard data */
  if (!to)
    {
      return EOK;
    }

  if (*to)
    {
      char *bigger;

      l = strlen (*to);

      bigger = realloc (*to, l + n + 1);

      if (!bigger)
	{
	  return ENOMEM;
	}

      *to = bigger;
    }
  else
    {
      *to = malloc (n + 1);
    }

  strncpy (&to[0][l], from, n);

  /* strncpy is lame, nul terminate our buffer */

  to[0][l + n] = 0;

  return EOK;
}

static int
str_append (char **to, const char *from)
{
  if (!from)
    return 0;
  return str_append_n (to, from, strlen (from));
}

static int
str_append_char (char **to, char c)
{
  return str_append_n (to, &c, 1);
}

static int
str_append_range (char **to, const char *b, const char *e)
{
  return str_append_n (to, b, e - b);
}

static void
str_free (char **s)
{
  if (s && *s)
    {
      free (*s);
      *s = 0;
    }
}

/*
 * Character Classification - could be rewritten in a C library
 * independent way, my system's C library matches the RFC
 * definitions. I don't know if that's guaranteed.
 *  
 * Note that all return values are:
 *   1 -> TRUE
 *   0 -> FALSE
 * This may be appear different than the 0 == success return
 * values of the other functions, but I was getting lost in
 * boolean arithmetic.
 */
int
mu_parse822_is_char (char c)
{
  return mu_isascii (c);
}

int
mu_parse822_is_digit (char c)
{
  /* digit = <any ASCII decimal digit> */

  return mu_isdigit ((unsigned) c);
}

int
mu_parse822_is_ctl (char c)
{
  return mu_iscntrl ((unsigned) c) || c == 127 /* DEL */ ;
}

int
mu_parse822_is_space (char c)
{
  return c == ' ';
}

int
mu_parse822_is_htab (char c)
{
  return c == '\t';
}

int
mu_parse822_is_lwsp_char (char c)
{
  return mu_parse822_is_space (c) || mu_parse822_is_htab (c);
}

int
mu_parse822_is_special (char c)
{
  return strchr ("()<>@,;:\\\".[]", c) ? 1 : 0;
}

int
parse822_is_atom_char_ex (char c)
{
  return !mu_parse822_is_special (c)
    && !mu_parse822_is_space (c)
    && !mu_parse822_is_ctl (c);
}

int
mu_parse822_is_atom_char (char c)
{
  return mu_parse822_is_char (c) && parse822_is_atom_char_ex (c);
}

int
mu_parse822_is_q_text (char c)
{
  return
    mu_parse822_is_char (c) &&
    c != '"' &&
    c != '\\' &&
    c != '\r';
}

int
mu_parse822_is_d_text (char c)
{
  return
    mu_parse822_is_char (c) &&
    c != '[' &&
    c != ']' &&
    c != '\\' &&
    c != '\r';
}
/*
 * SMTP's version of qtext, called <q> in the RFC 821 syntax,
 * also excludes <LF>.
 */
int
mu_parse822_is_smtp_q (char c)
{
  return
    mu_parse822_is_q_text (c) &&
    c != '\n';
}

/***** From RFC 822, 3.3 Lexical Tokens *****/

int
mu_parse822_skip_nl (const char **p, const char *e)
{
  /* Here we consider a new-line (NL) to be either a bare LF, or
   * a CRLF pair as required by the RFC.
   */
  const char *s = *p;

  if ((&s[1] < e) && s[0] == '\r' && s[1] == '\n')
    {
      *p += 2;

      return EOK;
    }

  if ((&s[0] < e) && s[0] == '\n')
    {
      *p += 1;

      return EOK;
    }

  return EPARSE;
}

int
mu_parse822_skip_lwsp_char (const char **p, const char *e)
{
  if (*p < e && mu_parse822_is_lwsp_char (**p))
    {
      *p += 1;
      return EOK;
    }
  return EPARSE;
}

int
mu_parse822_skip_lwsp (const char **p, const char *e)
{
  /*
   * linear-white-space = 1*([[CR]LF] LWSP-char)
   *
   *   We interpret a bare LF as identical to the canonical CRLF
   *   line ending, I don't know another way since on a Unix system
   *   all CRLF will be translated to the local convention, a bare
   *   LF, and thus we can not deal with bare NLs in the message.
   */
  int space = 0;

  while (*p != e)
    {
      const char *save = *p;

      if (mu_parse822_skip_lwsp_char (p, e) == EOK)
	{
	  space = 1;
	  continue;
	}
      if (mu_parse822_skip_nl (p, e) == EOK)
	{
	  if (mu_parse822_skip_lwsp_char (p, e) == EOK)
	    {
	      continue;
	    }
	  *p = save;
	  return EPARSE;
	}
      break;
    }
  return space ? EOK : EPARSE;
}

int
mu_parse822_skip_comments (const char **p, const char *e)
{
  int status;

  while ((status = mu_parse822_comment (p, e, 0)) == EOK)
    ;

  return EOK;
}

int
mu_parse822_digits (const char **p, const char *e, int min, int max, int *digits)
{
  const char *save = *p;

  int i = 0;

  assert (digits);

  *digits = 0;

  while (*p < e && mu_parse822_is_digit (**p))
    {
      *digits = *digits * 10 + **p - '0';
      *p += 1;
      ++i;
      if (max != 0 && i == max)
	{
	  break;
	}
    }
  if (i < min)
    {
      *p = save;
      return EPARSE;
    }

  return EOK;
}

int
mu_parse822_special (const char **p, const char *e, char c)
{
  mu_parse822_skip_lwsp (p, e);	/* not comments, they start with a special... */

  if ((*p != e) && **p == c)
    {
      *p += 1;
      return EOK;
    }
  return EPARSE;
}

int
mu_parse822_comment (const char **p, const char *e, char **comment)
{
  /* comment = "(" *(ctext / quoted-pair / comment) ")"
   * ctext = <any char except "(", ")", "\", & CR, including lwsp>
   */
  const char *save = *p;
  int rc;

  if ((rc = mu_parse822_special (p, e, '(')))
    {
      return rc;
    }

  while (*p != e)
    {
      char c = **p;

      if (c == ')')
	{
	  *p += 1;
	  return EOK;		/* found end-of-comment */
	}
      else if (c == '(')
	{
	  rc = mu_parse822_comment (p, e, comment);
	}
      else if (c == '\\')
	{
	  rc = mu_parse822_quoted_pair (p, e, comment);
	}
      else if (c == '\r')
	{
	  /* invalid character... */
	  *p += 1;
	}
      else if (mu_parse822_is_char (c))
	{
	  rc = str_append_char (comment, c);
	  *p += 1;
	}
      else
	{
	  /* invalid character... */
	  *p += 1;
	}
      if (rc != EOK)
	break;
    }

  if (*p == e)
    {
      rc = EPARSE;		/* end-of-comment not found */
    }

  *p = save;

  assert (rc != EOK);

  return rc;
}

int
mu_parse822_atom (const char **p, const char *e, char **atom)
{
  /* atom = 1*<an atom char> */

  const char *save = *p;
  int rc = EPARSE;

  mu_parse822_skip_comments (p, e);

  save = *p;

  while ((*p != e) && (**p == '.' || mu_parse822_is_atom_char (**p)))
    {
      rc = str_append_char (atom, **p);
      *p += 1;
      if (rc != EOK)
	{
	  *p = save;
	  break;
	}
    }
  return rc;
}

int
parse822_atom_ex (const char **p, const char *e, char **atom)
{
  /* atom = 1*<an atom char> */
  const char *ptr;
  int rc;

  mu_parse822_skip_comments (p, e);

  for (ptr = *p; (ptr != e) && parse822_is_atom_char_ex (*ptr); ptr++)
    ;
  if (ptr - *p == 0)
    return EPARSE;
  rc = str_append_n (atom, *p, ptr - *p);
  if (rc == 0)
    *p = ptr;
  return rc;
}

int
mu_parse822_quoted_pair (const char **p, const char *e, char **qpair)
{
  /* quoted-pair = "\" char */

  int rc;

  /* need TWO characters to be available */
  if ((e - *p) < 2)
    return EPARSE;

  if (**p != '\\')
    return EPARSE;

  if ((rc = str_append_char (qpair, *(*p + 1))))
    return rc;

  *p += 2;

  return EOK;
}

int
mu_parse822_quoted_string (const char **p, const char *e, char **qstr)
{
  /* quoted-string = <"> *(qtext/quoted-pair) <">
   * qtext = char except <">, "\", & CR, including lwsp-char
   */

  const char *save = *p;
  int rc;

  mu_parse822_skip_comments (p, e);

  save = *p;

  if ((rc = mu_parse822_special (p, e, '"')))
    return rc;

  while (*p != e)
    {
      char c = **p;

      if (c == '"')
	{
	  *p += 1;
	  return EOK;		/* found end-of-qstr */
	}
      else if (c == '\\')
	{
	  rc = mu_parse822_quoted_pair (p, e, qstr);
	}
      else if (c == '\r')
	{
	  /* invalid character... */
	  *p += 1;
	}
      else if (mu_parse822_is_char (c))
	{
	  rc = str_append_char (qstr, c);
	  *p += 1;
	}
      else
	{
	  /* invalid character... */
	  *p += 1;
	}
      if (rc)
	{
	  *p = save;
	  str_free (qstr);
	  return rc;
	}
    }
  *p = save;
  str_free (qstr);
  return EPARSE;		/* end-of-qstr not found */
}

int
mu_parse822_word (const char **p, const char *e, char **word)
{
  /* word = atom / quoted-string */

  const char *save = *p;
  int rc = EOK;

  mu_parse822_skip_comments (p, e);

  save = *p;

  {
    char *qstr = 0;
    if ((rc = mu_parse822_quoted_string (p, e, &qstr)) == EOK && qstr)
      {
	rc = str_append (word, qstr);

	str_free (&qstr);

	if (rc != EOK)
	  *p = save;

	return rc;
      }
    assert (qstr == NULL);
  }

  if (rc != EPARSE)
    {
      /* it's fatal */
      return rc;
    }

  /* Necessary because the quoted string could have found
   * a partial string (invalid syntax). Thus reset, the atom
   * will fail to if the syntax is invalid.
   * We use parse822_atom_ex to allow for non-rfc-compliant atoms:
   *
   * "Be liberal in what you accept, and conservative in what you send."
   */

  {
    char *atom = 0;
    if (parse822_atom_ex (p, e, &atom) == EOK)
      {
	rc = str_append (word, atom);

	str_free (&atom);

	if (rc != EOK)
	  *p = save;

	return rc;
      }
    assert (atom == NULL);
  }

  return EPARSE;
}

/* Some mailers do not quote personal part even if it contains dot.
   Try to be smart about it.
*/
	
int
parse822_word_dot (const char **p, const char *e, char **word)
{
  int rc = mu_parse822_word (p, e, word);
  for (;rc == 0 && (*p != e) && **p == '.'; ++*p)
    rc = str_append (word, ".");
  return rc;
}

int
mu_parse822_phrase (const char **p, const char *e, char **phrase)
{
  /* phrase = 1*word */

  const char *save = *p;
  int rc;

  if ((rc = parse822_word_dot (p, e, phrase)))
    return rc;

  /* ok, got the 1 word, now append all the others we can */
  {
    char *word = 0;

    while ((rc = parse822_word_dot (p, e, &word)) == EOK)
      {
	rc = str_append_char (phrase, ' ');

	if (rc == EOK)
	  rc = str_append (phrase, word);

	str_free (&word);

	if (rc != EOK)
	  break;
      }
    assert (word == NULL);
    if (rc == EPARSE)
      rc = EOK;			/* its not an error to find no more words */
  }
  if (rc)
    *p = save;

  return rc;
}

/***** From RFC 822, 6.1 Address Specification Syntax *****/

static mu_address_t
new_mb (void)
{
  return calloc (1, sizeof (struct mu_address));
}

static char *
addr_field_by_mask (mu_address_t addr, int mask)
{
  switch (mask)						
    {
    case MU_ADDR_HINT_ADDR:
      return addr->addr;
	  
    case MU_ADDR_HINT_COMMENTS:				
      return addr->comments;					
	  
    case MU_ADDR_HINT_PERSONAL:				
      return addr->personal;					

    case MU_ADDR_HINT_EMAIL:
      return addr->email;

    case MU_ADDR_HINT_LOCAL:
      return addr->local_part;
      
    case MU_ADDR_HINT_DOMAIN:				
      return addr->domain;					

    case MU_ADDR_HINT_ROUTE:
      return addr->route;
    }
  return NULL;
}							

static char *
get_val (mu_address_t hint, int hflags, char *value, int mask, int *memflag)
{
  if (!value && hint && (hflags & mask))
    {
      char *p = addr_field_by_mask (hint, mask);
      if (p)							
	{
	  if (memflag)
	    *memflag |= mask;
	  value = strdup (p);
	}
    }
  return value;
}

static void
addr_free_fields (mu_address_t a, int memflag)
{
  char *p;
  
  if ((p = addr_field_by_mask (a, memflag & MU_ADDR_HINT_ADDR)))
    free (p);
  if ((p = addr_field_by_mask (a, memflag & MU_ADDR_HINT_COMMENTS)))
    free (p);
  if ((p = addr_field_by_mask (a, memflag & MU_ADDR_HINT_PERSONAL)))
    free (p);
  if ((p = addr_field_by_mask (a, memflag & MU_ADDR_HINT_EMAIL)))
    free (p);
  if ((p = addr_field_by_mask (a, memflag & MU_ADDR_HINT_LOCAL)))
    free (p);
  if ((p = addr_field_by_mask (a, memflag & MU_ADDR_HINT_DOMAIN)))
    free (p);
  if ((p = addr_field_by_mask (a, memflag & MU_ADDR_HINT_ROUTE)))
    free (p);
}

static int
fill_mb (mu_address_t *pa,
	 char *comments, char *personal, char *local, char *domain,
	 mu_address_t hint, int hflags)
{
  int rc = EOK;
  mu_address_t a;
  int memflag = 0;

  a = new_mb ();

  if (!a)
    return ENOMEM;

  a->comments = get_val (hint, hflags, comments, MU_ADDR_HINT_COMMENTS,
			 &memflag);
  a->personal = get_val (hint, hflags, personal, MU_ADDR_HINT_PERSONAL,
			 &memflag);

  domain = get_val (hint, hflags, domain, MU_ADDR_HINT_DOMAIN,
		    &memflag);
  local = get_val (hint, hflags, local, MU_ADDR_HINT_LOCAL,
		   &memflag);
  do
    {
      /* loop exists only to break out of */
      if (!local)
	/* no email to construct */
	break;

      if ((rc = mu_parse822_quote_local_part (&a->email, local)))
	break;
      if (domain)
	{
	  if ((rc = str_append (&a->email, "@")))
	    break;
	  if ((rc = str_append (&a->email, domain)))
	    break;
	}
    }
  while (0);

  a->local_part = local;
  a->domain = domain;

  if (rc != EOK)
    {
      addr_free_fields (a, memflag);
      /* note that the arguments have NOT been freed, we only own
       * them on success. */
      free (a);
    }
  else
    *pa = a;

  return rc;
}

int
mu_parse822_address_list (mu_address_t *a, const char *s,
			  mu_address_t hint, int hflags)
{
  /* address-list = #(address) */

  const char **p = &s;
  const char *e = &s[strlen (s)];
  int rc = EOK;
  mu_address_t *n = a;		/* the next address we'll be creating */

  rc = mu_parse822_address (p, e, n, hint, hflags);

  /* A list may start with a leading <,>, we'll find out if
   * that's not the case at the top of the while, but give
   * this a conditional OK unless there was some other kind
   * of error.
   */
  if (rc != EOK && rc != EPARSE)
    {
      return rc;
    }
  while (*p < e)
    {
      mu_parse822_skip_comments (p, e);

      /* An address can contain a group, so an entire
       * list of addresses may have been appended, or no
       * addresses at all. Walk to the end.
       */
      while (*n)
	{
	  n = &(*n)->next;
	}

      /* Remember that ',,a@b' is a valid list! So, we must find
       * the <,>, but the address after it can be empty.
       */
      if ((rc = mu_parse822_special (p, e, ',')))
	{
	  break;
	}
      mu_parse822_skip_comments (p, e);

      rc = mu_parse822_address (p, e, n, hint, hflags);

      if (rc == EOK || rc == EPARSE)
	{
	  /* that's cool, it may be a <,>, we'll find out if it isn't
	   * at the top of the loop
	   */
	  rc = EOK;
	}
      else
	{
	  /* anything else is a fatal error, break out */
	  break;
	}
    }

  if (rc)
    {
      mu_address_destroy (a);
    }

  return rc;
}

int
mu_parse822_address (const char **p, const char *e, mu_address_t *a,
		     mu_address_t hint, int hflags)
{
  /* address = mailbox / group / unix-mbox */

  int rc;

  if ((rc = mu_parse822_mail_box (p, e, a, hint, hflags)) == EPARSE)
    {
      if ((rc = mu_parse822_group (p, e, a, hint, hflags)) == EPARSE)
	{
	  rc = mu_parse822_unix_mbox (p, e, a, hint, hflags);
	}
    }

  if (rc == 0 && *a && !(*a)->route)
    (*a)->route = get_val (hint, hflags, NULL, MU_ADDR_HINT_ROUTE, NULL);

  return rc;
}

/* No longer put groups into an address node, it wasn't useful, was
 * troublesome, and since there wasn't an end-group marker, wasn't
 * even conceivably useful.
 */
#undef ADD_GROUPS
int
mu_parse822_group (const char **p, const char *e, mu_address_t *a,
		   mu_address_t hint, int hflags)
{
  /* group = phrase ":" [#mailbox] ";" */

  const char *save = *p;
  mu_address_t *asave = a;		/* so we can destroy these if parsing fails */
  int rc;
  char *phrase = 0;

  mu_parse822_skip_comments (p, e);

  *p = save;

  if ((rc = mu_parse822_phrase (p, e, &phrase)))
    {
      return rc;
    }

  mu_parse822_skip_comments (p, e);

  if ((rc = mu_parse822_special (p, e, ':')))
    {
      *p = save;
      str_free (&phrase);
      return rc;
    }
#ifdef ADD_GROUPS
  /* fake up an address node for the group's descriptive phrase, if
   * it fails, clean-up will happen after the loop
   */
  if ((rc = fill_mb (a, 0, phrase, 0, 0, hint, hflags)) == EOK)
    {
      a = &(*a)->next;
    }
  else
    {
      str_free (&phrase);
    }
#else
  str_free (&phrase);
#endif

  /* Basically, on each loop, we may find a mailbox, but we must find
   * a comma after the mailbox, otherwise we've popped off the end
   * of the list.
   */
  while (!rc)
    {
      mu_parse822_skip_comments (p, e);

      /* it's ok not be a mailbox, but other errors are fatal */
      rc = mu_parse822_mail_box (p, e, a, hint, hflags);
      if (rc == EOK)
	{
	  a = &(*a)->next;

	  mu_parse822_skip_comments (p, e);
	}
      else if (rc != EPARSE)
	{
	  break;
	}

      if ((rc = mu_parse822_special (p, e, ',')))
	{
	  /* the commas aren't optional */
	  break;
	}
    }
  if (rc == EPARSE)
    {
      rc = EOK;			/* ok, as long as we find the ";" next */
    }

  if (rc || (rc = mu_parse822_special (p, e, ';')))
    {
      *p = save;

      mu_address_destroy (asave);
    }

  return rc;
}

int
mu_parse822_mail_box (const char **p, const char *e, mu_address_t *a,
		      mu_address_t hint, int hflags)
{
  /* mailbox =
   *     addr-spec [ "(" comment ")" ] /
   *     [phrase] route-addr
   *
   *  Note: we parse the ancient comment on the right since
   *    it's such "common practice". :-(
   *  Note: phrase is called display-name in drums.
   *  Note: phrase is optional in drums, though not in RFC 822.
   */
  const char *save = *p;
  int rc;

  /* -> addr-spec */
  if ((rc = mu_parse822_addr_spec (p, e, a, hint, hflags)) == EOK)
    {
      mu_parse822_skip_lwsp (p, e);

      /* yuck. */
      if ((rc = mu_parse822_comment (p, e, &(*a)->personal)) == EPARSE)
	{
	  rc = EOK;
	  /* cool if there's no comment, */
	}
      /* but if something else is wrong, destroy the address */
      if (rc)
	{
	  mu_address_destroy (a);
	  *p = save;
	}

      return rc;
    }

  /* -> phrase route-addr */
  {
    char *phrase = 0;

    rc = mu_parse822_phrase (p, e, &phrase);

    if (rc != EPARSE && rc != EOK)
      {
	return rc;
      }

    if ((rc = mu_parse822_route_addr (p, e, a, hint, hflags)) == EOK)
      {
	/* add the phrase */
	(*a)->personal = phrase;

	return EOK;
      }
    /* some internal error, fail out */
    str_free (&phrase);
    *p = save;

    return rc;
  }

  return rc;
}

int
mu_parse822_route_addr (const char **p, const char *e, mu_address_t *a,
			mu_address_t hint, int hflags)
{
  /* route-addr = "<" [route] addr-spec ">" */

  const char *save = *p;
  char *route = NULL;
  int rc;
  int memflag = 0;
  
  mu_parse822_skip_comments (p, e);

  if ((rc = mu_parse822_special (p, e, '<')))
    {
      *p = save;

      return rc;
    }
  if (!(rc = mu_parse822_special (p, e, '>')))
    {
      if ((rc = fill_mb (a, 0, 0, 0, 0, hint, hflags)) == EOK)
	rc = str_append (&(*a)->email, "");
       
      return rc;
    }

  mu_parse822_route (p, e, &route);

  if ((rc = mu_parse822_addr_spec (p, e, a, hint, hflags)))
    {
      *p = save;

      str_free (&route);

      return rc;
    }

  (*a)->route = get_val (hint, hflags, route, MU_ADDR_HINT_ROUTE,
			 &memflag);

  mu_parse822_skip_comments (p, e);

  if ((rc = mu_parse822_special (p, e, '>')))
    {
      *p = save;

      mu_address_destroy (a);

      return rc;
    }

  return EOK;
}

int
mu_parse822_route (const char **p, const char *e, char **route)
{
  /* route = 1#("@" domain ) ":" */

  const char *save = *p;
  char *accumulator = 0;
  int rc = EOK;

  for (;;)
    {
      mu_parse822_skip_comments (p, e);

      if ((rc = mu_parse822_special (p, e, '@')))
	{
	  break;
	}

      if ((rc = str_append (&accumulator, "@")))
	{
	  break;
	}

      mu_parse822_skip_comments (p, e);

      if ((rc = mu_parse822_domain (p, e, &accumulator)))
	{
	  /* it looked like a route, but there's no domain! */
	  break;
	}

      mu_parse822_skip_comments (p, e);

      if ((rc = mu_parse822_special (p, e, ',')) == EPARSE)
	{
	  /* no more routes, but we got one so its ok */
	  rc = EOK;
	  break;
	}
      if ((rc = str_append (&accumulator, ", ")))
	{
	  break;
	}
    }

  mu_parse822_skip_comments (p, e);

  if (!rc)
    {
      rc = mu_parse822_special (p, e, ':');
    }

  if (!rc)
    {
      rc = str_append (route, accumulator);
    }
  if (rc)
    {
      *p = save;
    }
  str_free (&accumulator);
  return rc;
}

int
mu_parse822_addr_spec (const char **p, const char *e, mu_address_t *a,
		       mu_address_t hint, int hflags)
{
  /* addr-spec = local-part "@" domain */

  const char *save = *p;
  char *local_part = 0;
  char *domain = 0;
  int rc;

  rc = mu_parse822_local_part (p, e, &local_part);

  mu_parse822_skip_comments (p, e);

  if (!rc)
    {
      rc = mu_parse822_special (p, e, '@');
      
      if (!rc)
	{
	  rc = mu_parse822_domain (p, e, &domain);

	  if (!rc)
	    rc = fill_mb (a, 0, 0, local_part, domain, hint, hflags);
	}
    }
  
  if (rc)
    {
      *p = save;
      str_free (&local_part);
      str_free (&domain);
    }
  return rc;
}

int
mu_parse822_unix_mbox (const char **p, const char *e, mu_address_t *a,
		       mu_address_t hint, int hflags)
{
  /* unix-mbox = atom */

  const char *save = *p;
  char *mbox = 0;
  int rc;

  mu_parse822_skip_comments (p, e);

  rc = mu_parse822_atom (p, e, &mbox);

  if (!rc)
    rc = fill_mb (a, 0, 0, mbox, 0, hint, hflags);

  if (rc)
    {
      *p = save;
      str_free (&mbox);
    }
  return rc;
}

int
mu_parse822_local_part (const char **p, const char *e, char **local_part)
{
  /* local-part = word *("." word)
   *
   * Note: rewrite as ->  word ["." local-part]
   */

  const char *save = *p;
  const char *save2 = *p;
  int rc;

  mu_parse822_skip_comments (p, e);

  if ((rc = mu_parse822_word (p, e, local_part)))
    {
      *p = save;
      return rc;
    }
  /* We've got a local-part, but keep looking for more. */

  mu_parse822_skip_comments (p, e);

  /* If we get a parse error, we roll back to save2, but if
   * something else failed, we have to roll back to save.
   */
  save2 = *p;

  rc = mu_parse822_special (p, e, '.');

  if (!rc)
    {
      char *more = 0;
      if ((rc = mu_parse822_local_part (p, e, &more)) == EOK)
	{
	  /* append more */
	  if ((rc = str_append (local_part, ".")) == EOK)
	    {
	      rc = str_append (local_part, more);
	    }
	}
      str_free (&more);
    }

  if (rc == EPARSE)
    {
      /* if we didn't get more ("." word) pairs, that's ok */
      *p = save2;
      rc = EOK;
    }
  if (rc)
    {
      /* if anything else failed, that's real */
      *p = save;
      str_free (local_part);
    }
  return rc;
}

int
mu_parse822_domain (const char **p, const char *e, char **domain)
{
  /* domain = sub-domain *("." sub-domain)
   *
   * Note: rewrite as -> sub-domain ("." domain)
   */

  const char *save = *p;
  const char *save2 = 0;
  int rc;

  mu_parse822_skip_comments (p, e);

  if ((rc = mu_parse822_sub_domain (p, e, domain)))
    {
      *p = save;
      return rc;
    }


  /* We save before skipping comments to preserve the comment
   * at the end of a domain, the addr-spec may want to abuse it
   * for a personal name.
   */
  save2 = *p;

  /* we've got the 1, keep looking for more */

  mu_parse822_skip_comments (p, e);

  rc = mu_parse822_special (p, e, '.');

  if (!rc)
    {
      char *more = 0;
      if ((rc = mu_parse822_domain (p, e, &more)) == EOK)
	{
	  if ((rc = str_append (domain, ".")) == EOK)
	    {
	      rc = str_append (domain, more);
	    }
	}
      str_free (&more);
    }
  if (rc == EPARSE)
    {
      /* we didn't parse more ("." sub-domain) pairs, that's ok */
      *p = save2;
      rc = EOK;
    }

  if (rc)
    {
      /* something else failed, roll it all back */
      *p = save;
      str_free (domain);
    }
  return rc;
}

int
mu_parse822_sub_domain (const char **p, const char *e, char **sub_domain)
{
  /* sub-domain = domain-ref / domain-literal
   */

  int rc;

  if ((rc = mu_parse822_domain_ref (p, e, sub_domain)) == EPARSE)
    rc = mu_parse822_domain_literal (p, e, sub_domain);

  return rc;
}

int
mu_parse822_domain_ref (const char **p, const char *e, char **domain_ref)
{
  /* domain-ref = atom */

  return mu_parse822_atom (p, e, domain_ref);
}

int
mu_parse822_d_text (const char **p, const char *e, char **dtext)
{
  /* d-text = 1*dtext
   *
   * Note: dtext is only defined as a character class in
   *  RFC822, but this definition is more useful for
   *  slurping domain literals.
   */

  const char *start = *p;
  int rc = EOK;

  while (*p < e && mu_parse822_is_d_text (**p))
    {
      *p += 1;
    }

  if (start == *p)
    {
      return EPARSE;
    }

  if ((rc = str_append_range (dtext, start, *p)))
    {
      *p = start;
    }

  return rc;
}

int
mu_parse822_domain_literal (const char **p, const char *e, char **domain_literal)
{
  /* domain-literal = "[" *(dtext / quoted-pair) "]" */

  const char *save = *p;
  char *literal = 0;
  int rc;

  if ((rc = mu_parse822_special (p, e, '[')))
    {
      return rc;
    }
  if ((rc = str_append_char (&literal, '[')))
    {
      *p = save;
      return rc;
    }

  while ((rc = mu_parse822_d_text (p, e, &literal)) == EOK ||
	 (rc = mu_parse822_quoted_pair (p, e, &literal)) == EOK)
    {
      /* Eat all of this we can get! */
    }
  if (rc == EPARSE)
    {
      rc = EOK;
    }
  if (!rc)
    {
      rc = mu_parse822_special (p, e, ']');
    }
  if (!rc)
    {
      rc = str_append_char (&literal, ']');
    }
  if (!rc)
    {
      rc = str_append (domain_literal, literal);
    }

  str_free (&literal);

  if (rc)
    {
      *p = save;
    }
  return rc;
}

/***** From RFC 822, 5.1 Date and Time Specification Syntax *****/

int
mu_parse822_day (const char **p, const char *e, int *day)
{
  /* day = "Mon" / "Tue" / "Wed" / "Thu" / "Fri" / "Sat" / "Sun" */

  const char *days[] = {
    "Sun",
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    NULL
  };

  int d;

  mu_parse822_skip_comments (p, e);

  if ((e - *p) < 3)
    return EPARSE;

  for (d = 0; days[d]; d++)
    {
      if (mu_c_strncasecmp (*p, days[d], 3) == 0)
	{
	  *p += 3;
	  if (day)
	    *day = d;
	  return EOK;
	}
    }
  return EPARSE;
}

int
mu_parse822_date (const char **p, const char *e, int *day, int *mon, int *year)
{
  /* date = 1*2DIGIT month 2*4DIGIT
   * month =  "Jan"  /  "Feb" /  "Mar"  /  "Apr"
   *       /  "May"  /  "Jun" /  "Jul"  /  "Aug"
   *       /  "Sep"  /  "Oct" /  "Nov"  /  "Dec"
   */

  const char *mons[] = {
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    NULL
  };

  const char *save = *p;
  int rc = EOK;
  int m = 0;
  int yr = 0;
  const char *yrbeg = 0;

  mu_parse822_skip_comments (p, e);

  if ((rc = mu_parse822_digits (p, e, 1, 2, day)))
    {
      *p = save;
      return rc;
    }

  mu_parse822_skip_comments (p, e);

  if ((e - *p) < 3)
    return EPARSE;

  for (m = 0; mons[m]; m++)
    {
      if (mu_c_strncasecmp (*p, mons[m], 3) == 0)
	{
	  *p += 3;
	  if (mon)
	    *mon = m;
	  break;
	}
    }

  if (!mons[m])
    {
      *p = save;
      return EPARSE;
    }

  mu_parse822_skip_comments (p, e);

  /* We need to count how many digits their were, and adjust the
   * interpretation of the year accordingly. This is from RFC 2822,
   * Section 4.3, Obsolete Date and Time. */
  yrbeg = *p;

  if ((rc = mu_parse822_digits (p, e, 2, 4, &yr)))
    {
      *p = save;
      return rc;
    }

  /* rationalize year to four digit, then adjust to tz notation */
  switch (*p - yrbeg)
    {
    case 2:
      if (yr >= 0 && yr <= 49)
	{
	  yr += 2000;
	  break;
	}
    case 3:
      yr += 1900;
      break;
    }

  if (year)
    *year = yr - 1900;

  return EOK;
}

int
mu_parse822_time (const char **p, const char *e,
	       int *hour, int *min, int *sec, int *tz, const char **tz_name)
{
  /* time        =  hour zone
   * hour        =  2DIGIT ":" 2DIGIT [":" 2DIGIT] ; 00:00:00 - 23:59:59
   * zone        =  "UT"  / "GMT"           ; Universal Time
   *                                        ; North American : UT
   *             /  "EST" / "EDT"           ;  Eastern:  - 5/ - 4
   *             /  "CST" / "CDT"           ;  Central:  - 6/ - 5
   *             /  "MST" / "MDT"           ;  Mountain: - 7/ - 6
   *             /  "PST" / "PDT"           ;  Pacific:  - 8/ - 7
   *             /  1ALPHA                  ; RFC 822 was wrong, RFC 2822
   *                                        ; says treat these all as -0000.
   *             / ( ("+" / "-") 4DIGIT )   ; Local differential
   *                                        ;  hours+min. (HHMM)
   */

  struct
  {
    const char *tzname;
    int tz;
  }
  tzs[] =
  {
    { "UT",   0 * 60 * 60 },
    { "UTC",  0 * 60 * 60 },
    { "GMT",  0 * 60 * 60 },
    { "EST", -5 * 60 * 60 },
    { "EDT", -4 * 60 * 60 },
    { "CST", -6 * 60 * 60 },
    { "CDT", -5 * 60 * 60 },
    { "MST", -7 * 60 * 60 },
    { "MDT", -6 * 60 * 60 },
    { "PST", -8 * 60 * 60 },
    { "PDT", -7 * 60 * 60 },
    { NULL, 0}
  };

  const char *save = *p;
  int rc = EOK;
  int z = 0;
  char *zone = NULL;

  mu_parse822_skip_comments (p, e);

  if ((rc = mu_parse822_digits (p, e, 1, 2, hour)))
    {
      *p = save;
      return rc;
    }

  if ((rc = mu_parse822_special (p, e, ':')))
    {
      *p = save;
      return rc;
    }

  if ((rc = mu_parse822_digits (p, e, 1, 2, min)))
    {
      *p = save;
      return rc;
    }

  if ((rc = mu_parse822_special (p, e, ':')))
    {
      *sec = 0;
    }
  else if ((rc = mu_parse822_digits (p, e, 1, 2, sec)))
    {
      *p = save;
      return rc;
    }

  mu_parse822_skip_comments (p, e);

  if ((rc = mu_parse822_atom (p, e, &zone)))
    {
      /* zone is optional */
      if (tz)
	*tz = 0;
      return EOK;
    }

  /* see if it's a timezone */
  for (; tzs[z].tzname; z++)
    {
      if (mu_c_strcasecmp (zone, tzs[z].tzname) == 0)
	break;
    }
  if (tzs[z].tzname)
    {
      if (tz_name)
	*tz_name = tzs[z].tzname;

      if (tz)
	*tz = tzs[z].tz;
    }
  else if (strlen (zone) > 5 || strlen (zone) < 4)
    {
      if (*tz)
	*tz = 0; /* Assume UTC */
    }
  else
    {
      /* zone = ( + / - ) hhmm */
      int hh;
      int mm;
      int sign;
      char *zp = zone;

      switch (zp[0])
	{
	case '-':
	  sign = -1;
	  zp++;
	  break;
	case '+':
	  sign = +1;
	  zp++;
	  break;
	default:
	  sign = 1;
	  break;
	}

      if (strspn (zp, "0123456789") == 4)
	{
      /* convert to seconds from UTC */
      hh = (zone[1] - '0') * 10 + (zone[2] - '0');
      mm = (zone[3] - '0') * 10 + (zone[4] - '0');
	}
      else
	{
	  hh = mm = 0; /* Consider equivalent to -0000 */
	}
      if (tz)
	*tz = sign * (hh * 60 * 60 + mm * 60);
    }

  str_free (&zone);

  return EOK;
}

#if 0
For reference, especially the for the required range and values of the
integer fields.

struct tm
{
  int tm_sec;			/* Seconds.	[0-60] (1 leap second) */
  int tm_min;			/* Minutes.	[0-59] */
  int tm_hour;			/* Hours.	[0-23] */
  int tm_mday;			/* Day.		[1-31] */
  int tm_mon;			/* Month.	[0-11] */
  int tm_year;			/* Year	- 1900.  */
  int tm_wday;			/* Day of week.	[0-6] */
  int tm_yday;			/* Days in year.[0-365]	*/
  int tm_isdst;			/* DST.		[-1/0/1]*/

  int tm_gmtoff;        /* Seconds east of UTC. */
  const char *tm_zone;	/* Timezone abbreviation.  */
};
#endif

int
mu_parse822_date_time (const char **p, const char *e, struct tm *tm,
		       struct mu_timezone *tz)
{
  /* date-time = [ day "," ] date time */

  const char *save = *p;
  int rc = 0;

  int wday = 0;

  int mday = 0;
  int mon = 0;
  int year = 0;

  int hour = 0;
  int min = 0;
  int sec = 0;

  int tzoffset = 0;
  const char *tz_name = 0;

  if ((rc = mu_parse822_day (p, e, &wday)))
    {
      if (rc != EPARSE)
	return rc;
    }
  else
    {
      /* If we got a day, we MUST have a ','. */
      mu_parse822_skip_comments (p, e);

      if ((rc = mu_parse822_special (p, e, ',')))
	{
	  *p = save;
	  return rc;
	}
    }

  if ((rc = mu_parse822_date (p, e, &mday, &mon, &year)))
    {
      *p = save;
      return rc;
    }
  if ((rc = mu_parse822_time (p, e, &hour, &min, &sec, &tzoffset, &tz_name)))
    {
      *p = save;
      return rc;
    }

  if (tm)
    {
      memset (tm, 0, sizeof (*tm));

      tm->tm_wday = wday;

      tm->tm_mday = mday;
      tm->tm_mon = mon;
      tm->tm_year = year;

      tm->tm_hour = hour;
      tm->tm_min = min;
      tm->tm_sec = sec;

#ifdef HAVE_STRUCT_TM_TM_ISDST
      tm->tm_isdst = -1;	/* unknown whether it's dst or not */
#endif
#ifdef HAVE_STRUCT_TM_TM_GMTOFF
      tm->tm_gmtoff = tzoffset;
#endif
#ifdef HAVE_STRUCT_TM_TM_ZONE
      tm->tm_zone = (char*) tz_name;
#endif
    }

  if (tz)
    {
      tz->utc_offset = tzoffset;
      tz->tz_name = tz_name;
    }

  return EOK;
}

/***** From RFC 822, 3.2 Header Field Definitions *****/

int
mu_parse822_field_name (const char **p, const char *e, char **fieldname)
{
  /* field-name = 1*<any char, excluding ctls, space, and ":"> ":" */

  const char *save = *p;

  char *fn = NULL;

  while (*p != e)
    {
      char c = **p;

      if (!mu_parse822_is_char (c))
	break;

      if (mu_parse822_is_ctl (c))
	break;
      if (mu_parse822_is_space (c))
	break;
      if (c == ':')
	break;

      str_append_char (&fn, c);
      *p += 1;
    }
  /* must be at least one char in the field name */
  if (!fn)
    {
      *p = save;
      return EPARSE;
    }
  mu_parse822_skip_comments (p, e);

  if (!mu_parse822_special (p, e, ':'))
    {
      *p = save;
      if (fn)
	free (fn);
      return EPARSE;
    }

  *fieldname = fn;

  return EOK;
}

int
mu_parse822_field_body (const char **p, const char *e, char **fieldbody)
{
  /* field-body = *text [CRLF lwsp-char field-body] */

  /*const char *save = *p; */

  char *fb = NULL;

  for (;;)
    {
      const char *eol = *p;
      while (eol != e)
	{
	  /*char c = *eol; */
	  if (eol[0] == '\r' && (eol + 1) != e && eol[1] == '\n')
	    break;
	  ++eol;
	}
      str_append_range (&fb, *p, eol);
      *p = eol;
      if (eol == e)
	break;			/* no more, so we're done */

      /*assert(p[0] == '\r'); */
      /*assert(p[1] == '\n'); */

      *p += 2;

      if (*p == e)
	break;			/* no more, so we're done */

      /* check if next line is a continuation line */
      if (**p != ' ' && **p != '\t')
	break;
    }

  *fieldbody = fb;

  return EOK;
}

/***** RFC 822 Quoting Functions *****/

int
mu_parse822_quote_string (char **quoted, const char *raw)
{
  /* quoted-string = <"> *(qtext/quoted-pair) <">
   *
   * So double quote the string, and back quote anything that
   * isn't qtext.
   */

  int rc = EOK;
  const char *s;

  if (!raw || !quoted || *quoted)
    {
      return EINVAL;
    }

  s = raw;

  rc = str_append_char (quoted, '"');

  while (!rc && *s)
    {
      if (!mu_parse822_is_q_text (*s))
	{
	  rc = str_append_char (quoted, '\\');
	}

      if (!rc)
	{
	  rc = str_append_char (quoted, *s);
	}
      ++s;
    }

  if (!rc)
    {
      rc = str_append_char (quoted, '"');
    }

  if (rc)
    {
      str_free (quoted);
    }
  return rc;
}

int
mu_parse822_quote_local_part (char **quoted, const char *raw)
{
  /* local-part = word * ("." word)
   * word = atom / quoted-string
   *
   * So, if any character isn't a "." or an atom character, we quote
   * the whole thing as a string, for simplicity, otherwise just
   * copy it.
   */

  const char *s = 0;

  if (!raw || !quoted || *quoted)
    {
      return EINVAL;
    }
  s = raw;

  while (*s)
    {
      if (*s != '.' && !mu_parse822_is_atom_char (*s))
	{
	  return mu_parse822_quote_string (quoted, raw);
	}
      ++s;
    }

  /* if we don't have to quote it, just copy it over */

  return str_append (quoted, raw);
}

