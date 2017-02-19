/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2003, 2004, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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

#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <mailutils/cctype.h>
#include <mailutils/mailcap.h>
#include <mailutils/stream.h>
#include <mailutils/mutil.h>
#include <mailutils/cstr.h>

/* Definition of the structure, this should be in mailutils/sys/mailcap.h.  */
struct _mu_mailcap_entry
{
  char * typefield;
  char * viewcommand;
  char ** fields;
  size_t fields_count;
};

/* Definition of the structure, this should be in mailutils/sys/mailcap.h.  */
struct _mu_mailcap
{
  mu_mailcap_entry_t *entries;
  size_t entries_count;
};


static int mu_mailcap_parse (mu_mailcap_t mailcap, mu_stream_t stream);
static int mu_mailcap_parse_entry (mu_mailcap_entry_t entry, char *buffer);
static char * tokenize (char *s, char **save_ptr);

int
mu_mailcap_create (mu_mailcap_t * pmailcap, mu_stream_t stream)
{
  mu_mailcap_t mailcap;
  int status = 0;

  if (stream == NULL)
    return EINVAL;
  if (pmailcap == NULL)
    return MU_ERR_OUT_PTR_NULL;

  mailcap = calloc (1, sizeof (*mailcap));
  if (mailcap != NULL)
    {
      status = mu_mailcap_parse (mailcap, stream);
      if (status != 0)
	{
	  mu_mailcap_destroy (&mailcap);
	}
      else
	{
	  *pmailcap = mailcap;
	}
    }
  else
    {
      status = ENOMEM;
    }
  return status;
}

void
mu_mailcap_destroy (mu_mailcap_t * pmailcap)
{
  if (pmailcap != NULL && *pmailcap != NULL)
    {
      int i;
      mu_mailcap_t mailcap = *pmailcap;

      for (i = 0; i < mailcap->entries_count; i++)
	{
	  int j;
	  mu_mailcap_entry_t entry = mailcap->entries[i];
	  free (entry->typefield);
	  free (entry->viewcommand);
	  for (j = 0; j < entry->fields_count; j++)
	    {
	      free (entry->fields[j]);
	    }
	}
    }
}

int
mu_mailcap_entries_count (mu_mailcap_t mailcap, size_t *pcount)
{
  int status = 0;

  if (mailcap == NULL)
    status = EINVAL;
  if (pcount != NULL)
    {
      *pcount = mailcap->entries_count;
    }
  return status;
}

int
mu_mailcap_get_entry (mu_mailcap_t mailcap, size_t no,
		      mu_mailcap_entry_t *pentry)
{
  int status = 0;
  if (mailcap == NULL)
    status = EINVAL;
  else if (pentry == NULL)
    status = MU_ERR_OUT_PTR_NULL;
  else if (no == 0 || no > mailcap->entries_count)
    {
      status = MU_ERR_NOENT;
    }
  else
    {
      *pentry = mailcap->entries[no - 1];
    }
  return status;
}

int
mu_mailcap_entry_get_typefield (mu_mailcap_entry_t entry, char *buffer,
				size_t buflen, size_t *pn)
{
  int status = 0;
  int len = 0;

  if (entry == NULL)
    {
      status = EINVAL;
    }
  else
    {
      len = strlen(entry->typefield);
      if (buffer != NULL && buflen > 0)
	{
	  buflen--;
	  len = (len < buflen) ? len : buflen;
	  memcpy (buffer, entry->typefield, len);
	  buffer[len] = '\0';
	}
    }
  if (pn)
    {
      *pn = len;
    }
  return status;
}

int
mu_mailcap_entry_get_viewcommand (mu_mailcap_entry_t entry, char *buffer,
				  size_t buflen, size_t *pn)
{
  int status = 0;
  int len = 0;

  if (entry == NULL)
    {
      status = EINVAL;
    }
  else
    {
      len = strlen(entry->viewcommand);
      if (buffer != NULL && buflen > 0)
	{
	  buflen--;
	  len = (len < buflen) ? len : buflen;
	  memcpy (buffer, entry->viewcommand, len);
	  buffer[len] = '\0';
	}
    }
  if (pn)
    {
      *pn = len;
    }
  return status;
}

int
mu_mailcap_entry_fields_count (mu_mailcap_entry_t entry, size_t *pcount)
{
  int status = 0;
  if (entry == NULL)
    {
      status = EINVAL;
    }
  if (pcount != NULL)
    {
      *pcount = entry->fields_count;
    }
  return status;
}

int
mu_mailcap_entry_get_field (mu_mailcap_entry_t entry, size_t no,
			    char *buffer, size_t buflen, size_t *pn)
{
  int status = 0;
  int len = 0;

  if (entry == NULL)
    {
      status = EINVAL;
    }
  else if ( no == 0 || no > entry->fields_count)
    {
      status = MU_ERR_NOENT;
    }
  else
    {
      len = strlen(entry->fields[no - 1]);
      if (buffer != NULL && buflen > 0)
	{
	  buflen--;
	  len = (len < buflen) ? len : buflen;
	  memcpy (buffer, entry->fields[no - 1], len);
	  buffer[len] = '\0';
	}
    }
  if (pn)
    {
      *pn = len;
    }
  return status;
}

int
mu_mailcap_entry_get_compose (mu_mailcap_entry_t entry,
			      char *buffer, size_t buflen, size_t *pn)
{
  return mu_mailcap_entry_get_value (entry, "compose", buffer, buflen, pn);
}

int
mu_mailcap_entry_get_composetyped (mu_mailcap_entry_t entry, char *buffer,
				   size_t buflen, size_t *pn)
{
  return mu_mailcap_entry_get_value (entry, "composetyped", buffer, buflen, pn);
}

int
mu_mailcap_entry_get_edit (mu_mailcap_entry_t entry, char *buffer,
			   size_t buflen, size_t *pn)
{
  return mu_mailcap_entry_get_value (entry, "edit", buffer, buflen, pn);
}

int
mu_mailcap_entry_get_test (mu_mailcap_entry_t entry,
			   char *buffer, size_t buflen, size_t *pn)
{
  return mu_mailcap_entry_get_value (entry, "test", buffer, buflen, pn);
}

int
mu_mailcap_entry_get_x11bitmap (mu_mailcap_entry_t entry,
				char *buffer, size_t buflen, size_t *pn)
{
  return mu_mailcap_entry_get_value (entry, "x11-bitmap", buffer, buflen, pn);
}

int
mu_mailcap_entry_get_description (mu_mailcap_entry_t entry,
				  char *buffer, size_t buflen, size_t *pn)
{
  return mu_mailcap_entry_get_value (entry, "description", buffer, buflen, pn);
}

int
mu_mailcap_entry_needsterminal (mu_mailcap_entry_t entry, int *on)
{
  int status = 0;
  int found = 0;
  if (entry == NULL)
    {
      status = EINVAL;
    }
  else
    {
      int i;
      for (i = 0; i < entry->fields_count; i++)
	{
	  int n = mu_c_strcasecmp (entry->fields[i], "needsterminal");
	  if (n == 0)
	    {
	      found = 1;
	      break;
	    }
	}
    }
  if (on)
    *on = found;
  return status;
}

int
mu_mailcap_entry_copiousoutput (mu_mailcap_entry_t entry, int *on)
{
  int status = 0;
  int found = 0;
  if (entry == NULL)
    {
      status = EINVAL;
    }
  else
    {
      int i;
      for (i = 0; i < entry->fields_count; i++)
	{
	  int n = mu_c_strcasecmp (entry->fields[i], "copiousoutput");
	  if (n == 0)
	    {
	      found = 1;
	      break;
	    }
	}
    }
  if (on)
    *on = found;
  return status;
}

int
mu_mailcap_entry_get_value (mu_mailcap_entry_t entry, const char *key,
			    char *buffer, size_t buflen, size_t *pn)
{
  int len = 0;
  int status = ENOENT;

  if (!entry)
    status = EINVAL;
  else
    {
      int i;
      int key_len = strlen (key);
      for (i = 0; i < entry->fields_count; i++)
	{
	  int n = mu_c_strncasecmp (entry->fields[i], key, key_len);
	  if (n == 0)
	    {
	      int field_len = strlen(entry->fields[i]);
	      status = 0;
	      if (field_len > key_len)
		{
		  int c = entry->fields[i][key_len];
		  if (mu_isspace (c) || c == '=')
		    {
		      char *value = strchr (entry->fields[i], '=');
		      if (value != NULL)
			{
			  value++; /* Pass the equal.  */
			  /* Remove leading space.  */
			  for (; mu_isspace (*value); value++)
			    ;
			  len = strlen (value);
			  /* Strip surrounding double quotes */
			  if (len > 1 && value[0] == '"' && value[len - 1] == '"')
			    {
			      value++;
			      len -= 2;
			    }
			  if (buffer != NULL && buflen > 0)
			    {
			      buflen--;
			      len = (len < buflen) ? len : buflen;
			      memcpy (buffer, value, len);
			      buffer[len] = '\0';
			    }
			  break;
			}
		    }
		}
	    }
	}
    }
  if (pn)
    *pn = len;
  return status;
}

/*
 * break the line on ';'.  Same as strtok() but
 * check for escaped "\;"
 */
static char *
tokenize (char *s, char **save_ptr)
{
  int c;
  char *token;

  if (s == NULL)
    {
      s = *save_ptr;
    }

  if (*s == '\0')
    {
      *save_ptr = s;
      return NULL;
    }

  for (token = s, c = 0; *s; s++)
    {
      if (*s == ';' && c != '\\')
	{
	  break;
	}
      c = *s;
    }

  if (*s == '\0')
    {
      *save_ptr = s;
    }
  else
    {
      *s = '\0';
      *save_ptr = s + 1;
    }
  return token;
}

/**
 * parse the mailcap line, fields are separated by ';'
 */
static int
mu_mailcap_parse_entry (mu_mailcap_entry_t entry, char *buffer)
{
  char *token = NULL;
  char *s = NULL;
  int i;
  for (i = 0, token = tokenize (buffer, &s);
       token != NULL; token = tokenize (NULL, &s), i++)
    {
      switch (i)
	{
	  /* The first entry in a mailcap line is the typefield.  */
	case 0:
	  entry->typefield = strdup (mu_str_stripws (token));
	  break;

	  /* The second entry in a mailcap line is the view-command.  */
	case 1:
	  entry->viewcommand = strdup (mu_str_stripws (token));
	  break;

	  /* The rest are the optional fields.  */
	default:
	  {
	    char **fields = realloc (entry->fields,
				     (entry->fields_count + 1) *
				     sizeof (*fields));
	    if (fields != NULL)
	      {
		entry->fields = fields;
		entry->fields[entry->fields_count] =
		  strdup (mu_str_stripws (token));
		entry->fields_count++;
	      }
	  }
	}
    }
  /* Make sure typefield and viewcommand are not null.  */
  if (entry->typefield == NULL)
    {
      entry->typefield = strdup ("");
    }
  if (entry->viewcommand == NULL)
    {
      entry->viewcommand = strdup ("");
    }
  return 0;
}

/*
 * parse a mailcap file or stream,
 * - ignore empty line.
 * - ignore line starting with '#'
 * - multiline is done with the '\' as continuation
 * example:
# comment
application/pgp; gpg < %s | metamail; needsterminal; \
 test=test %{encapsulation}=entity ; copiousoutput
 */
static int
mu_mailcap_parse (mu_mailcap_t mailcap, mu_stream_t stream)
{
  off_t off;
  int status;
  size_t n;
  char *previous;
  char *buffer;
  int buflen = 512;

  buffer = malloc (buflen * sizeof (*buffer));
  if (buffer == NULL)
    {
      return ENOMEM;
    }

  /*
   * We are doing this a little more complex then expected, because we do not
   * want to seek() back in the stream:
   * - we have to take care of continuation line i.e. line ending with '\'
   * - we have to take to account that the line may be bigger then the buffer
   *   and reallocate
   * - check the return of malloc/realloc
   * The old continuation line is saved in the "previous" pointer and
   * prepended to the buffer.
   */
  for (previous = NULL, off = n = 0;
       (status = mu_stream_readline (stream, buffer, buflen, off, &n)) == 0
	 && n > 0;
       off += n)
    {
      int len;

      /* If there is no trailing newline, that means the buffer was too small,
       * make room for the buffer and continue reading  */
      if (buffer[n - 1] != '\n')
        {
	  char *b = realloc (buffer, buflen * sizeof (*buffer));
          buflen *= 2;
          if (b == NULL)
	    {
	      status = ENOMEM;
	      break;
	    }
	  buffer = b;
	  /*
	   * Fake this as a continuation line, for simplicity.
	   */
	  strcat (buffer, "\\");
        }
      else
	{
	  /* Nuke the trailing newline.  */
	  buffer[n - 1] = '\0';
	}

      /* recalculate the len.  */
      len = strlen (buffer);

      /* Ending with a '\' means continuation line.  */
      if (len && buffer[len - 1] == '\\')
	{
	  buffer[len - 1] = '\0';
	  /*
	   * Check for any previous line:
	   * - if yes append the buffer to the previous line.
	   * - if not set the buffer as the previous line and continue.
	   */
	  if (previous == NULL)
	    {
	      previous = strdup (buffer);
	      if (previous == NULL)
		{
		  status = ENOMEM;
		  break;
		}
	    }
	  else
	    {
	      char *b = realloc (previous, strlen (previous) + len + 1);
	      if (b == NULL)
		{
		  status = ENOMEM;
		  break;
		}
	      previous = b;
	      strcat(previous, buffer);
	    }
	}
      else
	{
	  /* Did we have a previous incomplete line?
	   * if yes make one line from the previous and the buffer.
	   */
	  if (previous != NULL)
	    {
	      int prev_len = strlen (previous);
	      int total =  prev_len + len + 1;
	      if (total > buflen)
		{
		  char *b = realloc (buffer, total * sizeof (*buffer));
		  if (b == NULL)
		    {
		      status = ENOMEM;
		      break;
		    }
		  buffer = b;
		  buflen = total;
		}
	      memmove (buffer + prev_len, buffer, len + 1);
	      memcpy (buffer, previous, prev_len);
	      free (previous);
	      previous = NULL;
	    }
	}

      /* Parse the well-form mailcap line entry.  */
      if (previous == NULL) {
	/* Nuke the trailing/prepend spaces.  */
	char *line = mu_str_stripws (buffer);
	/* Ignore comments or empty lines  */
	if (*line != '#' && *line != '\0')
	  {
	    mu_mailcap_entry_t *entries;
	    entries = realloc (mailcap->entries,
			       (mailcap->entries_count + 1) *
			       sizeof (*entries));
	    if (entries != NULL)
	      {
		mailcap->entries = entries;
		mailcap->entries[mailcap->entries_count] = calloc (1,
							   sizeof(**entries));
		if (mailcap->entries[mailcap->entries_count] != NULL)
		  {
		    mu_mailcap_parse_entry (mailcap->entries[mailcap->entries_count], line);
		  }
		mailcap->entries_count++;
	      }
	    else
	      {
		status = ENOMEM;
		break;
	      }
	  }
      }
    }

  if (buffer != NULL)
    {
      free (buffer);
    }
  if (previous != NULL)
    {
      free (previous);
    }
  return status;
}

#ifdef STANDALONE_TEST
int main()
{
  mu_stream_t stream = NULL;
  int status = 0;

  status = mu_file_stream_create (&stream, "/home/alain/mailcap", MU_STREAM_READ);
  if (status == 0)
    {
      status = mu_stream_open(stream);
      if (status == 0)
	{
	  mu_mailcap_t mailcap;
	  status = mu_mailcap_create (&mailcap, stream);
	  if (status == 0)
	    {
	      int i, n;
	      size_t count = 0;
	      char buffer[256];

	      mu_mailcap_entries_count (mailcap, &count);
	      for (i = 1; i <= count; i++)
		{
		  int j;
		  mu_mailcap_entry_t entry = NULL;
		  int fields_count = 0;

		  printf ("entry[%d]\n", i);
#if 1

		  mu_mailcap_get_entry (mailcap, i, &entry);
		  /* Print typefield.  */
		  mu_mailcap_entry_get_typefield (entry, buffer,
						  sizeof (buffer), NULL);
		  printf ("\ttypefield: %s\n", buffer);

		  /* Print view-command.  */
		  mu_mailcap_entry_get_viewcommand (entry, buffer,
						    sizeof (buffer), NULL);
		  printf ("\tview-command: %s\n", buffer);

		  /* Print fields.  */
		  mu_mailcap_entry_fields_count (entry, &fields_count);
		  for (j = 1; j <= fields_count; j++)
		    {
		      mu_mailcap_entry_get_field (entry, j, buffer,
						  sizeof (buffer), NULL);
		      printf("\tfields[%d]: %s\n", j, buffer);
		    }
		  n = 0;
		  mu_mailcap_entry_get_compose (entry, buffer,
						sizeof (buffer), &n);
		  if (n > 0)
		    {
		      printf("\tcompose[%s]\n", buffer);
		    }
		  printf("\n");
		}
#else
	      for (i = 0; i < mailcap->entries_count; i++)
		{
		  int j;
		  mu_mailcap_entry_t entry = mailcap->entries[i];
		  printf("[%s];[%s]", entry->typefield, entry->viewcommand);
		  for (j = 0; j < entry->fields_count; j++)
		    {
		      printf(";[%s]", entry->fields[j]);
		    }
		  printf("\n");
		}
#endif
	      mu_mailcap_destroy (&mailcap);
	    }
	}
    }
  return 0;
}

#endif
