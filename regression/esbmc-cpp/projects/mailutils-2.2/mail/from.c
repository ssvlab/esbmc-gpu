/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2007, 2009, 2010
   Free Software Foundation, Inc.

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

#include "mail.h"
#include <mu_umaxtostr.h>

#define ALIGN_UNDEF -1
#define ALIGN_RIGHT 0
#define ALIGN_LEFT  1

struct header_call_args
{
  msgset_t *mspec;
  mu_message_t msg;
  size_t cols_rest;
  char *buf;
  size_t size;
};
  
struct header_segm
{
  struct header_segm *next;
  int align;
  size_t width;
  void *data;
  char *(*get) (struct header_call_args *args, void *data);
};

void
header_ensure_space (struct header_call_args *args, size_t size)
{
  if (size > args->size)
    {
      args->buf = xrealloc (args->buf, size);
      args->size = size;
    }
}

static char *
header_buf_string_len (struct header_call_args *args, const char *str,
		       size_t len)
{
  header_ensure_space (args, len + 1);
  memcpy (args->buf, str, len);
  args->buf[len] = 0;
  return args->buf;
}

static char *
header_buf_string (struct header_call_args *args, const char *str)
{
  if (!str)
    return header_buf_string_len (args, "", 0);
  return header_buf_string_len (args, str, strlen (str));
}

static void
format_pad (size_t n)
{
  for (; n; n--)
    fputc (' ', ofile);
}

static void
format_headline (struct header_segm *seg, msgset_t *mspec, mu_message_t msg)
{
  int screen_cols = util_getcols () - 2;
  int out_cols = 0;
  struct header_call_args args;

  args.mspec = mspec;
  args.msg = msg;
  args.buf = NULL;
  args.size = 0;
  
  for (; seg; seg = seg->next)
    {
      size_t width, len;
      size_t cols_rest = screen_cols - out_cols;
      char *p;

      args.cols_rest = cols_rest;
      p = seg->get (&args, seg->data);

      if (!p)
	p = "";
      len = strlen (p);
      
      if (seg->width)
	width = seg->width;
      else
	width = len;
      if (width > cols_rest)
	width = cols_rest;

      if (len > width)
	len = width;
      
      if (seg->align == ALIGN_RIGHT)
	{
	  format_pad (width - len);
	  fprintf (ofile, "%*.*s", (int) len, (int) len, p);
	}
      else
	{
	  fprintf (ofile, "%*.*s", (int) len, (int) len, p);
	  format_pad (width - len);
	}
      out_cols += width;
    }

  fprintf (ofile, "\n");
  free (args.buf);
}    

static void
free_headline (struct header_segm *seg)
{
  while (seg)
    {
      struct header_segm *next = seg->next;
      if (seg->data)
	free (seg->data);
      free (seg);
      seg = next;
    }
}


static char *
hdr_text (struct header_call_args *args, void *data)
{
  return data;
}

static char *
hdr_cur (struct header_call_args *args, void *data)
{
  if (is_current_message (args->mspec->msg_part[0]))
    return (char*) data;
  return " ";
}

/* %a */
static char *
hdr_attr (struct header_call_args *args, void *data)
{
  mu_attribute_t attr;
  char cflag;
  
  mu_message_get_attribute (args->msg, &attr);
  
  if (mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_MBOXED))
    cflag = 'M';
  else if (mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_PRESERVED))
    cflag = 'P';
  else if (mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_SAVED))
    cflag = '*';
  else if (mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_TAGGED))
    cflag = 'T';
  else if (mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_SHOWN))
    cflag = 'R';
  else if (mu_attribute_is_recent (attr))
    cflag = 'N';
  else if (!mu_attribute_is_read (attr))
    cflag = 'U';
  else
    cflag = ' ';
  return header_buf_string_len (args, &cflag, 1);
}
    
/* %d */
static char *
hdr_date (struct header_call_args *args, void *data)
{
  char date[80];
  mu_header_t hdr;

  mu_message_get_header (args->msg, &hdr);
  
  date[0] = 0;
  if (mailvar_get (NULL, "datefield", mailvar_type_boolean, 0) == 0
      && mu_header_get_value (hdr, MU_HEADER_DATE,
			      date, sizeof (date), NULL) == 0)
    {
      time_t t;
      if (mu_parse_date (date, &t, NULL) == 0)
	strftime (date, sizeof(date), "%a %b %e %H:%M", localtime (&t));
      else
	date[0] = 0;
    }

  if (date[0] == 0)
    {
      const char *p;
      struct tm tm;
      mu_timezone tz;
      mu_envelope_t env;
      
      mu_message_get_envelope (args->msg, &env);
      if (mu_envelope_sget_date (env, &p) == 0
          && mu_parse_ctime_date_time (&p, &tm, &tz) == 0)
	strftime (date, sizeof(date), "%a %b %e %H:%M", &tm);
    }
  return header_buf_string (args, date);
}

/* %f */
static char *
hdr_from (struct header_call_args *args, void *data)
{
  char *from = NULL;
  
  if (mailvar_get (NULL, "fromfield", mailvar_type_boolean, 0) == 0)
    {  
      mu_header_t hdr;
      
      mu_message_get_header (args->msg, &hdr);
      if (mu_header_aget_value_unfold (hdr, MU_HEADER_FROM, &from) == 0)
	{
	  mu_address_t address = NULL;
	  if (mu_address_create (&address, from) == 0)
	    {
	      char *name;
	      const char *email;
	  
	      if (mu_address_sget_email (address, 1, &email) == 0)
		{
		  if (mailvar_get (NULL, "showto",
				   mailvar_type_boolean, 0) == 0
		      && mail_is_my_name (email))
		    {
		      char *tmp;

		      if (mu_header_aget_value_unfold (hdr, MU_HEADER_TO, 
						       &tmp) == 0)
			{
			  mu_address_t addr_to;
			  if (mu_address_create (&addr_to, tmp) == 0)
			    {
			      mu_address_destroy (&address);
			      address = addr_to;
			    }
			  free (tmp);
			}
		    }
		}
	      
	      if ((mu_address_aget_personal (address, 1, &name) == 0
		   && name)
		  || (mu_address_aget_email (address, 1, &name) == 0
		      && name))
		{
		  free (from);
		  from = name;
		}
	      mu_address_destroy (&address);
	    }
	}
      util_rfc2047_decode (&from);
    }
  else
    {
      mu_envelope_t env = NULL;
      const char *sender = "";
      
      if (mu_message_get_envelope (args->msg, &env) == 0)
	mu_envelope_sget_sender (env, &sender);
      from = strdup (sender);
    }

  header_buf_string (args, from);
  free (from);
  return args->buf;
}

/* %l */
static char *
hdr_lines (struct header_call_args *args, void *data)
{
  size_t m_lines;
  char buf[UINTMAX_STRSIZE_BOUND];
  mu_message_lines (args->msg, &m_lines);
  
  return header_buf_string (args, umaxtostr (m_lines, buf));
}

/* %m */
static char *
hdr_number (struct header_call_args *args, void *data)
{
  char buf[UINTMAX_STRSIZE_BOUND];
  return header_buf_string (args, umaxtostr (args->mspec->msg_part[0], buf));
}

/* %o */
static char *
hdr_size (struct header_call_args *args, void *data)
{
  size_t m_size;
  char buf[UINTMAX_STRSIZE_BOUND];
  mu_message_size (args->msg, &m_size);
  
  return header_buf_string (args, umaxtostr (m_size, buf));
}

/* %s */
static char *
hdr_subject (struct header_call_args *args, void *data)
{
  mu_header_t hdr;
  char *subj = NULL;
  
  mu_message_get_header (args->msg, &hdr);
  mu_header_aget_value_unfold (hdr, MU_HEADER_SUBJECT, &subj);
  util_rfc2047_decode (&subj);
  
  header_buf_string (args, subj);
  free (subj);
  return args->buf;
}

/* %S */
static char *
hdr_q_subject (struct header_call_args *args, void *data)
{
  mu_header_t hdr;
  char *subj = NULL;
  size_t len;

  if (args->cols_rest <= 2)
    return "\"\"";
  
  mu_message_get_header (args->msg, &hdr);
  mu_header_aget_value_unfold (hdr, MU_HEADER_SUBJECT, &subj);
  if (!subj)
    return "";
  util_rfc2047_decode (&subj);

  len = strlen (subj);
  if (len + 2 > args->cols_rest)
    len = args->cols_rest - 2;
  header_ensure_space (args, len + 3);
  args->buf[0] = '"';
  memcpy (args->buf + 1, subj, len);
  args->buf[len+1] = '"';
  args->buf[len+2] = 0;
  free (subj);
  return args->buf;
}


static struct header_segm *
new_header_segment (int align, size_t width,
		    void *data,
		    char *(*get) (struct header_call_args *, void *))
{
  struct header_segm *seg = xmalloc (sizeof (*seg));
  seg->next = NULL;
  seg->align = align;
  seg->width = width;
  seg->data = data;
  seg->get = get;
  return seg;
}

struct header_segm *
compile_headline (const char *str)
{
  struct header_segm *head = NULL, *tail = NULL;
  char *text;
  int align;
  size_t width;
  
#define ALIGN_STRING (align == ALIGN_UNDEF ? ALIGN_LEFT : ALIGN_RIGHT)
#define ALIGN_NUMBER (align == ALIGN_UNDEF ? ALIGN_RIGHT : ALIGN_LEFT)
#define ATTACH(p)				\
  do						\
    {						\
      if (!head)				\
	head = p;				\
      else					\
	tail->next = p;				\
      tail = p;					\
    }						\
  while (0)
      
  while (*str)
    {
      struct header_segm *seg;
      size_t len;
      char *p = strchr (str, '%');
      if (!p)
	len = strlen (str);
      else
	len = p - str;
      if (len)
	{
	  text = xmalloc (len + 1);
	  memcpy (text, str, len);
	  text[len] = 0;
	  seg = new_header_segment (ALIGN_LEFT, 0, text, hdr_text);
	  ATTACH (seg);
	}
      if (!p)
	break;

      str = ++p;

      if (*str == '-')
	{
	  str++;
	  align = ALIGN_LEFT;
	}
      else if (*str == '+')
	{
	  str++;
	  align = ALIGN_RIGHT;
	}
      else
	align = ALIGN_UNDEF;
      
      if (mu_isdigit (*str))
	width = strtoul (str, (char**)&str, 10);
      else
	width = 0;

      switch (*str++)
	{
	case '%':
	  seg = new_header_segment (ALIGN_LEFT, 0, xstrdup ("%"), hdr_text);
	  break;
	  
	case 'a': /* Message attributes. */
	  seg = new_header_segment (ALIGN_STRING, width, NULL, hdr_attr);
	  break;

	  /* FIXME: %c    The score of the message. */
	  
	case 'd': /* Message date */
	  seg = new_header_segment (ALIGN_STRING, width, NULL, hdr_date);
	  break;
	  
	  /* FIXME: %e    The indenting level in threaded mode. */
	  
	case 'f': /* Message sender */
	  seg = new_header_segment (ALIGN_STRING, width, NULL, hdr_from);
	  break;

	  /* FIXME: %i    The message thread structure. */
	  
	case 'l': /* The number of lines of the message */
	  seg = new_header_segment (ALIGN_NUMBER, width, NULL, hdr_lines);
	  break;
	  
	case 'm': /* Message number */
	  seg = new_header_segment (ALIGN_NUMBER, width, NULL, hdr_number);
	  break;
	  
	case 'o': /* The number of octets (bytes) in the message */
	  seg = new_header_segment (ALIGN_NUMBER, width, NULL, hdr_size);
	  break;
	  
	case 's': /* Message subject (if any) */
	  seg = new_header_segment (ALIGN_STRING, width, NULL, hdr_subject);
	  break;

	case 'S': /* Message subject (if any) in double quotes */
	  seg = new_header_segment (ALIGN_STRING, width, NULL, hdr_q_subject);
	  break;
	  
	  /* FIXME: %t    The position in threaded/sorted order. */
	  
	case '>': /* A `>' for the current message, otherwise ` ' */
	  seg = new_header_segment (ALIGN_STRING, width, xstrdup (">"), hdr_cur);
	  break;
	  
	case '<': /* A `<' for the current message, otherwise ` ' */
	  seg = new_header_segment (ALIGN_STRING, width, xstrdup ("<"), hdr_cur);
	  break;

	default:
	  mu_error (_("unknown escape: %%%c"), str[-1]);
	  len = str - p;
	  text = xmalloc (len);
	  memcpy (text, p, len-1);
	  text[len-1] = 0;
	  seg = new_header_segment (ALIGN_STRING, width, text, hdr_text);
	}
      ATTACH (seg);
    }
  return head;
#undef ALIGN_STRING
#undef ALIGN_NUMBER
#undef ATTACH
}

/* FIXME: Should it be part of struct mailvar_variable for "headline"? */
static struct header_segm *mail_header_line;

void
mail_compile_headline (struct mailvar_variable *var)
{
  free_headline (mail_header_line);
  mail_header_line = compile_headline (var->value.string);
}


/*
 * f[rom] [msglist]
 */

int
mail_from0 (msgset_t *mspec, mu_message_t msg, void *data)
{
  format_headline (mail_header_line, mspec, msg);
  return 0;
}

int
mail_from (int argc, char **argv)
{
  return util_foreach_msg (argc, argv, MSG_NODELETED|MSG_SILENT,
			   mail_from0, NULL);
}

