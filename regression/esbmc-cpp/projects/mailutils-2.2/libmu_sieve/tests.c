/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  
#include <sieve-priv.h>

typedef int (*address_aget_t) (mu_address_t addr, size_t no, char **buf);

static int
_get_address_part (void *item, void *data)
{
  mu_sieve_runtime_tag_t *t = item;
  address_aget_t ret = NULL;
  
  if (strcmp (t->tag, "all") == 0)
    ret =  mu_address_aget_email;
  else if (strcmp (t->tag, "domain") == 0)
    ret =  mu_address_aget_domain;
  else if (strcmp (t->tag, "localpart") == 0)
    ret = mu_address_aget_local_part;
  if (ret)
    {	  
      *(address_aget_t*)data = ret;
      return 1; /* break the loop */
    }  
  return 0; /* continue */
}

address_aget_t
sieve_get_address_part (mu_list_t tags)
{
  address_aget_t ret = mu_address_aget_email;
  mu_list_do (tags, _get_address_part, &ret);
  return ret;
}

/* Structure shared between address and envelope tests */
struct address_closure
{
  address_aget_t aget;     /* appropriate address_aget_ function */
  void *data;              /* Either mu_header_t or mu_envelope_t */
  mu_address_t addr;       /* Obtained address */
};  

static int
do_count (mu_list_t args, mu_list_t tags, size_t count, int retval)
{
  mu_sieve_value_t *arg;
  
  if (mu_sieve_tag_lookup (tags, "count", &arg))
    {
      size_t limit;
      char *str;
      mu_sieve_value_t *val;
      mu_sieve_relcmpn_t stest;
      
      val = mu_sieve_value_get (args, 1);
      mu_list_get (val->v.list, 0, (void **) &str);
      limit = strtoul (str, &str, 10);

      mu_sieve_str_to_relcmp (arg->v.string, NULL, &stest);
      return stest (count, limit);
    }
  return retval;
}

int
retrieve_address (void *item, void *data, int idx, char **pval)
{
  struct address_closure *ap = data;
  char *val;
  int rc;
      
  if (!ap->addr)
    {
      if (mu_header_aget_value ((mu_header_t)ap->data, (char*)item, &val))
	return 1;
      rc = mu_address_create (&ap->addr, val);
      free (val);
      if (rc)
	return rc;
    }

  rc = ap->aget (ap->addr, idx+1, pval);
  if (rc)
    mu_address_destroy (&ap->addr);
  return rc;
}

int
sieve_test_address (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  mu_sieve_value_t *h, *v;
  mu_header_t header = NULL;
  mu_sieve_comparator_t comp = mu_sieve_get_comparator (mach, tags);
  mu_sieve_relcmp_t test = mu_sieve_get_relcmp (mach, tags);
  struct address_closure clos;
  int rc;
  size_t count;
  
  if (mach->debug_level & MU_SIEVE_DEBUG_TRACE)
    mu_sieve_debug (mach, "%s:%lu: ADDRESS\n",
		 mach->locus.source_file,
		 (unsigned long) mach->locus.source_line);

  h = mu_sieve_value_get (args, 0);
  if (!h)
    {
      mu_sieve_arg_error (mach, 1);
      mu_sieve_abort (mach);
    }
  v = mu_sieve_value_get (args, 1);
  if (!v)
    {
      mu_sieve_arg_error (mach, 2);
      mu_sieve_abort (mach);
    }

  mu_message_get_header (mu_sieve_get_message (mach), &header);
  clos.data = header;
  clos.aget = sieve_get_address_part (tags);
  clos.addr = NULL;
  rc = mu_sieve_vlist_compare (h, v, comp, test, retrieve_address, &clos, &count);
  mu_address_destroy (&clos.addr);
  
  return do_count (args, tags, count, rc);
}

struct header_closure {
  mu_header_t header;
  int index;
};

int
retrieve_header (void *item, void *data, int idx, char **pval)
{
  struct header_closure *hc = data;
  char buf[512];
  size_t n;
  
  if (idx == 0)
    hc->index = 1;

  while (!mu_header_get_field_name (hc->header, hc->index, buf, sizeof(buf), &n))
    {
      int i = hc->index++;
      if (mu_c_strcasecmp (buf, (char*)item) == 0)
	{
	  if (mu_header_aget_field_value_unfold (hc->header, i, pval))
	    return 1;
	  return 0;
	}
    }

  return 1;
}

int
sieve_test_header (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  mu_sieve_value_t *h, *v;
  mu_sieve_comparator_t comp = mu_sieve_get_comparator (mach, tags);
  mu_sieve_relcmp_t test = mu_sieve_get_relcmp (mach, tags);
  size_t count, mcount = 0;
  struct header_closure clos;
  
  if (mach->debug_level & MU_SIEVE_DEBUG_TRACE)
    mu_sieve_debug (mach, "%s:%lu: HEADER\n",
		 mach->locus.source_file,
		 (unsigned long) mach->locus.source_line);

  h = mu_sieve_value_get (args, 0);
  if (!h)
    {
      mu_sieve_arg_error (mach, 1);
      mu_sieve_abort (mach);
    }
  v = mu_sieve_value_get (args, 1);
  if (!v)
    {
      mu_sieve_arg_error (mach, 2);
      mu_sieve_abort (mach);
    }

  if (mu_sieve_tag_lookup (tags, "mime", NULL))
    {
      int ismime = 0;

      mu_message_is_multipart (mach->msg, &ismime);
      if (ismime)
	{
	  size_t i, nparts = 0;
	  
	  mu_message_get_num_parts (mach->msg, &nparts);
	  for (i = 1; i <= nparts; i++)
	    {
	      mu_message_t message = NULL;

	      if (mu_message_get_part (mach->msg, i, &message) == 0)
		{
		  mu_message_get_header (message, &clos.header);
		  if (mu_sieve_vlist_compare (h, v, comp, test,
					   retrieve_header, &clos, &mcount))
		    return 1;
		}
	    }
	}
    }
  mu_message_get_header (mach->msg, &clos.header);
  if (mu_sieve_vlist_compare (h, v, comp, test, retrieve_header, &clos, &count))
    return 1;

  return do_count (args, tags, count + mcount, 0);
}

int
retrieve_envelope (void *item, void *data, int idx, char **pval)
{
  struct address_closure *ap = data;
  int rc;
      
  if (!ap->addr)
    {
      const char *buf;
      
      if (mu_c_strcasecmp ((char*)item, "from") != 0)
	return 1;

      if (mu_envelope_sget_sender ((mu_envelope_t)ap->data, &buf))
	return 1;

      rc = mu_address_create (&ap->addr, buf);
      if (rc)
	return rc;
    }

  rc = ap->aget (ap->addr, idx+1, pval);
  if (rc)
    mu_address_destroy (&ap->addr);
  return rc;
}

int
sieve_test_envelope (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  mu_sieve_value_t *h, *v;
  mu_sieve_comparator_t comp = mu_sieve_get_comparator (mach, tags);
  mu_sieve_relcmp_t test = mu_sieve_get_relcmp (mach, tags);
  struct address_closure clos;
  int rc;
  size_t count;
  
  if (mach->debug_level & MU_SIEVE_DEBUG_TRACE)
    mu_sieve_debug (mach, "%s:%lu: ENVELOPE\n",
		 mach->locus.source_file,
		 (unsigned long) mach->locus.source_line);

  h = mu_sieve_value_get (args, 0);
  if (!h)
    {
      mu_sieve_arg_error (mach, 1);
      mu_sieve_abort (mach);
    }
  v = mu_sieve_value_get (args, 1);
  if (!v)
    {
      mu_sieve_arg_error (mach, 2);
      mu_sieve_abort (mach);
    }

  mu_message_get_envelope (mu_sieve_get_message (mach), 
                           (mu_envelope_t*)&clos.data);
  clos.aget = sieve_get_address_part (tags);
  clos.addr = NULL;
  rc = mu_sieve_vlist_compare (h, v, comp, test, retrieve_envelope, &clos,
			    &count);
  mu_address_destroy (&clos.addr);
  return do_count (args, tags, count, rc);
}

int
sieve_test_size (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  int rc = 1;
  mu_sieve_runtime_tag_t *tag = NULL;
  size_t size;
  
  mu_sieve_value_t *val = mu_sieve_value_get (args, 0);
  if (!val)
    {
      mu_sieve_arg_error (mach, 1);
      mu_sieve_abort (mach);
    }

  mu_message_size (mu_sieve_get_message (mach), &size);
  mu_list_get (tags, 0, (void **)&tag);
  if (!tag)
    rc = size == val->v.number;
  else if (strcmp (tag->tag, "over") == 0)
    rc = size > val->v.number;
  else if (strcmp (tag->tag, "under") == 0)
    rc = size < val->v.number;

  return rc;
}

int
sieve_test_true (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  if (mach->debug_level & MU_SIEVE_DEBUG_TRACE)
    mu_sieve_debug (mach, "%s:%lu: TRUE\n", mach->locus.source_file,
		 (unsigned long) mach->locus.source_line);
  return 1;
}

int
sieve_test_false (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  if (mach->debug_level & MU_SIEVE_DEBUG_TRACE)
    mu_sieve_debug (mach, "%s:%lu: FALSE\n", mach->locus.source_file,
		 (unsigned long) mach->locus.source_line);
  return 0;
}

int
_test_exists (void *item, void *data)
{
  mu_header_t hdr = data;
  size_t n;

  return mu_header_get_value (hdr, (char*)item, NULL, 0, &n);
}
  
int
sieve_test_exists (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  mu_header_t header = NULL;
  mu_sieve_value_t *val;   

  if (mach->debug_level & MU_SIEVE_DEBUG_TRACE)
    mu_sieve_debug (mach, "%s:%lu: EXISTS\n", mach->locus.source_file,
		 (unsigned long) mach->locus.source_line);

  mu_message_get_header (mu_sieve_get_message (mach), &header);
  val = mu_sieve_value_get (args, 0);
  if (!val)
    {
      mu_sieve_arg_error (mach, 1);
      mu_sieve_abort (mach);
    }

  return mu_sieve_vlist_do (val, _test_exists, header) == 0;
}

static mu_sieve_tag_def_t address_part_tags[] = {
  { "localpart", SVT_VOID },
  { "domain", SVT_VOID },
  { "all", SVT_VOID },
  { NULL }
};

static mu_sieve_tag_def_t match_part_tags[] = {
  { "is", SVT_VOID },
  { "contains", SVT_VOID },
  { "matches", SVT_VOID },
  { "regex", SVT_VOID },
  { "count", SVT_STRING },
  { "value", SVT_STRING },
  { "comparator", SVT_STRING },
  { NULL }
};

static mu_sieve_tag_def_t size_tags[] = {
  { "over", SVT_VOID },
  { "under", SVT_VOID },
  { NULL }
};

static mu_sieve_tag_def_t mime_tags[] = {
  { "mime", SVT_VOID },
  { NULL }
};

#define ADDRESS_PART_GROUP \
  { address_part_tags, NULL }

#define MATCH_PART_GROUP \
  { match_part_tags, mu_sieve_match_part_checker }

#define SIZE_GROUP { size_tags, NULL }

#define MIME_GROUP \
  { mime_tags, NULL }

mu_sieve_tag_group_t address_tag_groups[] = {
  ADDRESS_PART_GROUP,
  MATCH_PART_GROUP,
  { NULL }
};

mu_sieve_data_type address_req_args[] = {
  SVT_STRING_LIST,
  SVT_STRING_LIST,
  SVT_VOID
};

mu_sieve_tag_group_t size_tag_groups[] = {
  SIZE_GROUP,
  { NULL }
};

mu_sieve_data_type size_req_args[] = {
  SVT_NUMBER,
  SVT_VOID
};

mu_sieve_tag_group_t envelope_tag_groups[] = {
  ADDRESS_PART_GROUP,
  MATCH_PART_GROUP,
  { NULL }
};

mu_sieve_data_type exists_req_args[] = {
  SVT_STRING_LIST,
  SVT_VOID
};

mu_sieve_tag_group_t header_tag_groups[] = {
  MATCH_PART_GROUP,
  MIME_GROUP,
  { NULL }
};

void
mu_sv_register_standard_tests (mu_sieve_machine_t mach)
{
  mu_sieve_register_test (mach, "false", sieve_test_false, NULL, NULL, 1);
  mu_sieve_register_test (mach, "true", sieve_test_true, NULL, NULL, 1);
  mu_sieve_register_test (mach, "address", sieve_test_address,
		       address_req_args, address_tag_groups, 1);
  mu_sieve_register_test (mach, "size", sieve_test_size,
		       size_req_args, size_tag_groups, 1);
  mu_sieve_register_test (mach, "envelope", sieve_test_envelope,
		       address_req_args, envelope_tag_groups, 1);
  mu_sieve_register_test (mach, "exists", sieve_test_exists,
		       exists_req_args, NULL, 1);
  mu_sieve_register_test (mach, "header", sieve_test_header,
		       address_req_args, header_tag_groups, 1);
}
