/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; If not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef _MAILUTILS_ACL_H
#define _MAILUTILS_ACL_H

#include <mailutils/types.h>
#include <mailutils/iterator.h>

typedef enum mu_acl_action
  {
    mu_acl_accept,
    mu_acl_deny,
    mu_acl_log,
    mu_acl_exec,
    mu_acl_ifexec
  }
mu_acl_action_t;

typedef enum mu_acl_result
  {
    mu_acl_result_undefined,
    mu_acl_result_accept,
    mu_acl_result_deny
  }
mu_acl_result_t;

struct sockaddr;
struct in_addr;

int mu_acl_create (mu_acl_t *acl);
int mu_acl_destroy (mu_acl_t *acl);
int mu_acl_count (mu_acl_t acl, size_t *pcount);
int mu_acl_get_debug (mu_acl_t acl, mu_debug_t *pdebug);
int mu_acl_set_debug (mu_acl_t acl, mu_debug_t debug);
int mu_acl_get_iterator (mu_acl_t acl, mu_iterator_t *pitr);
int mu_acl_append (mu_acl_t acl, mu_acl_action_t act, void *data,
		   struct sockaddr *sa, int salen,
		   unsigned long netmask);
int mu_acl_prepend (mu_acl_t acl, mu_acl_action_t act, void *data,
		    struct sockaddr *sa, int salen, unsigned long netmask);
int mu_acl_insert (mu_acl_t acl, size_t pos, int before, 
		   mu_acl_action_t act, void *data,
		   struct sockaddr *sa, int salen, unsigned long netmask);

int mu_acl_check_ipv4 (mu_acl_t acl, unsigned int addr, mu_acl_result_t *pres);
int mu_acl_check_inaddr (mu_acl_t acl, const struct in_addr *inp,
			 mu_acl_result_t *pres);
int mu_acl_check_sockaddr (mu_acl_t acl, const struct sockaddr *sa, int salen,
			   mu_acl_result_t *pres);
int mu_acl_check_fd (mu_acl_t acl, int fd, mu_acl_result_t *pres);

int mu_acl_action_to_string (mu_acl_action_t act, const char **pstr);
int mu_acl_string_to_action (const char *str, mu_acl_action_t *pres);

#endif
