/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2005, 2007, 2008, 2010 Free Software Foundation,
   Inc.

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

#ifndef _MAILUTILS_TLS_H
#define _MAILUTILS_TLS_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mu_tls_module_config
{
  int enable;
  char *ssl_cert;
  char *ssl_key;
  char *ssl_cafile;
};

extern int mu_tls_module_init (enum mu_gocs_op, void *);

extern int mu_tls_stream_create (mu_stream_t *stream, 
				 mu_stream_t strin, mu_stream_t strout,
				 int flags);
extern int mu_tls_stream_create_client (mu_stream_t *stream,
					mu_stream_t strin, mu_stream_t strout,
					int flags);
extern int mu_tls_stream_create_client_from_tcp (mu_stream_t *stream,
						 mu_stream_t tcp_str,
						 int flags);

extern int mu_check_tls_environment (void);
extern int mu_init_tls_libs (void);
extern void mu_deinit_tls_libs (void);

typedef int (*mu_tls_readline_fn) (void *iodata);
typedef int (*mu_tls_writeline_fn) (void *iodata, char *buf); 
typedef void (*mu_tls_stream_ctl_fn) (void *iodata, mu_stream_t *pold,
				      mu_stream_t __new);

extern int mu_tls_begin (void *iodata, mu_tls_readline_fn reader,
			 mu_tls_writeline_fn writer,
			 mu_tls_stream_ctl_fn stream_ctl,
			 char *keywords[]);

extern int mu_tls_enable;
  
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_TLS_H */

