/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2010 Free Software
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

#ifndef _MAILUTILS_STREAM_H
# define _MAILUTILS_STREAM_H

#include <stdio.h>
#include <stdarg.h>
#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" { /*}*/
#endif

#define MU_STREAM_READ	      0x00000001
#define MU_STREAM_WRITE	      0x00000002
#define MU_STREAM_RDWR        0x00000004
#define MU_STREAM_APPEND      0x00000008
#define MU_STREAM_CREAT	      0x00000010
#define MU_STREAM_NONBLOCK    0x00000020
/* Stream will be destroyed on mu_stream_destroy without checking the owner. */
#define MU_STREAM_NO_CHECK    0x00000040
#define MU_STREAM_SEEKABLE    0x00000080
#define MU_STREAM_NO_CLOSE    0x00000100
#define MU_STREAM_ALLOW_LINKS 0x00000200
#define MU_STREAM_NONLOCK     0x00000400
/* This one affects only mailboxes */  
#define MU_STREAM_QACCESS     0x00000800
  
#define MU_STREAM_IRGRP       0x00001000
#define MU_STREAM_IWGRP       0x00002000
#define MU_STREAM_IROTH       0x00004000
#define MU_STREAM_IWOTH       0x00008000
#define MU_STREAM_IMASK       0x0000F000
  
/* Functions useful to users of the pre-defined stream types. */

extern int mu_file_stream_create    (mu_stream_t *stream, const char* filename,
				     int flags);
extern int mu_temp_file_stream_create (mu_stream_t *stream, const char *dir);
  
extern int mu_tcp_stream_create     (mu_stream_t *stream, const char* host,
				     int port, int flags);
extern int mu_tcp_stream_create_with_source_ip (mu_stream_t *stream,
						const char *host, int port,
						unsigned long source_ip,
						int flags);
extern int mu_tcp_stream_create_with_source_host (mu_stream_t *stream,
						  const char *host, int port,
						  const char *source_host,
						  int flags);
extern int mu_socket_stream_create (mu_stream_t *stream, const char *filename,
				    int flags);
  
extern int mu_mapfile_stream_create (mu_stream_t *stream, const char* filename,
				     int flags);
extern int mu_memory_stream_create  (mu_stream_t *stream, const char* filename,
				     int flags);
extern int mu_encoder_stream_create (mu_stream_t *stream, mu_stream_t iostream,
				     const char *encoding);
extern int mu_decoder_stream_create (mu_stream_t *stream, mu_stream_t iostream,
				     const char *encoding);
extern int mu_stdio_stream_create   (mu_stream_t *stream, FILE* stdio,
				     int flags);
extern int mu_prog_stream_create    (mu_stream_t *stream, const char *progname,
				     int flags);
int mu_filter_prog_stream_create    (mu_stream_t *stream, const char *progname,
				     mu_stream_t input);
  
extern void mu_stream_destroy    (mu_stream_t *, void *owner);

extern int mu_stream_open        (mu_stream_t);
extern int mu_stream_close       (mu_stream_t);
extern int mu_stream_is_seekable (mu_stream_t);
extern int mu_stream_get_transport2 (mu_stream_t stream, mu_transport_t *pt,
				     mu_transport_t *pt2);
extern int mu_stream_get_transport (mu_stream_t stream, mu_transport_t *pt);

extern int mu_stream_read        (mu_stream_t, char *, size_t, mu_off_t, size_t *);
extern int mu_stream_readline    (mu_stream_t, char *, size_t, mu_off_t, size_t *);
extern int mu_stream_getline     (mu_stream_t is, char **pbuf, size_t *pbufsize,
				  mu_off_t offset, size_t *pnread);
extern int mu_stream_size        (mu_stream_t, mu_off_t *);
extern int mu_stream_truncate    (mu_stream_t, mu_off_t);
extern int mu_stream_write       (mu_stream_t, const char *, size_t, mu_off_t,
				  size_t *);
extern int mu_stream_setbufsiz   (mu_stream_t stream, size_t size);
extern int mu_stream_flush       (mu_stream_t);
extern int mu_stream_shutdown    (mu_stream_t stream, int how);

extern int mu_stream_vprintf     (mu_stream_t os, mu_off_t *poff, 
                                  const char *fmt, va_list ap);
extern int mu_stream_printf      (mu_stream_t stream, mu_off_t *off, 
                                  const char *fmt, ...) MU_PRINTFLIKE(3,4);
extern int mu_stream_sequential_vprintf (mu_stream_t stream, const char *fmt,
                                         va_list ap);
extern int mu_stream_sequential_printf (mu_stream_t stream, const char *fmt,
                                        ...) MU_PRINTFLIKE(2,3);

#define MU_STREAM_READY_RD 0x1
#define MU_STREAM_READY_WR 0x2
#define MU_STREAM_READY_EX 0x4  
struct timeval;  /* Needed for the following declaration */ 

extern int mu_stream_wait        (mu_stream_t stream, int *pflags, struct timeval *);

/* Functions useful to implementors of new stream types. */

extern int mu_stream_create       (mu_stream_t *stream, int flags, void *owner);

extern void* mu_stream_get_owner  (mu_stream_t);
extern void mu_stream_set_owner   (mu_stream_t, void* owner);

extern int mu_stream_get_flags    (mu_stream_t, int *pflags);
extern int mu_stream_set_flags    (mu_stream_t, int flags);
extern int mu_stream_clr_flags    (mu_stream_t, int flags);

extern int mu_stream_get_property (mu_stream_t, mu_property_t *);
extern int mu_stream_set_property (mu_stream_t, mu_property_t, void *);

#define MU_STREAM_STATE_OPEN  1
#define MU_STREAM_STATE_READ  2
#define MU_STREAM_STATE_WRITE 4
#define MU_STREAM_STATE_CLOSE 8
extern int mu_stream_get_state    (mu_stream_t, int *pstate);

extern int mu_stream_set_destroy  (mu_stream_t,
      void (*_destroy) (mu_stream_t), void *owner);

extern int mu_stream_set_open     (mu_stream_t,
      int (*_open) (mu_stream_t), void *owner);

extern int mu_stream_set_close    (mu_stream_t,
      int (*_close) (mu_stream_t), void *owner);

extern int mu_stream_set_get_transport2  (mu_stream_t,
      int (*_get_fd) (mu_stream_t, mu_transport_t *, mu_transport_t *),
      void *owner);

extern int mu_stream_set_read     (mu_stream_t,
      int (*_read) (mu_stream_t, char *, size_t, mu_off_t, size_t *),
				   void *owner);

extern int mu_stream_set_readline (mu_stream_t, 
      int (*_readline) (mu_stream_t, char *, size_t, mu_off_t, size_t *),
			        void *owner);

extern int mu_stream_set_size     (mu_stream_t,
      int (*_size) (mu_stream_t, mu_off_t *), void *owner);

extern int mu_stream_set_truncate (mu_stream_t,
      int (*_truncate) (mu_stream_t, mu_off_t), void *owner);

extern int mu_stream_set_write    (mu_stream_t,
      int (*_write) (mu_stream_t, const char *, size_t, mu_off_t, size_t *),
			    void *owner);

extern int mu_stream_set_flush    (mu_stream_t,
      int (*_flush) (mu_stream_t), void *owner);

extern int mu_stream_set_strerror (mu_stream_t stream,
      int (*fp) (mu_stream_t, const char **), void *owner);

extern int mu_stream_set_wait (mu_stream_t stream,
      int (*wait) (mu_stream_t, int *, struct timeval *), void *owner);

extern int mu_stream_set_shutdown (mu_stream_t stream,
      int (*_shutdown) (mu_stream_t, int how), void *owner);
  
extern int mu_stream_sequential_read (mu_stream_t stream,
      char *buf, size_t size, size_t *nbytes);
  
extern int mu_stream_sequential_readline (mu_stream_t stream,
      char *buf, size_t size, size_t *nbytes);

extern int mu_stream_sequential_getline  (mu_stream_t is,
      char **pbuf, size_t *pbufsize, size_t *pnread);
  
extern int mu_stream_sequential_write (mu_stream_t stream,
				    const char *buf, size_t size);
extern int mu_stream_seek (mu_stream_t stream, mu_off_t off, int whence);
  
extern int mu_stream_strerror (mu_stream_t stream, const char **p);
  
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_STREAM_H */

