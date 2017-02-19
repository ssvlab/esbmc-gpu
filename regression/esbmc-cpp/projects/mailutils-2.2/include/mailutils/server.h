/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2008, 2010 Free Software Foundation, Inc.

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

#ifndef _MAILUTILS_SERVER_H
#define _MAILUTILS_SERVER_H

#include <mailutils/types.h>
#include <signal.h>

typedef int (*mu_conn_loop_fp) (int fd, void *conn_data, void *server_data);
typedef void (*mu_conn_free_fp) (void *conn_data, void *server_data);
typedef int (*mu_server_idle_fp) (void *server_data);
typedef void (*mu_server_free_fp) (void *server_data);

#define MU_SERVER_SUCCESS    0
#define MU_SERVER_CLOSE_CONN 1
#define MU_SERVER_SHUTDOWN   2

int mu_server_run (mu_server_t srv);
int mu_server_create (mu_server_t *psrv);
int mu_server_destroy (mu_server_t *psrv);
int mu_server_set_idle (mu_server_t srv, mu_server_idle_fp fp);
int mu_server_set_data (mu_server_t srv, void *data, mu_server_free_fp fp);
int mu_server_add_connection (mu_server_t srv,
			      int fd, void *data,
			      mu_conn_loop_fp loop, mu_conn_free_fp free);
struct timeval;
int mu_server_set_timeout (mu_server_t srv, struct timeval *to);
int mu_server_count (mu_server_t srv, size_t *pcount);


/* IP (TCP and UDP) server */
struct sockaddr;
typedef int (*mu_ip_server_conn_fp) (int fd, struct sockaddr *s, int len,
				     void *server_data, void *call_data,
				     mu_ip_server_t srv);
typedef int (*mu_ip_server_intr_fp) (void *data, void *call_data);
typedef void (*mu_ip_server_free_fp) (void *data);

#define MU_IP_TCP 0
#define MU_IP_UDP 1

int mu_ip_server_create (mu_ip_server_t *psrv, struct sockaddr *addr,
			 int len, int type);
int mu_ip_server_destroy (mu_ip_server_t *psrv);
int mu_ip_server_get_type (mu_ip_server_t srv, int *ptype);
int mu_ip_server_set_debug (mu_ip_server_t srv, mu_debug_t debug);
int mu_ip_server_get_debug (mu_ip_server_t srv, mu_debug_t *pdebug);
int mu_ip_server_set_ident (mu_ip_server_t srv, const char *ident);
int mu_ip_server_set_acl (mu_ip_server_t srv, mu_acl_t acl);
int mu_ip_server_set_conn (mu_ip_server_t srv, mu_ip_server_conn_fp conn);
int mu_ip_server_set_intr (mu_ip_server_t srv, mu_ip_server_intr_fp intr);
int mu_ip_server_set_data (mu_ip_server_t srv,
			    void *data, mu_ip_server_free_fp free);
int mu_ip_server_open (mu_ip_server_t srv);
int mu_ip_server_shutdown (mu_ip_server_t srv);
int mu_ip_server_accept (mu_ip_server_t srv, void *call_data);
int mu_ip_server_loop (mu_ip_server_t srv, void *call_data);
int mu_ip_server_get_fd (mu_ip_server_t srv);
int mu_ip_server_get_sockaddr (mu_ip_server_t srv, struct sockaddr *s,
			       int *size);

int mu_tcp_server_set_backlog (mu_ip_server_t srv, int backlog);
int mu_udp_server_set_bufsize (mu_ip_server_t srv, size_t size);
int mu_udp_server_get_bufsize (mu_ip_server_t srv, size_t *psize);
int mu_udp_server_get_rdata (mu_ip_server_t srv, char **pbuf,
			     size_t *pbufsize);


/* m-server */
typedef struct mu_m_server_connect_data mu_m_server_connect_data_t;
typedef int (*mu_m_server_conn_fp) (int fd, struct sockaddr *sa, int salen,
				    void *data, mu_ip_server_t srv,
				    time_t timeout, int transcript);
typedef int (*mu_m_server_prefork_fp) (int, void *,
				       struct sockaddr *s, int size);

void mu_m_server_create (mu_m_server_t *psrv, const char *ident);
void mu_m_server_destroy (mu_m_server_t *pmsrv);
void mu_m_server_set_mode (mu_m_server_t srv, int mode);
void mu_m_server_set_type (mu_m_server_t srv, int type);
void mu_m_server_get_type (mu_m_server_t srv, int *ptype);
void mu_m_server_set_conn (mu_m_server_t srv, mu_m_server_conn_fp f);
void mu_m_server_set_prefork (mu_m_server_t srv, mu_m_server_prefork_fp fun);
void mu_m_server_set_data (mu_m_server_t srv, void *data);
void mu_m_server_set_max_children (mu_m_server_t srv, size_t num);
int mu_m_server_set_pidfile (mu_m_server_t srv, const char *pidfile);
int mu_m_server_set_foreground (mu_m_server_t srv, int enable);
void mu_m_server_set_default_port (mu_m_server_t srv, int port);
void mu_m_server_set_default_address (mu_m_server_t srv, struct sockaddr *sa,
				      int salen);
void mu_m_server_set_timeout (mu_m_server_t srv, time_t t);
void mu_m_server_set_mode (mu_m_server_t srv, int mode);
void mu_m_server_set_sigset (mu_m_server_t srv, sigset_t *sigset);
void mu_m_server_set_strexit (mu_m_server_t srv, const char *(*fun) (int));

int mu_m_server_mode (mu_m_server_t srv);
int mu_m_server_foreground (mu_m_server_t srv);
time_t mu_m_server_timeout (mu_m_server_t srv);
const char * mu_m_server_pidfile (mu_m_server_t srv);
void mu_m_server_get_sigset (mu_m_server_t srv, sigset_t *sigset);
int mu_m_server_get_srvlist (mu_m_server_t srv, mu_list_t *plist);
int mu_m_server_get_default_address (mu_m_server_t srv, struct sockaddr *sa,
				     int *salen);

void mu_m_server_configured_count (mu_m_server_t msrv, size_t count);

void mu_m_server_begin (mu_m_server_t msrv);
int mu_m_server_run (mu_m_server_t msrv);
void mu_m_server_end (mu_m_server_t msrv);

void mu_m_server_stop (int code);
int mu_m_server_check_acl (mu_m_server_t msrv, struct sockaddr *s, int salen);

void mu_m_server_cfg_init (void);


#endif
