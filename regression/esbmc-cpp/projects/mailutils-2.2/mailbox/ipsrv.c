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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <sys/un.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <mailutils/acl.h>
#include <mailutils/server.h>
#include <mailutils/debug.h>
#include <mailutils/diag.h>
#include <mailutils/errno.h>
#include <mailutils/nls.h>


struct _mu_ip_server
{
  char *ident;
  struct sockaddr *addr;
  int addrlen;
  int fd;
  int type;
  mu_debug_t debug;
  mu_acl_t acl;
  mu_ip_server_conn_fp f_conn;
  mu_ip_server_intr_fp f_intr;
  mu_ip_server_free_fp f_free;
  void *data;
  union
  {
    struct
    {
      int backlog;
    } tcp_data;
    struct
    {
      char *buf;
      size_t bufsize;
      ssize_t rdsize;
    } udp_data;
  } v;
};

#define IDENTSTR(s) ((s)->ident ? (s)->ident : "default")

int
mu_ip_server_create (mu_ip_server_t *psrv, struct sockaddr *addr,
		     int addrlen, int type)
{
  struct _mu_ip_server *srv;
  mu_log_level_t level;

  switch (type)
    {
    case MU_IP_UDP:
    case MU_IP_TCP:
      break;
      
    default:
      return EINVAL;
    }
  
  srv = calloc (1, sizeof *srv);
  if (!srv)
    return ENOMEM;
  srv->addr = calloc (1, addrlen);
  if (!srv->addr)
    {
      free (srv);
      return ENOMEM;
    }
  memcpy (srv->addr, addr, addrlen);
  srv->addrlen = addrlen;
  srv->type = type;
  level = mu_global_debug_level ("ip_server");
  if (level)
    {
      mu_debug_create (&srv->debug, NULL);
      mu_debug_set_level (srv->debug, level);
    }
  srv->fd = -1;
  switch (type)
    {
    case MU_IP_UDP:
      srv->v.udp_data.bufsize = 4096;
      break;
      
    case MU_IP_TCP:
      srv->v.tcp_data.backlog = 4;
    }
      
  *psrv = srv;
  return 0;
}

int
mu_ip_server_destroy (mu_ip_server_t *psrv)
{
  mu_ip_server_t srv;
  if (!psrv)
    return EINVAL;
  srv = *psrv;
  if (!srv)
    return 0;
  if (srv->f_free)
    srv->f_free (srv->data);
  close (srv->fd);
  free (srv->addr);
  free (srv->ident);
  if (srv->type == MU_IP_UDP && srv->v.udp_data.buf)
    free (srv->v.udp_data.buf);
  free (srv);
  *psrv = NULL;
  return 0;
}

int
mu_ip_server_set_debug (mu_ip_server_t srv, mu_debug_t debug)
{
  if (!srv)
    return EINVAL;
  mu_debug_destroy (&srv->debug, NULL);
  srv->debug = debug;
  return 0;
}

int
mu_ip_server_get_debug (mu_ip_server_t srv, mu_debug_t *pdebug)
{
  if (!srv)
    return EINVAL;
  *pdebug = srv->debug;
  return 0;
}

int
mu_ip_server_get_type (mu_ip_server_t srv, int *ptype)
{
  if (!srv)
    return EINVAL;
  *ptype = srv->type;
  return 0;
}

int
mu_tcp_server_set_backlog (mu_ip_server_t srv, int backlog)
{
  if (!srv || srv->type != MU_IP_TCP)
    return EINVAL;
  srv->v.tcp_data.backlog = backlog;
  return 0;
}

int
mu_udp_server_get_bufsize (mu_ip_server_t srv, size_t *psize)
{
  if (!srv || srv->type != MU_IP_UDP)
    return EINVAL;
  *psize = srv->v.udp_data.bufsize;
  return 0;
}

int
mu_udp_server_set_bufsize (mu_ip_server_t srv, size_t size)
{
  if (!srv || srv->type != MU_IP_UDP)
    return EINVAL;
  srv->v.udp_data.bufsize = size;
  if (srv->v.udp_data.buf)
    {
      char *p = realloc (srv->v.udp_data.buf, size);
      if (!p)
	return ENOMEM;
      srv->v.udp_data.buf = p;
    }
  return 0;
}  

int
mu_ip_server_set_ident (mu_ip_server_t srv, const char *ident)
{
  if (!srv)
    return EINVAL;
  if (srv->ident)
    free (srv->ident);
  srv->ident = strdup (ident);
  if (!srv->ident)
    return ENOMEM;
  return 0;
}

int
mu_ip_server_set_acl (mu_ip_server_t srv, mu_acl_t acl)
{
  if (!srv)
    return EINVAL;
  srv->acl = acl;
  return 0;
}

int
mu_ip_server_set_conn (mu_ip_server_t srv, mu_ip_server_conn_fp conn)
{
  if (!srv)
    return EINVAL;
  srv->f_conn = conn;
  return 0;
}

int
mu_ip_server_set_intr (mu_ip_server_t srv, mu_ip_server_intr_fp intr)
{
  if (!srv)
    return EINVAL;
  srv->f_intr = intr;
  return 0;
}

int
mu_ip_server_set_data (mu_ip_server_t srv,
			void *data, mu_ip_server_free_fp free)
{
  if (!srv)
    return EINVAL;
  srv->data = data;
  srv->f_free = free;
  return 0;
}

int
mu_address_family_to_domain (int family)
{
  switch (family)
    {
    case AF_UNIX:
      return PF_UNIX;

    case AF_INET:
      return PF_INET;

    default:
      abort ();
    }
}

int
mu_ip_server_open (mu_ip_server_t srv)
{
  int fd;
  
  if (!srv || srv->fd != -1)
    return EINVAL;

  if (mu_debug_check_level (srv->debug, MU_DEBUG_TRACE0))
    {
      char *p = mu_sockaddr_to_astr (srv->addr, srv->addrlen);
      __MU_DEBUG2 (srv->debug, MU_DEBUG_TRACE0,
		   "opening server \"%s\" %s\n", IDENTSTR (srv),
		   p);
      free (p);
    }

  fd = socket (mu_address_family_to_domain (srv->addr->sa_family),
	       ((srv->type == MU_IP_UDP) ? SOCK_DGRAM : SOCK_STREAM), 0);
  if (fd == -1)
    {
      MU_DEBUG2 (srv->debug, MU_DEBUG_ERROR,
		 "%s: socket: %s\n", IDENTSTR (srv), mu_strerror (errno));
      return errno;
    }
  
  switch (srv->addr->sa_family)
    {
    case AF_UNIX:
      {
	struct stat st;
	struct sockaddr_un *s_un = (struct sockaddr_un *) srv->addr;
	
	if (stat (s_un->sun_path, &st))
	  {
	    if (errno != ENOENT)
	      {
		MU_DEBUG3 (srv->debug, MU_DEBUG_ERROR,
			   _("%s: file %s exists but cannot be stat'd: %s"),
			   IDENTSTR (srv),
			   s_un->sun_path,
			   mu_strerror (errno));
		return EAGAIN;
	      }
	  }
	else if (!S_ISSOCK (st.st_mode))
	  {
	    MU_DEBUG2 (srv->debug, MU_DEBUG_ERROR,
		       _("%s: file %s is not a socket"),
		       IDENTSTR (srv), s_un->sun_path);
	    return EAGAIN;
	  }
	else if (unlink (s_un->sun_path))
	  {
	    MU_DEBUG3 (srv->debug, MU_DEBUG_ERROR,
		       _("%s: cannot unlink file %s: %s"),
		       IDENTSTR (srv), s_un->sun_path, mu_strerror (errno));
	    return EAGAIN;
	  }
      }
      break;

    case AF_INET:
      {
	int t;
	
	t = 1;	 
	setsockopt (fd, SOL_SOCKET, SO_REUSEADDR, &t, sizeof (t));
      }
    }
  
  if (bind (fd, srv->addr, srv->addrlen) == -1)
    {
      MU_DEBUG2 (srv->debug, MU_DEBUG_ERROR,
		 "%s: bind: %s\n", IDENTSTR (srv), mu_strerror (errno));
      close (fd);
      return errno;
    }

  if (srv->type == MU_IP_TCP)
    {
      if (listen (fd, srv->v.tcp_data.backlog) == -1) 
	{
	  MU_DEBUG2 (srv->debug, MU_DEBUG_ERROR,
		     "%s: listen: %s\n", IDENTSTR (srv), mu_strerror (errno));
	  close (fd);
	  return errno;
	}
    }
  
  srv->fd = fd;
  return 0;
}

int
mu_ip_server_shutdown (mu_ip_server_t srv)
{
  if (!srv || srv->fd != -1)
    return EINVAL;
  if (mu_debug_check_level (srv->debug, MU_DEBUG_TRACE0))
    {
      char *p = mu_sockaddr_to_astr (srv->addr, srv->addrlen);
      __MU_DEBUG2 (srv->debug, MU_DEBUG_TRACE0,
		   "closing server \"%s\" %s\n", IDENTSTR (srv),
		   p);
      free (p);
    }
  close (srv->fd);
  return 0;
}

int
mu_ip_tcp_accept (mu_ip_server_t srv, void *call_data)
{
  int rc;
  int connfd;
  union
  {
    struct sockaddr sa;
    struct sockaddr_in s_in;
    struct sockaddr_un s_un;
  } client;
  
  socklen_t size = sizeof (client);
  
  if (!srv || srv->fd == -1 || srv->type == MU_IP_UDP)
    return EINVAL;

  connfd = accept (srv->fd, &client.sa, &size);
  if (connfd == -1)
    {
      int ec = errno;
      if (ec == EINTR)
	{
	  if (srv->f_intr && srv->f_intr (srv->data, call_data))
	    mu_ip_server_shutdown (srv);
	}
      return ec;
    }

  if (srv->acl)
    {
      mu_acl_result_t res;
      int rc = mu_acl_check_sockaddr (srv->acl, &client.sa, size, &res);
      if (rc)
	MU_DEBUG2 (srv->debug, MU_DEBUG_ERROR,
		   "%s: mu_acl_check_sockaddr: %s\n",
		   IDENTSTR (srv), strerror (rc));
      if (res == mu_acl_result_deny)
	{
	  char *p = mu_sockaddr_to_astr (&client.sa, size);
	  mu_diag_output (MU_DIAG_INFO, "Denying connection from %s", p);
	  free (p);
	  
	  close (connfd);
	  return 0;
	}
    }
  rc = srv->f_conn (connfd, &client.sa, size, srv->data, call_data, srv);
  close (connfd);
  return rc;
}

int
mu_ip_udp_accept (mu_ip_server_t srv, void *call_data)
{
  int rc;
  union
  {
    struct sockaddr sa;
    struct sockaddr_in s_in;
    struct sockaddr_un s_un;
  } client;
  fd_set rdset;
  
  socklen_t salen = sizeof (client);
  ssize_t size;

  if (!srv->v.udp_data.buf)
    {
      srv->v.udp_data.buf = malloc (srv->v.udp_data.bufsize);
      if (!srv->v.udp_data.buf)
	return ENOMEM;
    }
  
  FD_ZERO (&rdset);
  FD_SET (srv->fd, &rdset);
  for (;;)
    {
      rc = select (srv->fd + 1, &rdset, NULL, NULL, NULL);
      if (rc == -1)
	{
	  if (errno == EINTR)
	    {
	      if (srv->f_intr && srv->f_intr (srv->data, call_data))
		break;
	      else
		continue;
	    }
	}
      else
	break;
    }

  if (rc == -1)
    return errno;

  size = recvfrom (srv->fd, srv->v.udp_data.buf, srv->v.udp_data.bufsize,
		   0, &client.sa, &salen);
  if (size < 0)
    {
      MU_DEBUG2 (srv->debug, MU_DEBUG_ERROR,
		 "%s: recvfrom: %s",
		 IDENTSTR (srv), strerror (errno));
      return MU_ERR_FAILURE;
    }
  srv->v.udp_data.rdsize = size;
  
  if (srv->acl)
    {
      mu_acl_result_t res;
      int rc = mu_acl_check_sockaddr (srv->acl, &client.sa, size, &res);
      if (rc)
	MU_DEBUG2 (srv->debug, MU_DEBUG_ERROR,
		   "%s: mu_acl_check_sockaddr: %s\n",
		   IDENTSTR (srv), strerror (rc));
      if (res == mu_acl_result_deny)
	{
	  char *p = mu_sockaddr_to_astr (srv->addr, srv->addrlen);
	  mu_diag_output (MU_DIAG_INFO, "Denying connection from %s", p);
	  free (p);
	  return 0;
	}
    }
  rc = srv->f_conn (-1, &client.sa, size, srv->data, call_data, srv);
  return rc;
}

int
mu_ip_server_accept (mu_ip_server_t srv, void *call_data)
{
  int rc;
  if (!srv || srv->fd == -1)
    return EINVAL;
  switch (srv->type)
    {
    case MU_IP_UDP:
      rc = mu_ip_udp_accept (srv, call_data);
      break;

    case MU_IP_TCP:
      rc = mu_ip_tcp_accept (srv, call_data);
    }
  
  if (rc)
    mu_ip_server_shutdown (srv);
  return rc;
}

int
mu_ip_server_loop (mu_ip_server_t srv, void *call_data)
{
  if (!srv)
    return EINVAL;
  while (srv->fd != -1)
    {
      int rc = mu_ip_server_accept (srv, call_data);
      if (rc && rc != EINTR)
	{
	  mu_ip_server_shutdown (srv);
	  return rc;
	}
    }
  return 0;
}

int
mu_ip_server_get_fd (mu_ip_server_t srv)
{
  return srv->fd;
}

int
mu_udp_server_get_rdata (mu_ip_server_t srv, char **pbuf, size_t *pbufsize)
{
  if (!srv || srv->type != MU_IP_UDP)
    return EINVAL;
  *pbuf = srv->v.udp_data.buf;
  *pbufsize = srv->v.udp_data.rdsize;
  return 0;
}

int
mu_ip_server_get_sockaddr (mu_ip_server_t srv, struct sockaddr *s, int *size)
{
  int len;
  
  if (!srv || !s)
    return EINVAL;
  if (s == 0)
    len = srv->addrlen;
  else
    {
      len = srv->addrlen;
      if (*size < len)
	return MU_ERR_BUFSPACE;
      memcpy (s, srv->addr, len);
    }
  *size = len;
  return 0;
}
  
