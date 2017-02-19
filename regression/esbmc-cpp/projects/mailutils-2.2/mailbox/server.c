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
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <mailutils/server.h>
#include <mailutils/errno.h>


struct _mu_connection
{
  struct _mu_connection *next, *prev;
  int fd;
  mu_conn_loop_fp f_loop;
  mu_conn_free_fp f_free;
  void *data;
};

#define MU_SERVER_TIMEOUT 0x1

struct _mu_server
{
  int nfd;
  fd_set fdset;
  int flags;
  struct timeval timeout;
  struct _mu_connection *head, *tail;
  mu_server_idle_fp f_idle;
  mu_server_free_fp f_free;
  void *server_data;
};

void
recompute_nfd (mu_server_t srv)
{
  struct _mu_connection *p;
  int nfd = 0;
  for (p = srv->head; p; p = p->next)
    if (p->fd > nfd)
      nfd = p->fd;
  srv->nfd = nfd + 1;
}

void
destroy_connection (mu_server_t srv, struct _mu_connection *conn)
{
  if (conn->f_free)
    conn->f_free (conn->data, srv->server_data);
  free (conn);
}

void
remove_connection (mu_server_t srv, struct _mu_connection *conn)
{
  struct _mu_connection *p;
  
  close (conn->fd);
  FD_CLR (conn->fd, &srv->fdset);

  p = conn->prev;
  if (p)
    p->next = conn->next;
  else /* we're at head */
    srv->head = conn->next;

  p = conn->next;
  if (p)
    p->prev = conn->prev;
  else /* we're at tail */
    srv->tail = conn->prev;

  if (conn->fd == srv->nfd - 1)
    recompute_nfd (srv);
  
  destroy_connection (srv, conn);
}

int
connection_loop (mu_server_t srv, fd_set *fdset)
{
  struct _mu_connection *conn;
  for (conn = srv->head; conn;)
    {
      struct _mu_connection *next = conn->next;
      if (FD_ISSET (conn->fd, fdset))
	{
	  int rc = conn->f_loop (conn->fd, conn->data, srv->server_data);
	  switch (rc)
	    {
	    case 0:
	      break;
	      
	    case MU_SERVER_CLOSE_CONN:
	    default:
	      remove_connection (srv, conn);
	      break;
	      
	    case MU_SERVER_SHUTDOWN:
	      return 1;
	    }
	}
      conn = next;
    }
  return 0;
}

void
make_fdset (mu_server_t srv)
{
  struct _mu_connection *p;
  int nfd = 0;
  
  FD_ZERO (&srv->fdset);
  for (p = srv->head; p; p = p->next)
    {
      FD_SET (p->fd, &srv->fdset);
      if (p->fd > nfd)
	nfd = p->fd;
    }
  srv->nfd = nfd + 1;
}

int
mu_server_run (mu_server_t srv)
{
  int status = 0;
  
  if (!srv)
    return EINVAL;
  if (!srv->head)
    return MU_ERR_NOENT;
  
  make_fdset (srv);  
  
  while (1)
    {
      int rc;
      fd_set rdset;
      struct timeval *to;
      
      rdset = srv->fdset;
      to = (srv->flags & MU_SERVER_TIMEOUT) ? &srv->timeout : NULL;
      rc = select (srv->nfd, &rdset, NULL, NULL, to);
      if (rc == -1 && errno == EINTR)
	{
	  if (srv->f_idle && srv->f_idle (srv->server_data))
	    break;
	  continue;
	}
      if (rc < 0)
	return errno;

      if (connection_loop (srv, &rdset))
	{
	  status = MU_ERR_FAILURE;
	  break;
	}
    }
  return status;
}

int
mu_server_create (mu_server_t *psrv)
{
  mu_server_t srv = calloc (1, sizeof (*srv));
  if (!srv)
    return ENOMEM;
  *psrv = srv;
  return 0;
}

int
mu_server_destroy (mu_server_t *psrv)
{
  mu_server_t srv;
  struct _mu_connection *p;
  
  if (!psrv)
    return EINVAL;
  srv = *psrv;
  if (!srv)
    return 0;

  for (p = srv->head; p; )
    {
      struct _mu_connection *next = p->next;
      destroy_connection (srv, p);
      p = next;
    }

  if (srv->f_free)
    srv->f_free (srv->server_data);
  
  free (srv);
  *psrv = NULL;
  return 0;
}

int
mu_server_count (mu_server_t srv, size_t *pcount)
{
  size_t n = 0;
  struct _mu_connection *p;

  if (!srv)
    return EINVAL;
  for (p = srv->head; p; p = p->next)
    n++;
  *pcount = n;
  return 0;
}

int
mu_server_set_idle (mu_server_t srv, mu_server_idle_fp fp)
{
  if (!srv)
    return EINVAL;
  srv->f_idle = fp;
  return 0;
}

int
mu_server_set_data (mu_server_t srv, void *data, mu_server_free_fp fp)
{
  if (!srv)
    return EINVAL;
  srv->server_data = data;
  srv->f_free = fp;
  return 0;
}

int
mu_server_set_timeout (mu_server_t srv, struct timeval *to)
{
  if (!srv)
    return EINVAL;
  if (!to)
    srv->flags &= ~MU_SERVER_TIMEOUT;
  else
    {
      srv->timeout = *to;
      srv->flags |= MU_SERVER_TIMEOUT;
    }
  return 0;
}

int
mu_server_add_connection (mu_server_t srv,
			  int fd, void *data,
			  mu_conn_loop_fp loop, mu_conn_free_fp free)
{
  struct _mu_connection *p;

  if (!srv || !loop)
    return EINVAL;

  p = malloc (sizeof (*p));
  if (!p)
    return ENOMEM;
  p->fd = fd;
  p->f_loop = loop;
  p->f_free = free;
  p->data = data;

  p->next = NULL;
  p->prev = srv->tail;
  if (srv->tail)
    srv->tail->next = p;
  else
    srv->head = p;
  srv->tail = p;
  return 0;
}
