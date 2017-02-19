/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2007, 2010 Free Software
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

/* Tell GLIBC that we want UNIX98 pthread_rwlock_xx() functions.  */
#define _XOPEN_SOURCE   500
/* Tell QNX/Neutrino to define pthread_rwlock_xx() functions.  */
#define _QNX_SOURCE
#define _POSIX_C_SOURCE 199506
#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#ifdef WITH_PTHREAD
#  ifdef HAVE_PTHREAD_H
#    include <pthread.h>
#  endif
#endif
#include <errno.h>
#include <stdlib.h>

#include <monitor0.h>
#include <mailutils/errno.h>

#ifdef WITH_PTHREAD
pthread_mutex_t monitor_lock = PTHREAD_MUTEX_INITIALIZER;
#  define STATIC_LOCK(m) pthread_mutex_lock(m)
#  define STATIC_UNLOCK(m) pthread_mutex_unlock(m)
#else
#  define STATIC_LOCK(m) 0
#  define STATIC_UNLOCK(m)
int monitor_lock;
#endif

union _p_lock
{
#ifdef WITH_PTHREAD
#  ifdef HAVE_PTHREAD_RWLOCK_INIT
     pthread_rwlock_t mutex;
#  else
     pthread_mutex_t mutex;
#  endif
#else
  int dummy;
#endif
};

typedef union _p_lock *p_lock_t;
static int  monitor_pthread_create (p_lock_t *);
static void monitor_pthread_destroy (p_lock_t *);
static int monitor_pthread_rdlock (p_lock_t);
static int monitor_pthread_wrlock (p_lock_t);
static int monitor_pthread_unlock (p_lock_t);

/* The idea was to have a general/portable object mu_monitor_t.
   The mu_monitor_t object could have different implementation (on the fly ?)
   of locking.  Also the rest of the library would not have to know about
   different threading implementation.  So far we've pretty much hardcoded
   the concrete implementation of monitor on pthread and read/write locks,
   but changing to a different concrete implementation will not be hard if
   the need arise.
   For static initializers we take a small penality and since we have
   a global static lock.
 */

int
mu_monitor_create (mu_monitor_t *pmonitor, int flags, void *owner)
{
  mu_monitor_t monitor;

  if (pmonitor == NULL)
    return MU_ERR_OUT_PTR_NULL;

  monitor = calloc (1, sizeof (*monitor));
  if (monitor == NULL)
    return ENOMEM;

  if (flags == MU_MONITOR_PTHREAD)
    {
      int status = monitor_pthread_create ((p_lock_t *)&(monitor->data));
      if (status != 0)
	{
	  free (monitor);
	  return status;
	}
    }
  monitor->owner = owner;
  monitor->allocated = 1;
  monitor->flags = flags;
  *pmonitor = monitor;
  return 0;
}

void *
mu_monitor_get_owner (mu_monitor_t monitor)
{
  return (monitor == NULL) ? NULL : monitor->owner;
}

void
mu_monitor_destroy (mu_monitor_t *pmonitor, void *owner)
{
  if (pmonitor && *pmonitor)
    {
      mu_monitor_t monitor = *pmonitor;
      if (monitor->owner == owner)
	{
	  if (monitor->flags == MU_MONITOR_PTHREAD)
	    monitor_pthread_destroy ((p_lock_t *)&(monitor->data));
	}
      free (monitor);
      *pmonitor = NULL;
    }
}

int
mu_monitor_rdlock (mu_monitor_t monitor)
{
  if (monitor)
    {
      if (!monitor->allocated)
	{
	  int status = STATIC_LOCK (&monitor_lock);
	  if (monitor->data == NULL)
	    {
	      if (monitor->flags == MU_MONITOR_PTHREAD)
		status = monitor_pthread_create ((p_lock_t*)&(monitor->data));
	      if (status != 0)
		{
		  STATIC_UNLOCK (&monitor_lock);
		  return status;
		}
	    }
	  monitor->allocated = 1;
	  STATIC_UNLOCK (&monitor_lock);
	}
      if (monitor->flags == MU_MONITOR_PTHREAD)
	return monitor_pthread_rdlock ((p_lock_t)monitor->data);
    }
  return 0;
}

int
mu_monitor_wrlock  (mu_monitor_t monitor)
{
  if (monitor)
    {
      if (!monitor->allocated)
	{
	  int status = STATIC_LOCK (&monitor_lock);
	  if (monitor->data == NULL)
	    {
	      if (monitor->flags == MU_MONITOR_PTHREAD)
		status = monitor_pthread_create ((p_lock_t *)&(monitor->data));
	      if (status != 0)
		{
		  STATIC_UNLOCK (&monitor_lock);
		  return status;
		}
	    }
	  monitor->allocated = 1;
	  STATIC_UNLOCK (&monitor_lock);
	}
      if (monitor->flags == MU_MONITOR_PTHREAD)
	return monitor_pthread_wrlock ((p_lock_t)monitor->data);
    }
  return 0;
}

int
mu_monitor_unlock (mu_monitor_t monitor)
{
  if (monitor)
    {
      if (monitor->flags == MU_MONITOR_PTHREAD)
	return monitor_pthread_unlock ((p_lock_t)monitor->data);
    }
  return 0;
}

int
mu_monitor_wait (mu_monitor_t monitor MU_ARG_UNUSED)
{
  return ENOSYS;
}

int
mu_monitor_notify (mu_monitor_t monitor MU_ARG_UNUSED)
{
  return ENOSYS;
}


/* Concrete Implementation of pthread base on rwlocks.  */

#ifdef WITH_PTHREAD
#  ifdef HAVE_PTHREAD_RWLOCK_INIT
#    define RWLOCK_INIT(rwl, attr)  pthread_rwlock_init (rwl, attr)
#    define RWLOCK_DESTROY(rwl)     pthread_rwlock_destroy (rwl)
#    define RWLOCK_RDLOCK(rwl)      pthread_rwlock_rdlock (rwl)
#    define RWLOCK_WRLOCK(rwl)      pthread_rwlock_wrlock (rwl)
#    define RWLOCK_UNLOCK(rwl)      pthread_rwlock_unlock (rwl)
#  else
#    define RWLOCK_INIT(rwl, attr)  pthread_mutex_init (rwl, attr)
#    define RWLOCK_DESTROY(rwl)     pthread_mutex_destroy (rwl)
#    define RWLOCK_RDLOCK(rwl)      pthread_mutex_lock (rwl)
#    define RWLOCK_WRLOCK(rwl)      pthread_mutex_lock (rwl)
#    define RWLOCK_UNLOCK(rwl)      pthread_mutex_unlock (rwl)
#  endif
#else
#  define RWLOCK_INIT(rwl, attr)    0
#  define RWLOCK_DESTROY(rwl)
#  define RWLOCK_RDLOCK(rwl)        0
#  define RWLOCK_WRLOCK(rwl)        0
#  define RWLOCK_UNLOCK(rwl)        0
#  define flockfile(arg)
#  define funlockfile(arg)
#endif



static int
monitor_pthread_create (p_lock_t *plock)
{
  int status;
  p_lock_t lock = calloc (1, sizeof (*lock));
  if (lock == NULL)
    return ENOMEM;
  status = RWLOCK_INIT (&(lock->mutex), NULL);
  if (status != 0)
    {
      free (lock);
      return status;
    }
  *plock = lock;
  return 0;
}

static void
monitor_pthread_destroy (p_lock_t *plock)
{
  p_lock_t lock = *plock;
  if (lock)
    {
      RWLOCK_DESTROY (&(lock->mutex));
      free (lock);
    }
  *plock = NULL;
}

static int
monitor_pthread_rdlock (p_lock_t lock)
{
  return RWLOCK_RDLOCK (&(lock->mutex));
}

static int
monitor_pthread_wrlock (p_lock_t lock)
{
  return RWLOCK_WRLOCK (&(lock->mutex));
}

static int
monitor_pthread_unlock (p_lock_t lock)
{
  return RWLOCK_UNLOCK (&(lock->mutex));
}
