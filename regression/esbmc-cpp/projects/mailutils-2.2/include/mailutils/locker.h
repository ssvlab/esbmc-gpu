/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2008, 2010 Free Software
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

#ifndef _MAILUTILS_LOCKER_H
#define _MAILUTILS_LOCKER_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* lock expiry time */
#define MU_LOCKER_EXPIRE_TIME        (10 * 60)
#define MU_LOCKER_RETRIES            (10)
#define MU_LOCKER_RETRY_SLEEP        (1)
#define MU_LOCKER_EXTERNAL_PROGRAM   "dotlock"

/* return codes for the external locker */
#define MU_DL_EX_PERM    4 /* insufficient permissions */
#define MU_DL_EX_EXIST   3 /* lock requested, but file is already locked */
#define MU_DL_EX_NEXIST  2 /* unlock requested, but file is not locked */
#define MU_DL_EX_ERROR   1 /* failed due to some other error */
#define MU_DL_EX_OK      0 /* success */

enum mu_locker_set_mode
  {
    mu_locker_assign,
    mu_locker_set_bit,
    mu_locker_clear_bit
  };
    
/* mu_locker_create() flags */

/* Locker types */

#define MU_LOCKER_TYPE_DOTLOCK  0
#define MU_LOCKER_TYPE_EXTERNAL 1 
  /* Use an external program to lock the file. This is necessary
     for programs having permission to access a file, but do not
     have write permission on the directory that contains that file. */
#define MU_LOCKER_TYPE_KERNEL   2
  /* Use kernel locking (flock, lockf or ioctl) */
#define MU_LOCKER_TYPE_NULL     3
  /* Special locker type: means no lock. This is to be used with
     temporary mailboxes stored in memory. */

#define MU_LOCKER_TYPE_TO_FLAG(t) ((t) << 8)
#define MU_LOCKER_FLAG_TO_TYPE(f) ((f) >> 8)
#define MU_LOCKER_IS_TYPE(f,t) (MU_LOCKER_FLAG_TO_TYPE(f) == (t))
#define MU_LOCKER_SET_TYPE(f,t) ((f) = MU_LOCKER_TYPE_TO_FLAG(t) | MU_LOCKER_OPTIONS(f))
#define MU_LOCKER_TYPE_MASK 0xff00
#define MU_LOCKER_OPTION_MASK 0x00ff  
#define MU_LOCKER_OPTIONS(f) ((f) & MU_LOCKER_OPTION_MASK)

#define MU_LOCKER_NULL          MU_LOCKER_TYPE_TO_FLAG(MU_LOCKER_TYPE_NULL)
#define MU_LOCKER_DOTLOCK       MU_LOCKER_TYPE_TO_FLAG(MU_LOCKER_TYPE_DOTLOCK)
#define MU_LOCKER_EXTERNAL      MU_LOCKER_TYPE_TO_FLAG(MU_LOCKER_TYPE_EXTERNAL)
#define MU_LOCKER_KERNEL        MU_LOCKER_TYPE_TO_FLAG(MU_LOCKER_TYPE_KERNEL)
  
/* Options */
  
#define MU_LOCKER_SIMPLE   0x0000
  /* Just try and dotlock the file, not the default because its usually
     better to retry. */
#define MU_LOCKER_RETRY    0x0001
  /* This requests that we loop retries times, sleeping retry_sleep
     seconds in between trying to obtain the lock before failing with
     MU_LOCK_CONFLICT. */
#define MU_LOCKER_TIME     0x0002
  /* This mode checks the last update time of the lock, then removes
     it if older than MU_LOCKER_EXPIRE_TIME. If a client uses this,
     then the servers better periodically update the lock on the
     file... do they? */
#define MU_LOCKER_PID      0x0004
  /* PID locking is only useful for programs that aren't using
     an external dotlocker, non-setgid programs will use a dotlocker,
     which locks and exits imediately. This is a protection against
     a server crashing, it's not generally useful. */
  
#define MU_LOCKER_DEFAULT  (MU_LOCKER_DOTLOCK | MU_LOCKER_RETRY)

/* Use these flags for as the default locker flags (the default defaults
 * to MU_LOCKER_DEFAULT). A flags of 0 resets the flags back to the
 * the default.
 */
extern int mu_locker_set_default_flags (int flags, enum mu_locker_set_mode mode);
extern void mu_locker_set_default_retry_timeout (time_t to);
extern void mu_locker_set_default_retry_count (size_t n);
extern void mu_locker_set_default_expire_timeout (time_t t);
extern void mu_locker_set_default_external_program (char *path);

/* A flags of 0 means that the default will be used. */
extern int mu_locker_create (mu_locker_t *, const char *filename, int flags);
extern void mu_locker_destroy (mu_locker_t *);

/* Time is measured in seconds. */

extern int mu_locker_set_flags (mu_locker_t, int);
extern int mu_locker_mod_flags (mu_locker_t locker, int flags,
				enum mu_locker_set_mode mode);
extern int mu_locker_set_expire_time (mu_locker_t, int);
extern int mu_locker_set_retries (mu_locker_t, int);
extern int mu_locker_set_retry_sleep (mu_locker_t, int);
extern int mu_locker_set_external (mu_locker_t, const char* program);

extern int mu_locker_get_flags (mu_locker_t, int*);
extern int mu_locker_get_expire_time (mu_locker_t, int*);
extern int mu_locker_get_retries (mu_locker_t, int*);
extern int mu_locker_get_retry_sleep (mu_locker_t, int*);
extern int mu_locker_get_external (mu_locker_t, char**);

enum mu_locker_mode
{
   mu_lck_shr,   /* Shared (advisory) lock */
   mu_lck_exc,   /* Exclusive lock */
   mu_lck_opt    /* Optional lock = shared, if the locker supports it, no
                    locking otherwise */
}; 

extern int mu_locker_lock          (mu_locker_t);
extern int mu_locker_touchlock     (mu_locker_t);
extern int mu_locker_unlock        (mu_locker_t);
extern int mu_locker_remove_lock   (mu_locker_t);

#ifdef __cplusplus
}
#endif

#endif

