/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2004, 2005, 2006, 2007, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#ifdef HAVE_SHADOW_H
# include <shadow.h>
#endif
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#ifdef HAVE_SECURITY_PAM_APPL_H
# include <security/pam_appl.h>
#endif
#ifdef HAVE_CRYPT_H
# include <crypt.h>
#endif

#include <mailutils/argcv.h>
#include <mailutils/list.h>
#include <mailutils/iterator.h>
#include <mailutils/mailbox.h>
#include <mailutils/mu_auth.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/nls.h>
#include <mailutils/debug.h>

static mu_debug_t mu_auth_debug;

#define S(s) ((s) ? (s) : "(none)")
#define DEBUG_AUTH(a)                                                         \
     MU_DEBUG11 (mu_auth_debug, MU_DEBUG_TRACE,                               \
                       "source=%s, name=%s, passwd=%s, uid=%lu, gid=%lu, "    \
                       "gecos=%s, dir=%s, shell=%s, mailbox=%s, quota=%lu, "  \
                       "change_uid=%d\n",                                     \
                       S ((a)->source),                                       \
                       S ((a)->name),                                         \
                       S ((a)->passwd),                                       \
                       (unsigned long) (a)->uid,                              \
                       (unsigned long) (a)->gid,                              \
                       S ((a)->gecos),                                        \
                       S ((a)->dir),                                          \
                       S ((a)->shell),                                        \
                       S ((a)->mailbox),                                      \
                       (unsigned long) (a)->quota,                            \
                       (a)->change_uid)
     
mu_debug_t
mu_auth_set_debug (mu_debug_t debug)
{
  mu_debug_t prev = mu_auth_debug;
  mu_auth_debug = debug;
  return prev;
}


/* memory allocation */
int 
mu_auth_data_alloc (struct mu_auth_data **ptr,
                    const char *name,
                    const char *passwd,
                    uid_t uid,
                    gid_t gid,
                    const char *gecos,
                    const char *dir,
                    const char *shell,
                    const char *mailbox,
                    int change_uid)
{
  size_t size;
  char *p;
  char *tmp_mailbox_name = NULL;
  
  if (!name)
    return EINVAL;
  if (!passwd)
    passwd = "x";
  if (!gecos)
    gecos = "";
  if (!dir)
    dir = "/nonexisting";
  if (!shell)
    shell = "/dev/null";
  if (!mailbox)
    {
      int rc = mu_construct_user_mailbox_url (&tmp_mailbox_name, name);
      if (rc)
	return rc;
      mailbox = tmp_mailbox_name;
    }

  size = sizeof (**ptr) +
         strlen (name) + 1 +
         strlen (passwd) + 1 +
         strlen (gecos) + 1 +
         strlen (dir) + 1 +
         strlen (shell) + 1 +
         strlen (mailbox) + 1;
  
  *ptr = calloc (1, size);
  if (!*ptr)
    return ENOMEM;

  p = (char *)(*ptr + 1);
  
#define COPY(f) \
  (*ptr)->f = p; \
  strcpy (p, f); \
  p += strlen (f) + 1
  
  COPY (name);
  COPY (passwd);
  COPY (gecos);
  COPY (dir);
  COPY (shell);
  COPY (mailbox);
  (*ptr)->uid = uid;
  (*ptr)->gid = gid;
  (*ptr)->change_uid = change_uid;

  free (tmp_mailbox_name);
  return 0;
}

void
mu_auth_data_set_quota (struct mu_auth_data *ptr, mu_off_t q)
{
  ptr->flags |= MU_AF_QUOTA;
  ptr->quota = q;
}

void
mu_auth_data_free (struct mu_auth_data *ptr)
{
  free (ptr);
}

void
mu_auth_data_destroy (struct mu_auth_data **pptr)
{
  if (pptr)
    {
      mu_auth_data_free (*pptr);
      *pptr = NULL;
    }
}

/* Generic functions */

struct auth_stack_entry {
  char *name;        /* for diagnostics purposes */
  mu_auth_fp fun;
  void *func_data;
};

void
mu_insert_stack_entry (mu_list_t *pflist, struct auth_stack_entry *entry)
{
  if (!*pflist && mu_list_create (pflist))
    return;
  mu_list_append (*pflist, entry);
}

int
mu_auth_runlist (mu_list_t flist, struct mu_auth_data **return_data,
                 const void *key, void *data)
{
  int status = MU_ERR_AUTH_FAILURE;
  int rc;
  mu_iterator_t itr;

  if (mu_list_get_iterator (flist, &itr) == 0)
    {
      struct auth_stack_entry *ep;
      
      for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
           mu_iterator_next (itr))
        {
          mu_iterator_current (itr, (void **)&ep);
          MU_DEBUG1 (mu_auth_debug, MU_DEBUG_TRACE, "Trying %s...", ep->name);
          rc = ep->fun (return_data, key, ep->func_data, data);
          MU_DEBUG2 (mu_auth_debug, MU_DEBUG_TRACE, 
                     "result: %d=%s\n", rc, mu_strerror (rc));
          if (rc == 0)
            {
              if (return_data)
                {
                  struct mu_auth_data *auth = *return_data;
                  if (auth->source == NULL)
                    auth->source = ep->name;
                  DEBUG_AUTH (auth);
                }
              status = rc;
              break;
            }
          else if (rc == ENOSYS && status != 0)
	    /* nothing: do not overwrite last meaningful return code */;
	  else if (status != EAGAIN)
            status = rc;
        }

      mu_iterator_destroy (&itr);
    }
  return status;
}

int
mu_auth_nosupport (struct mu_auth_data **return_data MU_ARG_UNUSED,
                   const void *key MU_ARG_UNUSED,
                   void *func_data MU_ARG_UNUSED,
                   void *call_data MU_ARG_UNUSED)
{
  return ENOSYS;
}

/* II. Authorization: retrieving information about user */

static mu_list_t mu_auth_by_name_list, _tmp_auth_by_name_list;
static mu_list_t mu_auth_by_uid_list, _tmp_auth_by_uid_list;

int
mu_get_auth (struct mu_auth_data **auth, enum mu_auth_key_type type,
             const void *key)
{
  mu_list_t list;
  
  if (!mu_auth_by_name_list)
    mu_auth_begin_setup ();
  switch (type)
    {
    case mu_auth_key_name:
      MU_DEBUG1 (mu_auth_debug, MU_DEBUG_TRACE, 
                 "Getting auth info for user %s\n", 
                 (char*) key);
      list = mu_auth_by_name_list;
      break;

    case mu_auth_key_uid:
      MU_DEBUG1 (mu_auth_debug, MU_DEBUG_TRACE, 
                 "Getting auth info for UID %lu\n",
	         (unsigned long) *(uid_t*) key);
      list = mu_auth_by_uid_list;
      break;

    default:
      MU_DEBUG1 (mu_auth_debug, MU_DEBUG_ERROR, 
                 "Unknown mu_auth_key_type: %d\n", type);
      return EINVAL;
    }
  return mu_auth_runlist (list, auth, key, NULL);
}

struct mu_auth_data *
mu_get_auth_by_name (const char *username)
{
  struct mu_auth_data *auth = NULL;
  mu_get_auth (&auth, mu_auth_key_name, username);
  return auth;
}

struct mu_auth_data *
mu_get_auth_by_uid (uid_t uid)
{
  struct mu_auth_data *auth = NULL;
  mu_get_auth (&auth, mu_auth_key_uid, &uid);
  return auth;
}

/* III. Authentication: determining the authenticity of a user */

static mu_list_t mu_authenticate_list, _tmp_authenticate_list;

int
mu_authenticate (struct mu_auth_data *auth_data, const char *pass)
{
  if (!auth_data)
    return EINVAL;
  MU_DEBUG2 (mu_auth_debug, MU_DEBUG_TRACE, 
             "mu_authenticate, user %s, source %s\n", 
             auth_data->name, auth_data->source);
  if (!mu_authenticate_list)
    mu_auth_begin_setup ();
  return mu_auth_runlist (mu_authenticate_list, NULL, auth_data, (void*) pass);
}


/* ************************************************************************* */

struct _module_handler {
  struct auth_stack_entry authenticate;
  struct auth_stack_entry auth_by_name;
  struct auth_stack_entry auth_by_uid;
};

static mu_list_t module_handler_list;

void
mu_auth_register_module (struct mu_auth_module *mod)
{
  struct _module_handler *entry;
  
  if (mod->init)
    mu_gocs_register (mod->name, mod->init);
  
  if (!module_handler_list && mu_list_create (&module_handler_list))
    abort ();

  entry = malloc (sizeof (*entry));
  if (!entry)
    {
      mu_error ("not enough memory");
      exit (1);
    }
  entry->authenticate.name = mod->name;
  entry->authenticate.fun  = mod->authenticate;
  entry->authenticate.func_data = mod->authenticate_data;
  entry->auth_by_name.name = mod->name;
  entry->auth_by_name.fun = mod->auth_by_name;
  entry->auth_by_name.func_data = mod->auth_by_name_data;
  entry->auth_by_uid.name = mod->name;
  entry->auth_by_uid.fun = mod->auth_by_uid;
  entry->auth_by_uid.func_data = mod->auth_by_uid_data;
    
  mu_list_append (module_handler_list, entry);
  
}

static struct _module_handler *
_locate (const char *name)
{
  struct _module_handler *rp = NULL;
  mu_iterator_t itr;

  if (mu_list_get_iterator (module_handler_list, &itr) == 0)
    {
      struct _module_handler *p;
      
      for (mu_iterator_first (itr); !rp && !mu_iterator_is_done (itr);
           mu_iterator_next (itr))
        {
          mu_iterator_current (itr, (void **)&p);
          if (strcmp (p->authenticate.name, name) == 0)
            rp = p;
        }

      mu_iterator_destroy (&itr);
    }
  return rp;
}

static void
_add_module_list (const char *modlist, int (*fun)(const char *name))
{
  int argc;
  char **argv;
  int rc, i;
  
  rc = mu_argcv_get (modlist, ":", NULL, &argc, &argv);
  if (rc)
    {
      mu_error (_("cannot split line `%s': %s"), modlist, mu_strerror (rc));
      exit (1);
    }

  for (i = 0; i < argc; i += 2)
    {
      if (fun (argv[i]))
        {
          /* Historically,auth functions used ENOENT. We support this
             return value for backward compatibility. */
          if (errno == ENOENT || errno == MU_ERR_NOENT)
            mu_error (_("no such module: %s"), argv[i]);
          else
            mu_error (_("failed to add module %s: %s"),
                      argv[i], strerror (errno));
          exit (1);
        }
    }
  mu_argcv_free (argc, argv);
}


int
mu_authorization_add_module (const char *name)
{
  struct _module_handler *mod = _locate (name);
  
  if (!mod)
    {
      errno = MU_ERR_NOENT;
      return 1;
    }
  mu_insert_stack_entry (&_tmp_auth_by_name_list, &mod->auth_by_name);
  mu_insert_stack_entry (&_tmp_auth_by_uid_list, &mod->auth_by_uid);
  return 0;
}

void
mu_authorization_add_module_list (const char *modlist)
{
  _add_module_list (modlist, mu_authorization_add_module);
}

void
mu_authorization_clear_list ()
{
  mu_list_destroy (&_tmp_auth_by_name_list);
  mu_list_destroy (&_tmp_auth_by_uid_list);
}


int
mu_authentication_add_module (const char *name)
{
  struct _module_handler *mod = _locate (name);

  if (!mod)
    {
      errno = MU_ERR_NOENT;
      return 1;
    }
  mu_insert_stack_entry (&_tmp_authenticate_list, &mod->authenticate);
  return 0;
}

void
mu_authentication_add_module_list (const char *modlist)
{
  _add_module_list (modlist, mu_authentication_add_module);
}

void
mu_authentication_clear_list ()
{
  mu_list_destroy (&_tmp_authenticate_list);
}


/* ************************************************************************ */

/* Setup functions. Note that:
   1) If libmuath is not linked in, then "generic" and "system" modules
      are used unconditionally. This provides compatibility with the
      standard getpw.* functions.
   2) --authentication and --authorization modify only temporary lists,
      which get flushed to the main ones when mu_auth_finish_setup() is
      run. Thus, the default "generic:system" remain in force until
      argp_parse() exits. */
   
void
mu_auth_begin_setup ()
{
  mu_iterator_t itr;

  if (!module_handler_list)
    {
      mu_auth_register_module (&mu_auth_generic_module); 
      mu_auth_register_module (&mu_auth_system_module);
    }
  
  if (!mu_authenticate_list)
    {
      if (mu_list_get_iterator (module_handler_list, &itr) == 0)
        {
          struct _module_handler *mod;
          
          for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
               mu_iterator_next (itr))
            {
              mu_iterator_current (itr, (void **)&mod);
              mu_insert_stack_entry (&mu_authenticate_list,
                                     &mod->authenticate);
            }
          mu_iterator_destroy (&itr);
        }
    }

  if (!mu_auth_by_name_list)
    {
      if (mu_list_get_iterator (module_handler_list, &itr) == 0)
        {
          struct _module_handler *mod;
          
          for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
               mu_iterator_next (itr))
            {
              mu_iterator_current (itr, (void **)&mod);
              mu_insert_stack_entry (&mu_auth_by_name_list, &mod->auth_by_name);
              mu_insert_stack_entry (&mu_auth_by_uid_list, &mod->auth_by_uid);
            }
          mu_iterator_destroy (&itr);
        }
    }
}

void
mu_auth_finish_setup ()
{
  mu_list_destroy (&mu_authenticate_list);
  mu_list_destroy (&mu_auth_by_name_list);
  mu_list_destroy (&mu_auth_by_uid_list);
  
  mu_authenticate_list = _tmp_authenticate_list;
  _tmp_authenticate_list = NULL;
  mu_auth_by_name_list = _tmp_auth_by_name_list;
  _tmp_auth_by_name_list = NULL;
  mu_auth_by_uid_list = _tmp_auth_by_uid_list;
  _tmp_auth_by_uid_list = NULL;
  
  mu_auth_begin_setup ();
}


