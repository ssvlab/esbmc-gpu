/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2009, 2010 Free Software
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

#ifndef _IMAP0_H
#define _IMAP0_H

#ifdef DMALLOC
#  include <dmalloc.h>
#endif

#include <folder0.h>
#include <mailbox0.h>
#include <registrar0.h>
#include <auth0.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CLEAR_STATE(f_imap) \
 f_imap->selected = NULL, f_imap->state = IMAP_NO_STATE

/* Clear the state and close the stream.  */
#define CHECK_ERROR_CLOSE(folder, f_imap, status) \
do \
  { \
     if (status != 0) \
       { \
          mu_stream_close (folder->stream); \
          CLEAR_STATE (f_imap); \
          return status; \
       } \
  } \
while (0)

/* Clear the state.  */
#define CHECK_ERROR(f_imap, status) \
do \
  { \
     if (status != 0) \
       { \
          CLEAR_STATE (f_imap); \
          return status; \
       } \
  } \
while (0)

/* Clear the state for non recoverable error.  */
#define CHECK_EAGAIN(f_imap, status) \
do \
  { \
    if (status != 0) \
      { \
         if (status != EAGAIN && status != EINPROGRESS && status != EINTR) \
           { \
             CLEAR_STATE (f_imap); \
           } \
         return status; \
      } \
   }  \
while (0)


struct _f_imap;
struct _m_imap;
struct _msg_imap;
typedef struct _f_imap *f_imap_t;
typedef struct _m_imap *m_imap_t;
typedef struct _msg_imap *msg_imap_t;

enum imap_state
{
  IMAP_NO_STATE=0,
  IMAP_AUTH, IMAP_AUTH_DONE,
  IMAP_APPEND, IMAP_APPEND_CONT, IMAP_APPEND_SEND, IMAP_APPEND_ACK,
  IMAP_BODY,
  IMAP_CLOSE, IMAP_CLOSE_ACK,
  IMAP_COPY, IMAP_COPY_ACK,
  IMAP_CREATE, IMAP_CREATE_ACK,
  IMAP_DELETE, IMAP_DELETE_ACK,
  IMAP_EXPUNGE, IMAP_EXPUNGE_ACK,
  IMAP_FETCH, IMAP_FETCH_ACK,
  IMAP_GREETINGS,
  IMAP_HEADER,
  IMAP_HEADER_FIELD,
  IMAP_LIST, IMAP_LIST_PARSE, IMAP_LIST_ACK,
  IMAP_LOGIN, IMAP_LOGIN_ACK,
  IMAP_LOGOUT, IMAP_LOGOUT_ACK,
  IMAP_LSUB, IMAP_LSUB_ACK,
  IMAP_MESSAGE,
  IMAP_NOOP, IMAP_NOOP_ACK,
  IMAP_OPEN_CONNECTION,
  IMAP_RENAME, IMAP_RENAME_ACK,
  IMAP_SCAN, IMAP_SCAN_ACK,
  IMAP_SELECT, IMAP_SELECT_ACK,
  IMAP_STORE, IMAP_STORE_ACK,
  IMAP_SUBSCRIBE, IMAP_SUBSCRIBE_ACK,
  IMAP_UNSUBSCRIBE, IMAP_UNSUBSCRIBE_ACK
};

enum imap_auth_state
{
  /* ANONYMOUS */
  IMAP_AUTH_ANON_REQ_WRITE,
  IMAP_AUTH_ANON_REQ_SEND,
  IMAP_AUTH_ANON_WAIT_CONT,
  IMAP_AUTH_ANON_MSG,
  IMAP_AUTH_ANON_MSG_SEND,
  IMAP_AUTH_ANON_WAIT_RESP
};

struct literal_string
{
  char *buffer;
  size_t buflen;
  size_t total;
  msg_imap_t msg_imap;
  enum imap_state type;
  size_t nleft;  /* nleft to read in the literal. */
};

struct _f_imap
{
  /* Back pointer.  */
  mu_folder_t folder;
  m_imap_t selected;

  enum imap_state state;
  int imaps; /* IMAPS or IMAP? */

  size_t seq; /* Sequence number to build a tag.  */
  char **capav; /* Cabilities of the server.  */
  int capac;    /* Number of capabilities in the above array */
  int flags;

  /* IO use to hold the literal and quoted strings send by
     the IMAP server.  */
  struct
  {
    mu_stream_t stream;
    mu_off_t offset;
    size_t nleft;  /* nleft to read in the literal. */
    msg_imap_t msg_imap;
    enum imap_state type;
  } string;

  /* Use for LIST and LSUB.  */
  mu_list_t flist;
  mu_folder_enumerate_fp enum_fun;
  void *enum_data;
  int enum_stop;
  
  int isopen;

  /* Server channel buffer I/O  */
  size_t buflen;
  char *buffer;
  char *ptr;
  char *nl;
  mu_off_t offset; /* Dummy, this is used because of the stream buffering.
                   The mu_stream_t maintains and offset and the offset we
		   use must be in sync.  */

  /* Login  */
  char *user;
  mu_secret_t secret;

  /* AUTHENTICATE states */
  enum imap_auth_state auth_state;
};

struct _m_imap
{
  /* Back pointers.  */
  mu_mailbox_t mailbox;
  f_imap_t f_imap;
  size_t messages_count;
  size_t imessages_count;
  msg_imap_t *imessages;
  size_t recent;
  size_t unseen;
  unsigned long uidvalidity;
  size_t uidnext;
  char *name;
  enum imap_state state;
    /* mailbox operations can be sequences of folder operations, and
       thus need to keep meta-state, mailbox_imap_open(), for example. */
};

struct _msg_imap
{
  /* Back pointers.  */
  mu_message_t message;
  m_imap_t m_imap;
  size_t num;
  size_t part;
  size_t num_parts;
  msg_imap_t *parts;
  msg_imap_t parent;
  int flags;
  size_t uid;

  mu_header_t fheader;
  char *internal_date;

  size_t mu_message_size;
  size_t mu_message_lines;
  size_t body_size;
  size_t body_lines;
  size_t header_size;
  size_t header_lines;
};

int imap_writeline    (f_imap_t,  const char *format, ...) MU_PRINTFLIKE(2,3);
int imap_write        (f_imap_t);
int imap_send         (f_imap_t);
int imap_parse        (f_imap_t);
int imap_readline     (f_imap_t);
char *section_name    (msg_imap_t);

#ifdef __cplusplus
}
#endif

#endif /* _IMAP0_H */
