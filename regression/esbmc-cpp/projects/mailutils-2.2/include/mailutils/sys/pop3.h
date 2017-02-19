/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2004, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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

#ifndef _MAILUTILS_SYS_POP3_H
#define _MAILUTILS_SYS_POP3_H

#include <sys/types.h>
#include <mailutils/pop3.h>
#include <mailutils/errno.h>
#include <mailutils/cstr.h>

#ifdef DMALLOC
# include <dmalloc.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum mu_pop3_state
  {
    MU_POP3_NO_STATE,
    MU_POP3_CONNECT, MU_POP3_GREETINGS,
    MU_POP3_APOP,    MU_POP3_APOP_ACK,
    MU_POP3_AUTH,    MU_POP3_AUTH_ACK,
    MU_POP3_CAPA,    MU_POP3_CAPA_ACK, MU_POP3_CAPA_RX,
    MU_POP3_DELE,    MU_POP3_DELE_ACK,
    MU_POP3_LIST,    MU_POP3_LIST_ACK, MU_POP3_LIST_RX,
    MU_POP3_NOOP,    MU_POP3_NOOP_ACK,
    MU_POP3_PASS,    MU_POP3_PASS_ACK,
    MU_POP3_QUIT,    MU_POP3_QUIT_ACK,
    MU_POP3_RETR,    MU_POP3_RETR_ACK, MU_POP3_RETR_RX,
    MU_POP3_RSET,    MU_POP3_RSET_ACK,
    MU_POP3_STAT,    MU_POP3_STAT_ACK,
    MU_POP3_STLS,    MU_POP3_STLS_ACK, MU_POP3_STLS_CONNECT,
    MU_POP3_TOP,     MU_POP3_TOP_ACK,  MU_POP3_TOP_RX,
    MU_POP3_UIDL,    MU_POP3_UIDL_ACK, MU_POP3_UIDL_RX,
    MU_POP3_USER,    MU_POP3_USER_ACK,
    MU_POP3_DONE,    MU_POP3_UNKNOWN,  MU_POP3_ERROR
  };

/* Structure holding the data necessary to do proper buffering.  */
struct mu_pop3_work_buf
  {
    char *buf;
    char *ptr;
    char *nl;
    size_t len;
  };

/* Structure to hold things general to POP3 mailbox, like its state, etc ... */
struct _mu_pop3
  {
    /* Working I/O buffer.
       io.buf: Working io buffer
       io.ptr: Points to the end of the buffer, the non consumed chars
       io.nl: Points to the '\n' char in the string
       io.len: Len of io_buf.  */
    struct mu_pop3_work_buf io;

    /* Holds the first line response of the last command, i.e the ACK:
       ack.buf: Buffer for the ack
       ack.ptr: Working pointer, indicate the start of the non consumed chars
       ack.len: Size 512 according to RFC2449.  */
    struct mu_pop3_work_buf ack;
    int acknowledge;

    char *timestamp; /* For apop, if supported.  */
    unsigned timeout;  /* Default is 10 minutes.  */

    mu_debug_t debug; /* debugging trace.  */

    enum mu_pop3_state state;  /* Indicate the state of the running command.  */

    mu_stream_t carrier; /* TCP Connection.  */
  };

extern int  mu_pop3_debug_cmd       (mu_pop3_t);
extern int  mu_pop3_debug_ack       (mu_pop3_t);
extern int  mu_pop3_iterator_create (mu_pop3_t pop3, mu_iterator_t *piterator);
extern int  mu_pop3_stream_create (mu_pop3_t pop3, mu_stream_t *pstream);
extern int  mu_pop3_carrier_is_ready (mu_stream_t carrier, int flag, int timeout);

/* Check for non recoverable error.
   The error is consider not recoverable if not part of the signal set:
   EAGAIN, EINPROGRESS, EINTR.
   For unrecoverable error we reset, by moving the working ptr
   to the begining of the buffer and setting the state to error.
 */
#define MU_POP3_CHECK_EAGAIN(pop3, status) \
do \
  { \
    if (status != 0) \
      { \
         if (status != EAGAIN && status != EINPROGRESS && status != EINTR) \
           { \
             pop3->io.ptr = pop3->io.buf; \
             pop3->state = MU_POP3_ERROR; \
           } \
         return status; \
      } \
   }  \
while (0)

/* If error return.
   Check status an reset(see MU_POP2_CHECK_EAGAIN) the buffer.
  */
#define MU_POP3_CHECK_ERROR(pop3, status) \
do \
  { \
     if (status != 0) \
       { \
          pop3->io.ptr = pop3->io.buf; \
          pop3->state = MU_POP3_ERROR; \
          return status; \
       } \
  } \
while (0)

/* Check if we got "+OK".
   In POP3 protocol and ack of "+OK" means the command was successfull.
 */
#define MU_POP3_CHECK_OK(pop3) \
do \
  { \
     if (mu_c_strncasecmp (pop3->ack.buf, "+OK", 3) != 0) \
       { \
          pop3->state = MU_POP3_NO_STATE; \
          return EACCES; \
       } \
  } \
while (0)

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_SYS_POP3_H */
