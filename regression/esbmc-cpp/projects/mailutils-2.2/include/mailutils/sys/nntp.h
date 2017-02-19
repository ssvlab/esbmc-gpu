/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2007, 2010 Free Software Foundation, Inc.

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

#ifndef _MAILUTILS_SYS_NNTP_H
#define _MAILUTILS_SYS_NNTP_H

#include <sys/types.h>
#include <mailutils/nntp.h>
#include <mailutils/errno.h>

#ifdef DMALLOC
# include <dmalloc.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Response codes. */

#define MU_NNTP_RESP_CODE_HELP_FOLLOW           100
#define MU_NNTP_RESP_CODE_SERVER_DATE           111

#define MU_NNTP_RESP_CODE_POSTING_ALLOWED       200
#define MU_NNTP_RESP_CODE_POSTING_PROHIBITED    201
#define MU_NNTP_RESP_CODE_EXTENSIONS_FOLLOW     202

#define MU_NNTP_RESP_CODE_CLOSING               205
#define MU_NNTP_RESP_CODE_GROUP_SELECTED        211

#define MU_NNTP_RESP_CODE_LIST_FOLLOW           215

#define MU_NNTP_RESP_CODE_ARTICLE_FOLLOW        220
#define MU_NNTP_RESP_CODE_HEAD_FOLLOW           221
#define MU_NNTP_RESP_CODE_BODY_FOLLOW           222
#define MU_NNTP_RESP_CODE_ARTICLE_FOUND         223

#define MU_NNTP_RESP_CODE_NEWNEWS_FOLLOW        230
#define MU_NNTP_RESP_CODE_NEWGROUPS_FOLLOW      231
#define MU_NNTP_RESP_CODE_TRANSFER_OK           235

#define MU_NNTP_RESP_CODE_ARTICLE_RECEIVED      240

#define MU_NNTP_RESP_CODE_TRANSFER_ARTICLE      335
#define MU_NNTP_RESP_CODE_SEND_ARTICLE          340

#define MU_NNTP_RESP_CODE_TEMP_UNAVAILABLE      400
#define MU_NNTP_RESP_CODE_NO_EXTENSION          402
#define MU_NNTP_RESP_CODE_NO_ARTICLE_WITH_MID   430
#define MU_NNTP_RESP_CODE_NO_GROUP_SELECTED     412
#define MU_NNTP_RESP_CODE_NUMBER_INVALID        420
#define MU_NNTP_RESP_CODE_NO_ARTICLE            422
#define MU_NNTP_RESP_CODE_NO_ARTICLE_IN_RANGE   423
#define MU_NNTP_RESP_CODE_ARTICLE_NOT_WANTED    435
#define MU_NNTP_RESP_CODE_TRANSFER_NOT_POSSIBLE 436
#define MU_NNTP_RESP_CODE_TRANSFER_REJECTED     437
#define MU_NNTP_RESP_CODE_POSTING_NOT_PERMITTED 440
#define MU_NNTP_RESP_CODE_POSTING_FAILED        441
#define MU_NNTP_RESP_CODE_PERM_UNAVAILABLE      502

enum mu_nntp_state
  {
    MU_NNTP_NO_STATE,
    MU_NNTP_CONNECT,            MU_NNTP_GREETINGS,
    MU_NNTP_MODE_READER,        MU_NNTP_MODE_READER_ACK,
    MU_NNTP_LIST_EXTENSIONS,    MU_NNTP_LIST_EXTENSIONS_ACK,    MU_NNTP_LIST_EXTENSIONS_RX,
    MU_NNTP_LIST_ACTIVE,        MU_NNTP_LIST_ACTIVE_ACK,        MU_NNTP_LIST_ACTIVE_RX,
    MU_NNTP_LIST_ACTIVE_TIMES,  MU_NNTP_LIST_ACTIVE_TIMES_ACK,  MU_NNTP_LIST_ACTIVE_TIMES_RX,
    MU_NNTP_LIST_DISTRIBUTIONS, MU_NNTP_LIST_DISTRIBUTIONS_ACK, MU_NNTP_LIST_DISTRIBUTIONS_RX,
    MU_NNTP_LIST_DISTRIB_PATS,  MU_NNTP_LIST_DISTRIB_PATS_ACK,  MU_NNTP_LIST_DISTRIB_PATS_RX,
    MU_NNTP_LIST_NEWSGROUPS,    MU_NNTP_LIST_NEWSGROUPS_ACK,    MU_NNTP_LIST_NEWSGROUPS_RX,
    MU_NNTP_QUIT,               MU_NNTP_QUIT_ACK,
    MU_NNTP_GROUP,              MU_NNTP_GROUP_ACK,
    MU_NNTP_LAST,               MU_NNTP_LAST_ACK,
    MU_NNTP_NEXT,               MU_NNTP_NEXT_ACK,
    MU_NNTP_ARTICLE,            MU_NNTP_ARTICLE_ACK,            MU_NNTP_ARTICLE_RX,
    MU_NNTP_HEAD,               MU_NNTP_HEAD_ACK,               MU_NNTP_HEAD_RX,
    MU_NNTP_BODY,               MU_NNTP_BODY_ACK,               MU_NNTP_BODY_RX,
    MU_NNTP_STAT,               MU_NNTP_STAT_ACK,
    MU_NNTP_DATE,               MU_NNTP_DATE_ACK,
    MU_NNTP_HELP,               MU_NNTP_HELP_ACK,               MU_NNTP_HELP_RX,
    MU_NNTP_NEWGROUPS,          MU_NNTP_NEWGROUPS_ACK,          MU_NNTP_NEWGROUPS_RX,
    MU_NNTP_NEWNEWS,            MU_NNTP_NEWNEWS_ACK,            MU_NNTP_NEWNEWS_RX,
    MU_NNTP_POST,               MU_NNTP_POST_ACK,               MU_NNTP_POST_0, MU_NNTP_POST_1, MU_NNTP_POST_2, MU_NNTP_POST_3,
    MU_NNTP_IHAVE,              MU_NNTP_IHAVE_ACK,              MU_NNTP_IHAVE_0, MU_NNTP_IHAVE_1, MU_NNTP_IHAVE_2, MU_NNTP_IHAVE_3,
    MU_NNTP_DONE,               MU_NNTP_UNKNOWN,                MU_NNTP_ERROR
  };

/* Structure holding the data necessary to do proper buffering.  */
struct mu_nntp_work_buf
  {
    char *buf;
    char *ptr;
    char *nl;
    size_t len;
  };

/* Structure holding the data for post.  */
struct mu_nntp_post_buf
  {
    char *buf;
    int sent_crlf;
    size_t nread;
    long offset;
    size_t len;
  };

/* Structure to hold things general to nntp connection, like its state, etc ... */
struct _mu_nntp
  {
    /* Working I/O buffer.
       io.buf: Working io buffer
       io.ptr: Points to the end of the buffer, the non consumed chars
       io.nl: Points to the '\n' char in the string
       io.len: Len of io_buf.  */
    struct mu_nntp_work_buf io;

    /* Holds the first line response of the last command, i.e the ACK:
       ack.buf: Buffer for the ack
       ack.ptr: Working pointer, indicate the start of the non consumed chars
       ack.len: Size 512 according to RFC2449.  */
    struct mu_nntp_work_buf ack;
    int acknowledge;

    /* Holds the buffer/offset etc ... for the article to post.  */
    struct mu_nntp_post_buf post;

    unsigned timeout;  /* Default is 10 minutes.  */

    mu_debug_t debug; /* debugging trace.  */

    enum mu_nntp_state state;  /* Indicate the state of the running command.  */

    mu_stream_t carrier; /* TCP Connection.  */
  };

extern int  mu_nntp_debug_cmd        (mu_nntp_t);
extern int  mu_nntp_debug_ack        (mu_nntp_t);
extern int  mu_nntp_iterator_create  (mu_nntp_t, mu_iterator_t *iterator);
extern int  mu_nntp_stream_create    (mu_nntp_t nntp, mu_stream_t *pstream);
extern int  mu_nntp_carrier_is_ready (mu_stream_t carrier, int flag, int timeout);
extern int  mu_nntp_parse_article    (mu_nntp_t nntp, int code, unsigned long *pnum, char **mid);


/* Check for non recoverable error.
   The error is consider not recoverable if not part of the signal set:
   EAGAIN, EINPROGRESS, EINTR.
   For unrecoverable error we reset, by moving the working ptr
   to the begining of the buffer and setting the state to error.
 */
#define MU_NNTP_CHECK_EAGAIN(nntp, status) \
do \
  { \
    if (status != 0) \
      { \
         if (status != EAGAIN && status != EINPROGRESS && status != EINTR) \
           { \
             nntp->io.ptr = nntp->io.buf; \
             nntp->state = MU_NNTP_ERROR; \
           } \
         return status; \
      } \
   }  \
while (0)

/* If error return.
   Check status an reset(see MU_NNTP_CHECK_EAGAIN) the buffer.
  */
#define MU_NNTP_CHECK_ERROR(nntp, status) \
do \
  { \
     if (status != 0) \
       { \
          nntp->io.ptr = nntp->io.buf; \
          nntp->state = MU_NNTP_ERROR; \
          return status; \
       } \
  } \
while (0)

/* Check if we got the rigth. In NNTP protocol and ack of "2xx" means the command was completed.
 */
#define MU_NNTP_CHECK_CODE(nntp, code) \
do \
  { \
     if (mu_nntp_response_code (nntp) != code) \
       { \
          nntp->state = MU_NNTP_NO_STATE; \
          return EACCES; \
       } \
  } \
while (0)

#define MU_NNTP_CHECK_CODE2(nntp, code1, code2) \
do \
  { \
     if (mu_nntp_response_code (nntp) != code1 && mu_nntp_response_code (nntp) != code2) \
       { \
          nntp->state = MU_NNTP_NO_STATE; \
          return EACCES; \
       } \
  } \
while (0)

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_SYS_NNTP_H */
