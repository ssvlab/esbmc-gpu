/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2010 Free Software Foundation, Inc.

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

#ifndef _MAILUTILS_NNTP_H
#define _MAILUTILS_NNTP_H

#include <mailutils/debug.h>
#include <mailutils/stream.h>

#ifdef __cplusplus
extern "C" {
#endif

struct _mu_nntp;
typedef struct _mu_nntp* mu_nntp_t;

#define MU_NNTP_DEFAULT_PORT  119
#define MU_NNTP_URL_SCHEME    "nntp"

extern int  mu_nntp_create          (mu_nntp_t *nntp);
extern void mu_nntp_destroy         (mu_nntp_t *nntp);

extern int  mu_nntp_set_carrier     (mu_nntp_t nntp, mu_stream_t carrier);
extern int  mu_nntp_get_carrier     (mu_nntp_t nntp, mu_stream_t *pcarrier);

extern int  mu_nntp_connect         (mu_nntp_t nntp);
extern int  mu_nntp_disconnect      (mu_nntp_t nntp);

extern int  mu_nntp_set_timeout     (mu_nntp_t nntp, int timeout);
extern int  mu_nntp_get_timeout     (mu_nntp_t nntp, int *timeout);

extern int  mu_nntp_set_debug       (mu_nntp_t nntp, mu_debug_t debug);

extern int  mu_nntp_stls            (mu_nntp_t nntp);


extern int  mu_nntp_mode_reader     (mu_nntp_t nntp);

/* An iterator is return with the multi-line answer. It is the responsibility
   of the caller to call mu_iterator_destroy() to dispose of the iterator.  */
extern int  mu_nntp_list_extensions (mu_nntp_t nntp, mu_iterator_t *iterator);

extern int  mu_nntp_quit            (mu_nntp_t nntp);

/* The argument name is allocated with malloc(3).  The caller is responsible
   to call free(3)  */
extern int  mu_nntp_group           (mu_nntp_t nntp,
				     const char *group,
				     unsigned long *total,
				     unsigned long *first,
				     unsigned long *last, char **name);

/* The argument mid is allocated with malloc(3). The caller is responsible
   to call free(3)  */
extern int  mu_nntp_last            (mu_nntp_t nntp,
				     unsigned long *number, char **mid);
extern int  mu_nntp_next            (mu_nntp_t nntp,
				     unsigned long *number, char **mid);

/* The argument mid is allocated with malloc(3). The caller is responsible
   to call free(3). The caller must call stream_destoy() when done, no
   other commands are permitted until the stream is destroyed.  */
extern int  mu_nntp_article         (mu_nntp_t nntp, unsigned long number,
				     unsigned long *pnum, char **mid,
				     mu_stream_t *stream);
extern int  mu_nntp_article_id      (mu_nntp_t nntp, const char *id,
				     unsigned long *pnum, char **mid,
				     mu_stream_t *stream);
extern int  mu_nntp_head            (mu_nntp_t nntp, unsigned long number,
				     unsigned long *pnum, char **mid,
				     mu_stream_t *stream);
extern int  mu_nntp_head_id         (mu_nntp_t nntp, const char *name,
				     unsigned long *pnum, char **mid,
				     mu_stream_t *stream);
extern int  mu_nntp_body            (mu_nntp_t nntp, unsigned long number,
				     unsigned long *pnum, char **mid,
				     mu_stream_t *stream);
extern int  mu_nntp_body_id         (mu_nntp_t nntp, const char *id,
				     unsigned long *pnum, char **mid,
				     mu_stream_t *stream);

/* The argument mid is allocated with malloc(3). The caller is responsible
   to call free(3)  */
extern int  mu_nntp_stat            (mu_nntp_t nntp, unsigned long number,
				     unsigned long *pnum, char **mid);
extern int  mu_nntp_stat_id         (mu_nntp_t nntp, const char *id,
				     unsigned long *pnum, char **mid);

extern int  mu_nntp_date            (mu_nntp_t nntp, unsigned int *year,
				     unsigned int *month, unsigned int *day,
				     unsigned int *hour, unsigned int *minute,
				     unsigned int *second);

/* The caller must call stream_destoy() when done, no other commands are
   permitted until the stream is destroyed.  */
extern int  mu_nntp_help            (mu_nntp_t nntp, mu_stream_t *stream);


/* An iterator is return with the multi-line answer. It is the responsibility
   of the caller to call mu_iterator_destroy() to dispose of the iterator.  */
extern int  mu_nntp_newgroups       (mu_nntp_t nntp, unsigned int year,
				     unsigned int month, unsigned int day,
				     unsigned int hour, unsigned int minute,
				     unsigned int second, int is_gmt,
				     mu_iterator_t *iterator);
  
/* A iterator is return with the multi-line answer. It is the responsibility
   of the caller to call mu_iterator_destroy() to dispose of the iterator.  */
extern int  mu_nntp_newnews       (mu_nntp_t nntp, const char *wildmat,
				   unsigned int year, unsigned int month,
				   unsigned int day, unsigned int hour,
				   unsigned int minute, unsigned int second,
				   int is_gmt, mu_iterator_t *iterator);

extern int  mu_nntp_post            (mu_nntp_t nntp, mu_stream_t stream);
extern int  mu_nntp_ihave           (mu_nntp_t nntp, const char *mid,
				     mu_stream_t stream);


/* A iterator is return with the multi-line answer. It is the responsibility
   of the caller to call mu_iterator_destroy() to dispose of the iterator.  */
extern int  mu_nntp_list_active     (mu_nntp_t nntp, const char *wildmat,
				     mu_iterator_t *iterator);
extern int  mu_nntp_list_active_times      (mu_nntp_t nntp,
					    const char *wildmat,
					    mu_iterator_t *iterator);
extern int  mu_nntp_list_distributions (mu_nntp_t nntp,
					const char *wildmat,
					mu_iterator_t *iterator);
extern int  mu_nntp_list_distrib_pats  (mu_nntp_t nntp, mu_iterator_t *iterator);
extern int  mu_nntp_list_newsgroups    (mu_nntp_t nntp, 
					const char *wildmat,
					mu_iterator_t *iterator);


/* Parse the list active response.
   "group high low status"
   group: is the name of the group
   high:  high wather mark
   low: low water mark
   status: current status
      'y': posting is permitted
      'm': posting is not permitted
      'm': postings will be moderated

 The argument group is allocated with malloc(3). The caller is responsible
 to call free(3).
*/
  
extern int mu_nntp_parse_list_active (const char *buffer, char **group,
				      unsigned long *high, unsigned long *low,
				      char *status);
extern int mu_nntp_parse_newgroups   (const char *buffer, char **group,
				      unsigned long *high, unsigned long *low,
				      char *status);
  
/* Parse the list active.times response.
   "group time creator"
   group: is the name of the group
   time: measure in seconds since Jan 1 1970
   creator: entity taht created the newsgroup

 The argument group/creator is allocated with malloc(3). The caller is
 responsible to call free(3).
*/
  
extern int mu_nntp_parse_list_active_times  (const char *buffer, char **group,
					     unsigned long *time,
					     char **creator);
/* Parse the list distributions response.
   "key value"
   key: field key.
   value: short explaination of key

 The argument key/value is allocated with malloc(3). The caller is responsible
 to call free(3).
*/
extern int mu_nntp_parse_list_distributions  (const char *buffer,
					      char **key, char **value);
  
/* Parse the list distributions response.
   "weight:wildmat:distrib"
   weight:
   wildmat:
   distrib:

 The argument wildmat/distrib is allocated with malloc(3). The caller is
 responsible to call free(3).
*/
extern int mu_nntp_parse_list_distrib_pats  (const char *buffer,
					     unsigned long *weight,
					     char **wildmat, char **distrib);
  
/* Parse the list distributions response.
   "group description"

 The argument group/description is allocated with malloc(3). The caller is
 responsible to call free(3).
*/
extern int mu_nntp_parse_list_newsgroups  (const char *buffer, char **group,
					   char **description);

/* Reads the multi-line response of the server, nread will be 0 when the
   termination octets are detected. Clients should not use this function
   unless they are sending a direct command.  */
extern int  mu_nntp_readline         (mu_nntp_t nntp, char *buffer,
				      size_t buflen, size_t *nread);

/* Returns the last command acknowledge. If the server supports RESP-CODE,
   the message could be retrieved, but it is up the caller to do the parsing.
*/
extern int  mu_nntp_response     (mu_nntp_t nntp, char *buffer,
				  size_t buflen, size_t *nread);

/* pop3_writeline copies the line in the internal buffer, a mu_pop3_send() is
   needed to do the actual transmission.  */
extern int  mu_nntp_writeline    (mu_nntp_t nntp, const char *format, ...)
                                  MU_PRINTFLIKE(2,3);

/* mu_pop3_sendline() is equivalent to:
       mu_pop3_writeline (pop3, line);
       mu_pop3_send (pop3);
 */
extern int  mu_nntp_sendline     (mu_nntp_t nntp, const char *line);

/* Transmit via the carrier the internal buffer data.  */
extern int  mu_nntp_send         (mu_nntp_t nntp);

extern int mu_nntp_response_code (mu_nntp_t nntp);
  
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_POP3_H */
