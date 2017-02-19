/* -*- buffer-read-only: t -*- vi: set ro:
   THIS FILE IS GENERATED AUTOMATICALLY.  PLEASE DO NOT EDIT.
*/
/* GNU Mailutils -- a suite of utilities for electronic mail -*- c -*-
   Copyright (C) 1999, 2000, 2005, 2007, 2008, 2010 Free Software
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

#ifndef _MAILUTILS_DEBUG_H
#define _MAILUTILS_DEBUG_H

#include <stdarg.h>

#include <mailutils/types.h>
#include <mailutils/error.h>

#define MU_DEBUG(d,l,s) MU_DEBUG1(d,l,"%s",s)

#ifdef __cplusplus
extern "C" {
#endif

#define MU_DEBUG_ERROR  0
#define MU_DEBUG_TRACE0 1
#define MU_DEBUG_TRACE MU_DEBUG_TRACE0
#define MU_DEBUG_TRACE1 2 
#define MU_DEBUG_TRACE2 3
#define MU_DEBUG_TRACE3 4
#define MU_DEBUG_TRACE4 5
#define MU_DEBUG_TRACE5 6
#define MU_DEBUG_TRACE6 7
#define MU_DEBUG_TRACE7 8 

#define MU_DEBUG_PROT   9

#define MU_DEBUG_LEVEL_MASK(lev) (1 << (lev))
#define MU_DEBUG_LEVEL_UPTO(lev) ((1 << ((lev)+1)) - 1)

#define MU_DEBUG_INHERIT    0xf0000
#define MU_DEBUG_EXTRACT_LEVEL(s) ((s) & ~MU_DEBUG_INHERIT)

struct mu_debug_locus
{
  const char *file;
  int line;
};
  
int mu_debug_create    (mu_debug_t *, void *);
void mu_debug_destroy  (mu_debug_t *, void *);
void *mu_debug_get_owner (mu_debug_t);
int mu_debug_set_level (mu_debug_t, mu_log_level_t);
int mu_debug_get_level (mu_debug_t, mu_log_level_t *);
int mu_debug_set_locus (mu_debug_t, const char *, int);
int mu_debug_get_locus (mu_debug_t, struct mu_debug_locus *);
int mu_debug_set_function (mu_debug_t, const char *);
int mu_debug_get_function (mu_debug_t, const char **);
  
int mu_debug_print     (mu_debug_t, mu_log_level_t, const char *, ...)
                         MU_PRINTFLIKE(3,4);
int mu_debug_printv    (mu_debug_t, mu_log_level_t, const char *, va_list);
int mu_debug_check_level (mu_debug_t, mu_log_level_t);

int mu_debug_printf (mu_debug_t, mu_log_level_t, const char *, ...)
                        MU_PRINTFLIKE(3,4);
  
int mu_debug_vprintf (mu_debug_t, mu_log_level_t, const char *, va_list);

extern int mu_debug_line_info;
  
typedef int (*mu_debug_printer_fp) (void*, mu_log_level_t, const char *);
  
int mu_debug_set_print (mu_debug_t, mu_debug_printer_fp, void *);
int mu_debug_set_data (mu_debug_t, void *, void (*) (void*), void *);
extern mu_debug_printer_fp mu_debug_default_printer;
  
int mu_debug_syslog_printer (void *, mu_log_level_t, const char *);
int mu_debug_stderr_printer (void *, mu_log_level_t, const char *);

mu_log_level_t mu_global_debug_level (const char *);
int mu_global_debug_set_level (const char *, mu_log_level_t);
int mu_global_debug_clear_level (const char *);
int mu_global_debug_from_string (const char *, const char *);
int mu_debug_level_from_string (const char *string, mu_log_level_t *plev,
				mu_debug_t debug);

struct sockaddr;
void mu_sockaddr_to_str (const struct sockaddr *sa, int salen,
			 char *bufptr, size_t buflen,
			 size_t *plen);
char *mu_sockaddr_to_astr (const struct sockaddr *sa, int salen);

  
  
#define MU_ASSERT(expr)						\
 do								\
  {								\
    int rc = expr;						\
    if (rc)							\
      {								\
	mu_error ("%s:%d: " #expr " failed: %s",                \
                  __FILE__, __LINE__, mu_strerror (rc));	\
	abort ();						\
      }								\
  }                                                             \
 while (0)

  

#define __MU_DEBUG1(dbg, lev, fmt, x1) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1); \
   } \
 while (0)

#define MU_DEBUG1(dbg, lev, fmt, x1) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG1(dbg, lev, fmt, x1); \
   } \
 while(0)

#define __MU_DEBUG2(dbg, lev, fmt, x1, x2) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1, x2); \
   } \
 while (0)

#define MU_DEBUG2(dbg, lev, fmt, x1, x2) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG2(dbg, lev, fmt, x1, x2); \
   } \
 while(0)

#define __MU_DEBUG3(dbg, lev, fmt, x1, x2, x3) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1, x2, x3); \
   } \
 while (0)

#define MU_DEBUG3(dbg, lev, fmt, x1, x2, x3) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG3(dbg, lev, fmt, x1, x2, x3); \
   } \
 while(0)

#define __MU_DEBUG4(dbg, lev, fmt, x1, x2, x3, x4) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1, x2, x3, x4); \
   } \
 while (0)

#define MU_DEBUG4(dbg, lev, fmt, x1, x2, x3, x4) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG4(dbg, lev, fmt, x1, x2, x3, x4); \
   } \
 while(0)

#define __MU_DEBUG5(dbg, lev, fmt, x1, x2, x3, x4, x5) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1, x2, x3, x4, x5); \
   } \
 while (0)

#define MU_DEBUG5(dbg, lev, fmt, x1, x2, x3, x4, x5) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG5(dbg, lev, fmt, x1, x2, x3, x4, x5); \
   } \
 while(0)

#define __MU_DEBUG6(dbg, lev, fmt, x1, x2, x3, x4, x5, x6) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1, x2, x3, x4, x5, x6); \
   } \
 while (0)

#define MU_DEBUG6(dbg, lev, fmt, x1, x2, x3, x4, x5, x6) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG6(dbg, lev, fmt, x1, x2, x3, x4, x5, x6); \
   } \
 while(0)

#define __MU_DEBUG7(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7); \
   } \
 while (0)

#define MU_DEBUG7(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG7(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7); \
   } \
 while(0)

#define __MU_DEBUG8(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8); \
   } \
 while (0)

#define MU_DEBUG8(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG8(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8); \
   } \
 while(0)

#define __MU_DEBUG9(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9); \
   } \
 while (0)

#define MU_DEBUG9(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG9(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9); \
   } \
 while(0)

#define __MU_DEBUG10(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10); \
   } \
 while (0)

#define MU_DEBUG10(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG10(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10); \
   } \
 while(0)

#define __MU_DEBUG11(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11) \
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11); \
   } \
 while (0)

#define MU_DEBUG11(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG11(dbg, lev, fmt, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11); \
   } \
 while(0)


#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_DEBUG_H */
