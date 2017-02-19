%{
/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2007, 2008, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.
   
   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA  */
  
/* A heavily modified version of the well-known public domain getdate.y.
   It was originally written by Steven M. Bellovin <smb@research.att.com>
   while at the University of North Carolina at Chapel Hill.  Later tweaked
   by a couple of people on Usenet.  Completely overhauled by Rich $alz
   <rsalz@bbn.com> and Jim Berets <jberets@bbn.com> in August, 1990.
   Rewritten using a proper union by Sergey Poznyakoff <gray@gnu.org> */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <ctype.h>

#if HAVE_STDLIB_H
# include <stdlib.h> /* for `free'; used by Bison 1.27 */
#endif

#if defined (STDC_HEADERS) || (!defined (isascii) && !defined (HAVE_ISASCII))
# define IN_CTYPE_DOMAIN(c) 1
#else
# define IN_CTYPE_DOMAIN(c) isascii(c)
#endif

#define ISSPACE(c) (IN_CTYPE_DOMAIN (c) && isspace (c))
#define ISALPHA(c) (IN_CTYPE_DOMAIN (c) && isalpha (c))
#define ISUPPER(c) (IN_CTYPE_DOMAIN (c) && isupper (c))
#define ISDIGIT_LOCALE(c) (IN_CTYPE_DOMAIN (c) && isdigit (c))

/* ISDIGIT differs from ISDIGIT_LOCALE, as follows:
   - Its arg may be any int or unsigned int; it need not be an unsigned char.
   - It's guaranteed to evaluate its argument exactly once.
   - It's typically faster.
   Posix 1003.2-1992 section 2.5.2.1 page 50 lines 1556-1558 says that
   only '0' through '9' are digits.  Prefer ISDIGIT to ISDIGIT_LOCALE unless
   it's important to use the locale's definition of `digit' even when the
   host does not conform to Posix.  */
#define ISDIGIT(c) ((unsigned) (c) - '0' <= 9)

#if defined (STDC_HEADERS) || defined (USG)
# include <string.h>
#endif

/* Some old versions of bison generate parsers that use bcopy.
   That loses on systems that don't provide the function, so we have
   to redefine it here.  */
#if !defined (HAVE_BCOPY) && defined (HAVE_MEMCPY) && !defined (bcopy)
# define bcopy(from, to, len) memcpy ((to), (from), (len))
#endif

static int yylex ();
static int yyerror ();

#define EPOCH		1970
#define HOUR(x)		((x) * 60)

#define MAX_BUFF_LEN    128   /* size of buffer to read the date into */

/*
**  An entry in the lexical lookup table.
*/
typedef struct _lex_tab {
  const char	*name;
  int		type;
  int		value;
} SYMBOL;


/*
**  Meridian:  am, pm, or 24-hour style.
*/
typedef enum meridian {
  MERam,
  MERpm,
  MER24
} MERIDIAN;

#define PD_MASK_SECOND   00001
#define PD_MASK_MINUTE   00002
#define PD_MASK_HOUR     00004
#define PD_MASK_DAY      00010
#define PD_MASK_MONTH    00020
#define PD_MASK_YEAR     00040
#define PD_MASK_TZ       00100 
#define PD_MASK_MERIDIAN 00200
#define PD_MASK_ORDINAL  00400
#define PD_MASK_NUMBER   01000

#define PD_MASK_TIME PD_MASK_SECOND|PD_MASK_MINUTE|PD_MASK_HOUR
#define PD_MASK_DATE PD_MASK_DAY|PD_MASK_MONTH|PD_MASK_YEAR
#define PD_MASK_DOW PD_MASK_NUMBER
 
#define MASK_IS_SET(f,m) (((f)&(m))==(m))
#define MASK_TEST(f,m)   ((f)&(m)) 
struct pd_date
{
  int mask;
  int day;
  int hour;
  int minute;
  int month;
  int second;
  int year;
  int tz;
  enum meridian meridian;
  int number;
  int ordinal;
};

#define DATE_INIT(date) memset(&(date), 0, sizeof(date))
#define DATE_SET(date, memb, m, val, lim, onerror)                        \
 do                                                                       \
   {                                                                      \
     int __x = val;                                                       \
     if (((m) != PD_MASK_TZ && __x < 0) || (lim && __x > lim)) onerror;   \
     date . memb = __x; date.mask |= m;                                   \
   }                                                                      \
 while (0)
   
#define __SET_SECOND(d,v,a)   DATE_SET(d,second,PD_MASK_SECOND,v,59,a)
#define __SET_MINUTE(d,v,a)   DATE_SET(d,minute,PD_MASK_MINUTE,v,59,a)  
#define __SET_HOUR(d,v,a)     DATE_SET(d,hour,PD_MASK_HOUR,v,23,a)
#define __SET_DAY(d,v,a)      DATE_SET(d,day,PD_MASK_DAY,v,31,a)   
#define __SET_MONTH(d,v,a)    DATE_SET(d,month,PD_MASK_MONTH,v,12,a)
#define __SET_YEAR(d,v,a)     DATE_SET(d,year,PD_MASK_YEAR,v,0,a)  
#define __SET_TZ(d,v,a)       DATE_SET(d,tz,PD_MASK_TZ,v,0,a)
#define __SET_MERIDIAN(d,v,a) DATE_SET(d,meridian,PD_MASK_MERIDIAN,v,MER24,a)
#define __SET_ORDINAL(d,v,a)  DATE_SET(d,ordinal,PD_MASK_ORDINAL,v,0,a)
#define __SET_NUMBER(d,v,a)   DATE_SET(d,number,PD_MASK_NUMBER,v,0,a) 
 
#define SET_SECOND(d,v)   __SET_SECOND(d,v,YYERROR)   
#define SET_MINUTE(d,v)   __SET_MINUTE(d,v,YYERROR)   
#define SET_HOUR(d,v)     __SET_HOUR(d,v,YYERROR)     
#define SET_DAY(d,v)      __SET_DAY(d,v,YYERROR)      
#define SET_MONTH(d,v)    __SET_MONTH(d,v,YYERROR)    
#define SET_YEAR(d,v)     __SET_YEAR(d,v,YYERROR)     
#define SET_TZ(d,v)       __SET_TZ(d,v,YYERROR)       
#define SET_MERIDIAN(d,v) __SET_MERIDIAN(d,v,YYERROR) 
#define SET_ORDINAL(d,v)  __SET_ORDINAL(d,v,YYERROR)  
#define SET_NUMBER(d,v)   __SET_NUMBER(d,v,YYERROR)   

int
pd_date_union (struct pd_date *a, struct pd_date *b)
{
  int diff = (~a->mask) & b->mask;
  if (!diff)
    return 1;

  a->mask |= diff;
  
  if (diff & PD_MASK_SECOND)
    a->second = b->second;
  
  if (diff & PD_MASK_MINUTE)
    a->minute = b->minute;

  if (diff & PD_MASK_HOUR)
    a->hour = b->hour;

  if (diff & PD_MASK_DAY)
    a->day = b->day;

  if (diff & PD_MASK_MONTH)
    a->month = b->month;

  if (diff & PD_MASK_YEAR)
    a->year = b->year;

  if (diff & PD_MASK_TZ)
    a->tz = b->tz;

  if (diff & PD_MASK_MERIDIAN)
    a->meridian = b->meridian;

  if (diff & PD_MASK_ORDINAL)
    a->ordinal = b->ordinal;

  if (diff & PD_MASK_NUMBER)
    a->number = b->number;

  return 0;
}

struct pd_datespec
{
  struct pd_date date;
  struct pd_date rel;
};

static struct pd_datespec pd;
 
static const char	*yyinput;

%}

%union {
  int number;
  enum meridian meridian;
  struct pd_date date;
  struct pd_datespec datespec;
}

/*FIXME: do we need T_ID? */
%token  T_AGO T_DST  T_ID

%token <number> T_DAY T_DAY_UNIT T_DAYZONE T_HOUR_UNIT T_MINUTE_UNIT
%token <number> T_MONTH T_MONTH_UNIT
%token <number>	T_SEC_UNIT T_SNUMBER T_UNUMBER T_YEAR_UNIT T_ZONE
%token <meridian> T_MERIDIAN

%type <meridian> o_merid
%type <date> item time zone date day rel relspec relunit
%type <datespec> spec

%%

input   : spec
          {
	    pd = $1;
	  }
        ;

spec	: /* NULL */
          {
	    DATE_INIT ($$.date);
	    DATE_INIT ($$.rel);
	  }
	| spec item
          {
	    if (pd_date_union (&$1.date, &$2))
	      YYERROR;
	    $$ = $1;
	  }
        | spec rel
          {
	    if (pd_date_union (&$1.rel, &$2))
	      YYERROR;
	    $$ = $1;
	  }
        | spec T_UNUMBER
          {
	    if (MASK_IS_SET ($1.date.mask, (PD_MASK_TIME|PD_MASK_DATE))
		&& !$1.rel.mask)
	      SET_YEAR ($1.date, $2);
	    else
	      {
		if ($2 > 10000)
		  {
		    SET_DAY ($1.date, $2 % 100);
		    SET_MONTH ($1.date, ($2 / 100) %100);
		    SET_YEAR ($1.date, $2 / 10000);
		  }
		else
		  {
		    if ($2 < 100)
		      {
			SET_YEAR ($1.date, $2);
		      }
		    else
		      {
		    	SET_HOUR ($1.date, $2 / 100);
		    	SET_MINUTE ($1.date, $2 % 100);
		      }
		    SET_MERIDIAN ($1.date, MER24);
		  }
	      }
	    $$ = $1;
	  }
	;

item	: time
        | zone
	| date
	| day
	;

time	: T_UNUMBER T_MERIDIAN
          {
	    DATE_INIT ($$);
	    SET_HOUR ($$, $1);
	    SET_MERIDIAN ($$, $2);
	  }
	| T_UNUMBER ':' T_UNUMBER o_merid
          {
	    DATE_INIT ($$);
	    SET_HOUR ($$, $1);
	    SET_MINUTE ($$, $3);
	    SET_MERIDIAN ($$, $4);
	  }
	| T_UNUMBER ':' T_UNUMBER T_SNUMBER
          {
	    DATE_INIT ($$);
	    SET_HOUR ($$, $1);
	    SET_MINUTE ($$, $3);
	    SET_MERIDIAN ($$, MER24);
	    SET_TZ ($$, ($4 < 0
			   ? -$4 % 100 + (-$4 / 100) * 60
			   : - ($4 % 100 + ($4 / 100) * 60)));

	  }
	| T_UNUMBER ':' T_UNUMBER ':' T_UNUMBER o_merid
          {
	    DATE_INIT ($$);
	    SET_HOUR ($$, $1);
	    SET_MINUTE ($$, $3);
	    SET_SECOND ($$, $5);
	    SET_MERIDIAN ($$, $6);
	  }
	| T_UNUMBER ':' T_UNUMBER ':' T_UNUMBER T_SNUMBER
          {
	    DATE_INIT ($$);
	    SET_HOUR ($$, $1);
	    SET_MINUTE ($$, $3);
	    SET_SECOND ($$, $5);
	    SET_MERIDIAN ($$, MER24);
	    SET_TZ ($$, ($6 < 0
			 ? -$6 % 100 + (-$6 / 100) * 60
			 : - ($6 % 100 + ($6 / 100) * 60)));
	  }
	;

zone	: T_ZONE
          {
	    DATE_INIT ($$);
	    SET_TZ ($$, $1);
	  }
	| T_DAYZONE
          {
	    DATE_INIT ($$);
	    SET_TZ ($$, $1 - 60);
	  }
	| T_ZONE T_DST
          {
	    DATE_INIT ($$);
	    SET_TZ ($$, $1 - 60);
	  }
	;

day	: T_DAY
          {
	    DATE_INIT ($$);
	    SET_ORDINAL ($$, 1);
	    SET_NUMBER ($$, $1);
	  }
	| T_DAY ','
          {
	    DATE_INIT ($$);
	    SET_ORDINAL ($$, 1);
	    SET_NUMBER ($$, $1);
	  }
	| T_UNUMBER T_DAY
          {
	    DATE_INIT ($$);
	    SET_ORDINAL ($$, $1);
	    SET_NUMBER ($$, $2);
	  }
	;

date	: T_UNUMBER '/' T_UNUMBER
          {
	    DATE_INIT ($$);
	    SET_MONTH ($$, $1);
	    SET_DAY ($$, $3);
	  }
	| T_UNUMBER '/' T_UNUMBER '/' T_UNUMBER
          {
	    DATE_INIT ($$);
	    /* Interpret as YYYY/MM/DD if $1 >= 1000, otherwise as MM/DD/YY.
	       The goal in recognizing YYYY/MM/DD is solely to support legacy
	       machine-generated dates like those in an RCS log listing.  If
	       you want portability, use the ISO 8601 format.  */
	    if ($1 >= 1000)
	      {
		SET_YEAR ($$, $1);
		SET_MONTH ($$, $3);
		SET_DAY ($$, $5);
	      }
	    else
	      {
		SET_MONTH ($$, $1);
		SET_DAY ($$, $3);
		SET_YEAR ($$, $5);
	      }
	  }
	| T_UNUMBER T_SNUMBER T_SNUMBER
          {
	    DATE_INIT ($$);
	    /* ISO 8601 format.  yyyy-mm-dd.  */
	    SET_YEAR ($$, $1);
	    SET_MONTH ($$, -$2);
	    SET_DAY ($$, -$3);
	  }
	| T_UNUMBER T_MONTH T_SNUMBER
          {
	    /* either 17-JUN-1992 or 1992-JUN-17 */
	    DATE_INIT ($$);
	    if ($1 < 32)
	      {
		SET_DAY ($$, $1);
		SET_MONTH ($$, $2);
		SET_YEAR ($$, -$3);
	      }
	    else
	      {
		SET_DAY ($$, -$3);
		SET_MONTH ($$, $2);
		SET_YEAR ($$, $1);
	      }
	  }
	| T_MONTH T_UNUMBER
          {
	    DATE_INIT ($$);
	    SET_MONTH ($$, $1);
	    SET_DAY ($$, $2);
	  }
	| T_MONTH T_UNUMBER ',' T_UNUMBER
          {
	    DATE_INIT ($$);
	    SET_MONTH ($$, $1);
	    SET_DAY ($$, $2);
	    SET_YEAR ($$, $4);
	  }
	| T_UNUMBER T_MONTH
          {
	    DATE_INIT ($$);
	    SET_MONTH ($$, $2);
	    SET_DAY ($$, $1);
	  }
	| T_UNUMBER T_MONTH T_UNUMBER
          {
	    DATE_INIT ($$);
	    SET_MONTH ($$, $2);
	    SET_DAY ($$, $1);
	    SET_YEAR ($$, $3);
	  }
        | T_DAY T_MONTH T_UNUMBER time T_UNUMBER
          {
	    DATE_INIT ($$);

	    SET_ORDINAL ($$, 1);
	    SET_NUMBER ($$, $1);

	    SET_MONTH ($$, $2);
	    SET_DAY ($$, $3);
	    SET_YEAR ($$, $5);
	    if (pd_date_union (&$$, &$4))
	      YYERROR;
	  }
	;

rel	: relspec T_AGO
          {
	    $1.second = - $1.second;
	    $1.minute = - $1.minute;
	    $1.hour = - $1.hour;
	    $1.day = - $1.day;
	    $1.month = - $1.month;
	    $1.year = - $1.year;
	    $$ = $1;
	  }
	| relspec
	;


relspec : relunit
          {
	    DATE_INIT ($$);
	    if (pd_date_union (&$$, &$1))
	      YYERROR;
	  }
        | relspec relunit
          {
	    if (pd_date_union (&$1, &$2))
	      YYERROR;
	    $$ = $1;
	  }
        ;

relunit	: T_UNUMBER T_YEAR_UNIT
          {
	    DATE_INIT ($$);
	    SET_YEAR ($$, $1 * $2);
	  }
	| T_SNUMBER T_YEAR_UNIT
          {
	    DATE_INIT ($$);
	    SET_YEAR ($$, $1 * $2);
	  }
	| T_YEAR_UNIT
          {
	    DATE_INIT ($$);
	    SET_YEAR ($$, $1);
	  }
	| T_UNUMBER T_MONTH_UNIT
          {
	    DATE_INIT ($$);
	    SET_MONTH ($$, $1 * $2);
	  }
	| T_SNUMBER T_MONTH_UNIT
          {
	    DATE_INIT ($$);
	    SET_MONTH ($$, $1 * $2);
	  }
	| T_MONTH_UNIT
          {
	    DATE_INIT ($$);
	    SET_MONTH ($$, $1);
	  }
	| T_UNUMBER T_DAY_UNIT
          {
	    DATE_INIT ($$);
	    SET_DAY ($$, $1 * $2);
	  }
	| T_SNUMBER T_DAY_UNIT
          {
	    DATE_INIT ($$);
	    SET_DAY ($$, $1 * $2);
	  }
	| T_DAY_UNIT
          {
	    DATE_INIT ($$);
	    SET_DAY ($$, $1);
	  }
	| T_UNUMBER T_HOUR_UNIT
          {
	    DATE_INIT ($$);
	    SET_HOUR ($$, $1 * $2);
	  }
	| T_SNUMBER T_HOUR_UNIT
          {
	    DATE_INIT ($$);
	    SET_HOUR ($$, $1 * $2);
	  }
	| T_HOUR_UNIT
          {
	    DATE_INIT ($$);
	    SET_HOUR ($$, $1);
	  }
	| T_UNUMBER T_MINUTE_UNIT
          {
	    DATE_INIT ($$);
	    SET_MINUTE ($$, $1 * $2);
	  }
	| T_SNUMBER T_MINUTE_UNIT
          {
	    DATE_INIT ($$);
	    SET_MINUTE ($$, $1 * $2);
	  }
	| T_MINUTE_UNIT
          {
	    DATE_INIT ($$);
	    SET_MINUTE ($$, $1);
	  }
	| T_UNUMBER T_SEC_UNIT
          {
	    DATE_INIT ($$);
	    SET_SECOND ($$, $1 * $2);
	  }
	| T_SNUMBER T_SEC_UNIT
          {
	    DATE_INIT ($$);
	    SET_SECOND ($$, $1 * $2);
	  }
	| T_SEC_UNIT
          {
	    DATE_INIT ($$);
	    SET_SECOND ($$, $1);
	  }
	;

o_merid	: /* empty */
          {
	    $$ = MER24;
	  }
	| T_MERIDIAN
	  {
	    $$ = $1;
	  }
	;

%%

#include <mailutils/types.h>

#if TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif

#include <mailutils/mutil.h>

/* Month and day table. */
static SYMBOL const month_day_tab[] = {
  { "january",	T_MONTH,  1 },
  { "february",	T_MONTH,  2 },
  { "march",	T_MONTH,  3 },
  { "april",	T_MONTH,  4 },
  { "may",	T_MONTH,  5 },
  { "june",	T_MONTH,  6 },
  { "july",	T_MONTH,  7 },
  { "august",	T_MONTH,  8 },
  { "september",T_MONTH,  9 },
  { "sept",	T_MONTH,  9 },
  { "october",	T_MONTH, 10 },
  { "november",	T_MONTH, 11 },
  { "december",	T_MONTH, 12 },
  { "sunday",	T_DAY,   0 },
  { "monday",	T_DAY,   1 },
  { "tuesday",	T_DAY,   2 },
  { "tues",	T_DAY,   2 },
  { "wednesday",T_DAY,   3 },
  { "wednes",	T_DAY,   3 },
  { "thursday",	T_DAY,   4 },
  { "thur",	T_DAY,   4 },
  { "thurs",	T_DAY,   4 },
  { "friday",	T_DAY,   5 },
  { "saturday",	T_DAY,   6 },
  { NULL, 0, 0 }
};

/* Time units table. */
static SYMBOL const units_tab[] = {
  { "year",	T_YEAR_UNIT,	1 },
  { "month",	T_MONTH_UNIT,	1 },
  { "fortnight",T_DAY_UNIT,	14 },
  { "week",	T_DAY_UNIT,	7 },
  { "day",	T_DAY_UNIT,	1 },
  { "hour",	T_HOUR_UNIT,	1 },
  { "minute",	T_MINUTE_UNIT,	1 },
  { "min",	T_MINUTE_UNIT,	1 },
  { "second",	T_SEC_UNIT,	1 },
  { "sec",	T_SEC_UNIT,	1 },
  { NULL, 0, 0 }
};

/* Assorted relative-time words. */
static SYMBOL const other_tab[] = {
  { "tomorrow",	T_MINUTE_UNIT,	1 * 24 * 60 },
  { "yesterday",T_MINUTE_UNIT,	-1 * 24 * 60 },
  { "today",	T_MINUTE_UNIT,	0 },
  { "now",	T_MINUTE_UNIT,	0 },
  { "last",	T_UNUMBER,	-1 },
  { "this",	T_MINUTE_UNIT,	0 },
  { "next",	T_UNUMBER,	1 },
  { "first",	T_UNUMBER,	1 },
/*  { "second",	T_UNUMBER,	2 }, */
  { "third",	T_UNUMBER,	3 },
  { "fourth",	T_UNUMBER,	4 },
  { "fifth",	T_UNUMBER,	5 },
  { "sixth",	T_UNUMBER,	6 },
  { "seventh",	T_UNUMBER,	7 },
  { "eighth",	T_UNUMBER,	8 },
  { "ninth",	T_UNUMBER,	9 },
  { "tenth",	T_UNUMBER,	10 },
  { "eleventh",	T_UNUMBER,	11 },
  { "twelfth",	T_UNUMBER,	12 },
  { "ago",	T_AGO,	        1 },
  { NULL, 0, 0 }
};

/* The timezone table. */
static SYMBOL const tz_tab[] = {
  { "gmt",	T_ZONE,     HOUR ( 0) },	/* Greenwich Mean */
  { "ut",	T_ZONE,     HOUR ( 0) },	/* Universal (Coordinated) */
  { "utc",	T_ZONE,     HOUR ( 0) },
  { "wet",	T_ZONE,     HOUR ( 0) },	/* Western European */
  { "bst",	T_DAYZONE,  HOUR ( 0) },	/* British Summer */
  { "wat",	T_ZONE,     HOUR ( 1) },	/* West Africa */
  { "at",	T_ZONE,     HOUR ( 2) },	/* Azores */
#if	0
  /* For completeness.  BST is also British Summer, and GST is
   * also Guam Standard. */
  { "bst",	T_ZONE,     HOUR ( 3) },	/* Brazil Standard */
  { "gst",	T_ZONE,     HOUR ( 3) },	/* Greenland Standard */
#endif
#if 0
  { "nft",	T_ZONE,     HOUR (3.5) },	/* Newfoundland */
  { "nst",	T_ZONE,     HOUR (3.5) },	/* Newfoundland Standard */
  { "ndt",	T_DAYZONE,  HOUR (3.5) },	/* Newfoundland Daylight */
#endif
  { "ast",	T_ZONE,     HOUR ( 4) },	/* Atlantic Standard */
  { "adt",	T_DAYZONE,  HOUR ( 4) },	/* Atlantic Daylight */
  { "est",	T_ZONE,     HOUR ( 5) },	/* Eastern Standard */
  { "edt",	T_DAYZONE,  HOUR ( 5) },	/* Eastern Daylight */
  { "cst",	T_ZONE,     HOUR ( 6) },	/* Central Standard */
  { "cdt",	T_DAYZONE,  HOUR ( 6) },	/* Central Daylight */
  { "mst",	T_ZONE,     HOUR ( 7) },	/* Mountain Standard */
  { "mdt",	T_DAYZONE,  HOUR ( 7) },	/* Mountain Daylight */
  { "pst",	T_ZONE,     HOUR ( 8) },	/* Pacific Standard */
  { "pdt",	T_DAYZONE,  HOUR ( 8) },	/* Pacific Daylight */
  { "yst",	T_ZONE,     HOUR ( 9) },	/* Yukon Standard */
  { "ydt",	T_DAYZONE,  HOUR ( 9) },	/* Yukon Daylight */
  { "hst",	T_ZONE,     HOUR (10) },	/* Hawaii Standard */
  { "hdt",	T_DAYZONE,  HOUR (10) },	/* Hawaii Daylight */
  { "cat",	T_ZONE,     HOUR (10) },	/* Central Alaska */
  { "ahst",	T_ZONE,     HOUR (10) },	/* Alaska-Hawaii Standard */
  { "nt",	T_ZONE,     HOUR (11) },	/* Nome */
  { "idlw",	T_ZONE,     HOUR (12) },	/* International Date Line West */
  { "cet",	T_ZONE,     -HOUR (1) },	/* Central European */
  { "met",	T_ZONE,     -HOUR (1) },	/* Middle European */
  { "mewt",	T_ZONE,     -HOUR (1) },	/* Middle European Winter */
  { "mest",	T_DAYZONE,  -HOUR (1) },	/* Middle European Summer */
  { "mesz",	T_DAYZONE,  -HOUR (1) },	/* Middle European Summer */
  { "swt",	T_ZONE,     -HOUR (1) },	/* Swedish Winter */
  { "sst",	T_DAYZONE,  -HOUR (1) },	/* Swedish Summer */
  { "fwt",	T_ZONE,     -HOUR (1) },	/* French Winter */
  { "fst",	T_DAYZONE,  -HOUR (1) },	/* French Summer */
  { "eet",	T_ZONE,     -HOUR (2) },	/* Eastern Europe, USSR Zone 1 */
  { "bt",	T_ZONE,     -HOUR (3) },	/* Baghdad, USSR Zone 2 */
#if 0
  { "it",	T_ZONE,     -HOUR (3.5) },/* Iran */
#endif
  { "zp4",	T_ZONE,     -HOUR (4) },	/* USSR Zone 3 */
  { "zp5",	T_ZONE,     -HOUR (5) },	/* USSR Zone 4 */
#if 0
  { "ist",	T_ZONE,     -HOUR (5.5) },/* Indian Standard */
#endif
  { "zp6",	T_ZONE,     -HOUR (6) },	/* USSR Zone 5 */
#if	0
  /* For completeness.  NST is also Newfoundland Standard, and SST is
   * also Swedish Summer. */
  { "nst",	T_ZONE,     -HOUR (6.5) },/* North Sumatra */
  { "sst",	T_ZONE,     -HOUR (7) },	/* South Sumatra, USSR Zone 6 */
#endif	/* 0 */
  { "wast",	T_ZONE,     -HOUR (7) },	/* West Australian Standard */
  { "wadt",	T_DAYZONE,  -HOUR (7) },	/* West Australian Daylight */
#if 0
  { "jt",	T_ZONE,     -HOUR (7.5) },/* Java (3pm in Cronusland!) */
#endif
  { "cct",	T_ZONE,     -HOUR (8) },	/* China Coast, USSR Zone 7 */
  { "jst",	T_ZONE,     -HOUR (9) },	/* Japan Standard, USSR Zone 8 */
#if 0
  { "cast",	T_ZONE,     -HOUR (9.5) },/* Central Australian Standard */
  { "cadt",	T_DAYZONE,  -HOUR (9.5) },/* Central Australian Daylight */
#endif
  { "east",	T_ZONE,     -HOUR (10) },	/* Eastern Australian Standard */
  { "eadt",	T_DAYZONE,  -HOUR (10) },	/* Eastern Australian Daylight */
  { "gst",	T_ZONE,     -HOUR (10) },	/* Guam Standard, USSR Zone 9 */
  { "nzt",	T_ZONE,     -HOUR (12) },	/* New Zealand */
  { "nzst",	T_ZONE,     -HOUR (12) },	/* New Zealand Standard */
  { "nzdt",	T_DAYZONE,  -HOUR (12) },	/* New Zealand Daylight */
  { "idle",	T_ZONE,     -HOUR (12) },	/* International Date Line
						   East */
  {  NULL, 0, 0  }
};

/* Military timezone table. */
static SYMBOL const mil_tz_tab[] = {
  { "a",	T_ZONE,	HOUR (  1) },
  { "b",	T_ZONE,	HOUR (  2) },
  { "c",	T_ZONE,	HOUR (  3) },
  { "d",	T_ZONE,	HOUR (  4) },
  { "e",	T_ZONE,	HOUR (  5) },
  { "f",	T_ZONE,	HOUR (  6) },
  { "g",	T_ZONE,	HOUR (  7) },
  { "h",	T_ZONE,	HOUR (  8) },
  { "i",	T_ZONE,	HOUR (  9) },
  { "k",	T_ZONE,	HOUR ( 10) },
  { "l",	T_ZONE,	HOUR ( 11) },
  { "m",	T_ZONE,	HOUR ( 12) },
  { "n",	T_ZONE,	HOUR (- 1) },
  { "o",	T_ZONE,	HOUR (- 2) },
  { "p",	T_ZONE,	HOUR (- 3) },
  { "q",	T_ZONE,	HOUR (- 4) },
  { "r",	T_ZONE,	HOUR (- 5) },
  { "s",	T_ZONE,	HOUR (- 6) },
  { "t",	T_ZONE,	HOUR (- 7) },
  { "u",	T_ZONE,	HOUR (- 8) },
  { "v",	T_ZONE,	HOUR (- 9) },
  { "w",	T_ZONE,	HOUR (-10) },
  { "x",	T_ZONE,	HOUR (-11) },
  { "y",	T_ZONE,	HOUR (-12) },
  { "z",	T_ZONE,	HOUR (  0) },
  { NULL, 0, 0 }
};




/* ARGSUSED */
static int
yyerror (char *s MU_ARG_UNUSED)
{
  return 0;
}

static int
norm_hour (int hours, MERIDIAN meridian)
{
  switch (meridian)
    {
    case MER24:
      if (hours < 0 || hours > 23)
	return -1;
      return hours;
      
    case MERam:
      if (hours < 1 || hours > 12)
	return -1;
      if (hours == 12)
	hours = 0;
      return hours;
      
    case MERpm:
      if (hours < 1 || hours > 12)
	return -1;
      if (hours == 12)
	hours = 0;
      return hours + 12;
      
    default:
      abort ();
    }
  /* NOTREACHED */
}

static int
norm_year (int year)
{
  if (year < 0)
    year = -year;
  
  /* XPG4 suggests that years 00-68 map to 2000-2068, and
     years 69-99 map to 1969-1999.  */
  if (year < 69)
    year += 2000;
  else if (year < 100)
    year += 1900;

  return year;
}

static int
sym_lookup (char *buff)
{
  register char *p;
  register char *q;
  register const SYMBOL *tp;
  int i;
  int abbrev;
  
  /* Make it lowercase. */
  for (p = buff; *p; p++)
    if (ISUPPER ((unsigned char) *p))
      *p = tolower (*p);
  
  if (strcmp (buff, "am") == 0 || strcmp (buff, "a.m.") == 0)
    {
      yylval.meridian = MERam;
      return T_MERIDIAN;
    }
  if (strcmp (buff, "pm") == 0 || strcmp (buff, "p.m.") == 0)
    {
      yylval.meridian = MERpm;
      return T_MERIDIAN;
    }
  
  /* See if we have an abbreviation for a month. */
  if (strlen (buff) == 3)
    abbrev = 1;
  else if (strlen (buff) == 4 && buff[3] == '.')
    {
      abbrev = 1;
      buff[3] = '\0';
    }
  else
    abbrev = 0;

  for (tp = month_day_tab; tp->name; tp++)
    {
      if (abbrev)
	{
	  if (strncmp (buff, tp->name, 3) == 0)
	    {
	      yylval.number = tp->value;
	      return tp->type;
	    }
	}
      else if (strcmp (buff, tp->name) == 0)
	{
	  yylval.number = tp->value;
	  return tp->type;
	}
    }

  for (tp = tz_tab; tp->name; tp++)
    if (strcmp (buff, tp->name) == 0)
      {
	yylval.number = tp->value;
	return tp->type;
      }

  if (strcmp (buff, "dst") == 0)
    return T_DST;

  for (tp = units_tab; tp->name; tp++)
    if (strcmp (buff, tp->name) == 0)
      {
	yylval.number = tp->value;
	return tp->type;
      }

  /* Strip off any plural and try the units table again. */
  i = strlen (buff) - 1;
  if (buff[i] == 's')
    {
      buff[i] = '\0';
      for (tp = units_tab; tp->name; tp++)
	if (strcmp (buff, tp->name) == 0)
	  {
	    yylval.number = tp->value;
	    return tp->type;
	  }
      buff[i] = 's';		/* Put back for "this" in other_tab. */
    }

  for (tp = other_tab; tp->name; tp++)
    if (strcmp (buff, tp->name) == 0)
      {
	yylval.number = tp->value;
	return tp->type;
      }

  /* Military timezones. */
  if (buff[1] == '\0' && ISALPHA ((unsigned char) *buff))
    {
      for (tp = mil_tz_tab; tp->name; tp++)
	if (strcmp (buff, tp->name) == 0)
	  {
	    yylval.number = tp->value;
	    return tp->type;
	  }
    }

  /* Drop out any periods and try the timezone table again. */
  for (i = 0, p = q = buff; *q; q++)
    if (*q != '.')
      *p++ = *q;
    else
      i++;
  *p = '\0';
  if (i)
    for (tp = tz_tab; tp->name; tp++)
      if (strcmp (buff, tp->name) == 0)
	{
	  yylval.number = tp->value;
	  return tp->type;
	}

  return T_ID;
}

static int
yylex ()
{
  register unsigned char c;
  register char *p;
  char buff[20];
  int count;
  int sign;

  for (;;)
    {
      while (ISSPACE ((unsigned char) *yyinput))
	yyinput++;

      if (ISDIGIT (c = *yyinput) || c == '-' || c == '+')
	{
	  if (c == '-' || c == '+')
	    {
	      sign = c == '-' ? -1 : 1;
	      if (!ISDIGIT (*++yyinput))
		/* skip the '-' sign */
		continue;
	    }
	  else
	    sign = 0;
	  for (yylval.number = 0; ISDIGIT (c = *yyinput++);)
	    yylval.number = 10 * yylval.number + c - '0';
	  yyinput--;
	  if (sign < 0)
	    yylval.number = -yylval.number;
	  return sign ? T_SNUMBER : T_UNUMBER;
	}
      if (ISALPHA (c))
	{
	  for (p = buff; (c = *yyinput++, ISALPHA (c)) || c == '.';)
	    if (p < &buff[sizeof buff - 1])
	      *p++ = c;
	  *p = '\0';
	  yyinput--;
	  return sym_lookup (buff);
	}
      if (c != '(')
	return *yyinput++;
      count = 0;
      do
	{
	  c = *yyinput++;
	  if (c == '\0')
	    return c;
	  if (c == '(')
	    count++;
	  else if (c == ')')
	    count--;
	}
      while (count > 0);
    }
}

#define TM_YEAR_ORIGIN 1900

/* Yield A - B, measured in seconds.  */
static long
difftm (struct tm *a, struct tm *b)
{
  int ay = a->tm_year + (TM_YEAR_ORIGIN - 1);
  int by = b->tm_year + (TM_YEAR_ORIGIN - 1);
  long days = (
  /* difference in day of year */
		a->tm_yday - b->tm_yday
  /* + intervening leap days */
		+ ((ay >> 2) - (by >> 2))
		- (ay / 100 - by / 100)
		+ ((ay / 100 >> 2) - (by / 100 >> 2))
  /* + difference in years * 365 */
		+ (long) (ay - by) * 365
  );
  return (60 * (60 * (24 * days + (a->tm_hour - b->tm_hour))
		+ (a->tm_min - b->tm_min))
	  + (a->tm_sec - b->tm_sec));
}

int
mu_parse_date (const char *p, time_t *rettime, const time_t *now)
{
  struct tm tm, tm0, *tmp;
  time_t start;

  yyinput = p;
  start = now ? *now : time ((time_t *) NULL);
  tmp = localtime (&start);
  if (!tmp)
    return -1;

  memset (&tm, 0, sizeof tm);
  tm.tm_isdst = tmp->tm_isdst;

  if (yyparse ())
    return -1;
  
  if (!MASK_IS_SET (pd.date.mask, PD_MASK_YEAR))
    __SET_YEAR (pd.date, tmp->tm_year + TM_YEAR_ORIGIN, return -1);
  if (!MASK_IS_SET (pd.date.mask, PD_MASK_MONTH))
    __SET_MONTH (pd.date, tmp->tm_mon + 1, return -1);
  if (!MASK_IS_SET (pd.date.mask, PD_MASK_DAY))
    __SET_DAY (pd.date, tmp->tm_mday, return -1);
  if (!MASK_IS_SET (pd.date.mask, PD_MASK_HOUR))
    __SET_HOUR (pd.date, tmp->tm_hour, return -1);
  if (!MASK_IS_SET (pd.date.mask, PD_MASK_MERIDIAN))
    __SET_MERIDIAN (pd.date, MER24, return -1);
  if (!MASK_IS_SET (pd.date.mask, PD_MASK_MINUTE))
    __SET_MINUTE (pd.date, tmp->tm_min, return -1);
  if (!MASK_IS_SET (pd.date.mask, PD_MASK_SECOND))
    __SET_SECOND (pd.date, tmp->tm_sec, return -1);
  
  tm.tm_year = norm_year (pd.date.year) - TM_YEAR_ORIGIN + pd.rel.year;
  tm.tm_mon = pd.date.month - 1 + pd.rel.month;
  tm.tm_mday = pd.date.day + pd.rel.day;
  if (MASK_TEST (pd.date.mask, PD_MASK_TIME)
      || (pd.rel.mask && !MASK_TEST (pd.date.mask, PD_MASK_DATE)
	  && !MASK_TEST (pd.date.mask, PD_MASK_DOW)))
    {
      tm.tm_hour = norm_hour (pd.date.hour, pd.date.meridian);
      if (tm.tm_hour < 0)
	return -1;
      tm.tm_min = pd.date.minute;
      tm.tm_sec = pd.date.second;
    }
  else
    {
      tm.tm_hour = tm.tm_min = tm.tm_sec = 0;
    }
  tm.tm_hour += pd.rel.hour;
  tm.tm_min += pd.rel.minute;
  tm.tm_sec += pd.rel.second;

  /* Let mktime deduce tm_isdst if we have an absolute timestamp,
     or if the relative timestamp mentions days, months, or years.  */
  if (MASK_TEST (pd.date.mask, PD_MASK_DATE | PD_MASK_DOW | PD_MASK_TIME)
      || MASK_TEST (pd.rel.mask, PD_MASK_DOW | PD_MASK_MONTH | PD_MASK_YEAR))
    tm.tm_isdst = -1;

  tm0 = tm;

  start = mktime (&tm);

  if (start == (time_t) -1)
    {

      /* Guard against falsely reporting errors near the time_t boundaries
         when parsing times in other time zones.  For example, if the min
         time_t value is 1970-01-01 00:00:00 UTC and we are 8 hours ahead
         of UTC, then the min localtime value is 1970-01-01 08:00:00; if
         we apply mktime to 1970-01-01 00:00:00 we will get an error, so
         we apply mktime to 1970-01-02 08:00:00 instead and adjust the time
         zone by 24 hours to compensate.  This algorithm assumes that
         there is no DST transition within a day of the time_t boundaries.  */
      if (MASK_TEST (pd.date.mask, PD_MASK_TZ))
	{
	  tm = tm0;
	  if (tm.tm_year <= EPOCH - TM_YEAR_ORIGIN)
	    {
	      tm.tm_mday++;
	      pd.date.tz -= 24 * 60;
	    }
	  else
	    {
	      tm.tm_mday--;
	      pd.date.tz += 24 * 60;
	    }
	  start = mktime (&tm);
	}

      if (start == (time_t) -1)
	return -1;
    }

  if (MASK_TEST (pd.date.mask, PD_MASK_DOW)
      && !MASK_TEST (pd.date.mask, PD_MASK_DATE))
    {
      tm.tm_mday += ((pd.date.number - tm.tm_wday + 7) % 7
		     + 7 * (pd.date.ordinal - (0 < pd.date.ordinal)));
      start = mktime (&tm);
      if (start == (time_t) -1)
	return -1;
    }
  
  if (MASK_TEST (pd.date.mask, PD_MASK_TZ))
    {
      long delta;
      struct tm *gmt = gmtime (&start);
      if (gmt)
	{
	  delta = pd.date.tz * 60L + difftm (&tm, gmt);
	  if ((start + delta < start) != (delta < 0))
	    return -1;		/* time_t overflow */
	  start += delta;
	}
    }
  
  *rettime = start;
  return 0;
}

#ifdef STANDALONE
int
main (int argc, char *argv[])
{
  char buff[MAX_BUFF_LEN + 1];
  time_t d;

  if (argc > 1 && strcmp (argv[1], "-d") == 0)
    yydebug++;
  printf ("Enter date, or blank line to exit.\n\t> ");
  fflush (stdout);

  buff[MAX_BUFF_LEN] = 0;
  while (fgets (buff, MAX_BUFF_LEN, stdin) && buff[0])
    {
      if (mu_parse_date (buff, &d, NULL))
	printf ("Bad format - couldn't convert.\n");
      else
	printf ("%s", ctime (&d));
      printf ("\t> ");
      fflush (stdout);
    }
  exit (0);
  /* NOTREACHED */
}

#endif
