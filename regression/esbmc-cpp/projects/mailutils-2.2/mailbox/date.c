/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2007, 2009, 2010 Free Software
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mailutils/mutil.h>
#include <mailutils/cstr.h>

#define SECS_PER_DAY 86400
#define ADJUSTMENT -719162L

static time_t
jan1st (int year)
{
  year--;               /* Do not consider the current year */
  return  year*365L
               + year/4L    /* Years divisible by 4 are leap years */
               + year/400L  /* Years divisible by 400 are always leap years */
               - year/100L; /* Years divisible by 100 but not 400 aren't */
}

static int  month_start[]=
    {    0, 31, 59,  90, 120, 151, 181, 212, 243, 273, 304, 334, 365 };
     /* Jan Feb Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
         31  28  31   30   31   30   31   31   30   31   30   31
     */

/* NOTE: ignore GCC warning. The precedence of operators is OK here */
#define leap_year(y) ((y) % 4 == 0 && (y) % 100 != 0 || (y) % 400 == 0)

static int
dayofyear (time_t *pday, int year, int month, int day)
{
  int  leap, month_days;

  if (year < 0 || month < 0 || month > 11)
    return -1;
    
  leap = leap_year (year);
  
  month_days = month_start[month + 1] - month_start[month]
               + ((month == 2) ? leap : 0);

  if (day < 0 || day > month_days)
    return -1;  /* Illegal Date */

  if (month <= 2)
    leap = 0;

  *pday = month_start[month] + day + leap;
  return 0;
}


/* Convert struct tm into time_t, taking into account timezone offset. */
/* FIXME: It does not take DST into account */
time_t
mu_tm2time (struct tm *tm, mu_timezone *tz)
{
  time_t t;
  
  if (dayofyear (&t, tm->tm_year, tm->tm_mon, tm->tm_mday - 1))
    return -1;
  t = (t + ADJUSTMENT + jan1st (1900 + tm->tm_year)) * SECS_PER_DAY
            + (tm->tm_hour * 60 + tm->tm_min) * 60 + tm->tm_sec
      - tz->utc_offset;
  return t;
}

/* Convert time 0 at UTC to our localtime, that tells us the offset
   of our current timezone from UTC. */
time_t
mu_utc_offset (void)
{
  time_t t = 0;
  struct tm *tm = gmtime (&t);

  return - mktime (tm);
}

static const char *months[] =
{
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", NULL
};

static const char *wdays[] =
{
  "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", NULL
};

int
mu_parse_imap_date_time (const char **p, struct tm *tm, mu_timezone *tz)
{
  int year, mon, day, hour, min, sec;
  char zone[6] = "+0000";	/* ( "+" / "-" ) hhmm */
  char month[5] = "";
  int hh = 0;
  int mm = 0;
  int sign = 1;
  int scanned = 0, scanned3;
  int i;
  int tzoffset;

  day = mon = year = hour = min = sec = 0;

  memset (tm, 0, sizeof (*tm));

  switch (sscanf (*p,
		  "%2d-%3s-%4d%n %2d:%2d:%2d %5s%n",
		  &day, month, &year, &scanned3, &hour, &min, &sec, zone,
		  &scanned))
    {
    case 3:
      scanned = scanned3;
      break;
    case 7:
      break;
    default:
      return -1;
    }

  tm->tm_sec = sec;
  tm->tm_min = min;
  tm->tm_hour = hour;
  tm->tm_mday = day;

  for (i = 0; i < 12; i++)
    {
      if (mu_c_strncasecmp (month, months[i], 3) == 0)
	{
	  mon = i;
	  break;
	}
    }
  tm->tm_mon = mon;
  tm->tm_year = (year > 1900) ? year - 1900 : year;
  tm->tm_yday = 0;		/* unknown. */
  tm->tm_wday = 0;		/* unknown. */
#if HAVE_STRUCT_TM_TM_ISDST
  tm->tm_isdst = -1;		/* unknown. */
#endif

  hh = (zone[1] - '0') * 10 + (zone[2] - '0');
  mm = (zone[3] - '0') * 10 + (zone[4] - '0');
  sign = (zone[0] == '-') ? -1 : +1;
  tzoffset = sign * (hh * 60 * 60 + mm * 60);

#if HAVE_STRUCT_TM_TM_GMTOFF
  tm->tm_gmtoff = tzoffset;
#endif

  if (tz)
    {
      tz->utc_offset = tzoffset;
      tz->tz_name = NULL;
    }

  *p += scanned;

  return 0;
}

/* "ctime" format is: Thu Jul 01 15:58:27 1999, with no trailing \n.  */
int
mu_parse_ctime_date_time (const char **p, struct tm *tm, mu_timezone * tz)
{
  int wday = 0;
  int year = 0;
  int mon = 0;
  int day = 0;
  int hour = 0;
  int min = 0;
  int sec = 0;
  int n = 0;
  int i;
  char weekday[5] = "";
  char month[5] = "";

  if (sscanf (*p, "%3s %3s %2d %2d:%2d:%2d %d%n\n",
	weekday, month, &day, &hour, &min, &sec, &year, &n) != 7)
    return -1;

  *p += n;

  for (i = 0; i < 7; i++)
    {
      if (mu_c_strncasecmp (weekday, wdays[i], 3) == 0)
	{
	  wday = i;
	  break;
	}
    }

  for (i = 0; i < 12; i++)
    {
      if (mu_c_strncasecmp (month, months[i], 3) == 0)
	{
	  mon = i;
	  break;
	}
    }

  if (tm)
    {
      memset (tm, 0, sizeof (struct tm));

      tm->tm_sec = sec;
      tm->tm_min = min;
      tm->tm_hour = hour;
      tm->tm_mday = day;
      tm->tm_wday = wday;
      tm->tm_mon = mon;
      tm->tm_year = (year > 1900) ? year - 1900 : year;
#ifdef HAVE_STRUCT_TM_TM_ISDST
      tm->tm_isdst = -1;	/* unknown. */
#endif
    }

  /* ctime has no timezone information, set tz to UTC if they ask. */
  if (tz)
    memset (tz, 0, sizeof (struct mu_timezone));

  return 0;
}
