/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with pd_yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum pd_yytokentype {
     T_AGO = 258,
     T_DST = 259,
     T_ID = 260,
     T_DAY = 261,
     T_DAY_UNIT = 262,
     T_DAYZONE = 263,
     T_HOUR_UNIT = 264,
     T_MINUTE_UNIT = 265,
     T_MONTH = 266,
     T_MONTH_UNIT = 267,
     T_SEC_UNIT = 268,
     T_SNUMBER = 269,
     T_UNUMBER = 270,
     T_YEAR_UNIT = 271,
     T_ZONE = 272,
     T_MERIDIAN = 273
   };
#endif
/* Tokens.  */
#define T_AGO 258
#define T_DST 259
#define T_ID 260
#define T_DAY 261
#define T_DAY_UNIT 262
#define T_DAYZONE 263
#define T_HOUR_UNIT 264
#define T_MINUTE_UNIT 265
#define T_MONTH 266
#define T_MONTH_UNIT 267
#define T_SEC_UNIT 268
#define T_SNUMBER 269
#define T_UNUMBER 270
#define T_YEAR_UNIT 271
#define T_ZONE 272
#define T_MERIDIAN 273




/* Copy the first part of user declarations.  */
#line 1 "parsedate.y"

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

static int pd_yylex ();
static int pd_yyerror ();

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
 
static const char	*pd_yyinput;



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 214 "parsedate.y"
{
  int number;
  enum meridian meridian;
  struct pd_date date;
  struct pd_datespec datespec;
}
/* Line 187 of yacc.c.  */
#line 352 "parsedate.c"
	YYSTYPE;
# define pd_yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 365 "parsedate.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 pd_yytype_uint8;
#else
typedef unsigned char pd_yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 pd_yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char pd_yytype_int8;
#else
typedef short int pd_yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 pd_yytype_uint16;
#else
typedef unsigned short int pd_yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 pd_yytype_int16;
#else
typedef short int pd_yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined pd_yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined pd_yyoverflow || YYERROR_VERBOSE */


#if (! defined pd_yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union pd_yyalloc
{
  pd_yytype_int16 pd_yyss;
  YYSTYPE pd_yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union pd_yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (pd_yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T pd_yyi;				\
	  for (pd_yyi = 0; pd_yyi < (Count); pd_yyi++)	\
	    (To)[pd_yyi] = (From)[pd_yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T pd_yynewbytes;						\
	YYCOPY (&pd_yyptr->Stack, Stack, pd_yysize);				\
	Stack = &pd_yyptr->Stack;						\
	pd_yynewbytes = pd_yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	pd_yyptr += pd_yynewbytes / sizeof (*pd_yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   72

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  22
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  12
/* YYNRULES -- Number of rules.  */
#define YYNRULES  54
/* YYNRULES -- Number of states.  */
#define YYNSTATES  69

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   273

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? pd_yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const pd_yytype_uint8 pd_yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,    20,     2,     2,    21,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    19,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const pd_yytype_uint8 pd_yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    12,    15,    17,    19,
      21,    23,    26,    31,    36,    43,    50,    52,    54,    57,
      59,    62,    65,    69,    75,    79,    83,    86,    91,    94,
      98,   104,   107,   109,   111,   114,   117,   120,   122,   125,
     128,   130,   133,   136,   138,   141,   144,   146,   149,   152,
     154,   157,   160,   162,   163
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const pd_yytype_int8 pd_yyrhs[] =
{
      23,     0,    -1,    24,    -1,    -1,    24,    25,    -1,    24,
      30,    -1,    24,    15,    -1,    26,    -1,    27,    -1,    29,
      -1,    28,    -1,    15,    18,    -1,    15,    19,    15,    33,
      -1,    15,    19,    15,    14,    -1,    15,    19,    15,    19,
      15,    33,    -1,    15,    19,    15,    19,    15,    14,    -1,
      17,    -1,     8,    -1,    17,     4,    -1,     6,    -1,     6,
      20,    -1,    15,     6,    -1,    15,    21,    15,    -1,    15,
      21,    15,    21,    15,    -1,    15,    14,    14,    -1,    15,
      11,    14,    -1,    11,    15,    -1,    11,    15,    20,    15,
      -1,    15,    11,    -1,    15,    11,    15,    -1,     6,    11,
      15,    26,    15,    -1,    31,     3,    -1,    31,    -1,    32,
      -1,    31,    32,    -1,    15,    16,    -1,    14,    16,    -1,
      16,    -1,    15,    12,    -1,    14,    12,    -1,    12,    -1,
      15,     7,    -1,    14,     7,    -1,     7,    -1,    15,     9,
      -1,    14,     9,    -1,     9,    -1,    15,    10,    -1,    14,
      10,    -1,    10,    -1,    15,    13,    -1,    14,    13,    -1,
      13,    -1,    -1,    18,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const pd_yytype_uint16 pd_yyrline[] =
{
       0,   235,   235,   242,   246,   252,   258,   289,   290,   291,
     292,   295,   301,   308,   319,   327,   340,   345,   350,   357,
     363,   369,   377,   383,   403,   411,   428,   434,   441,   447,
     454,   469,   479,   483,   489,   497,   502,   507,   512,   517,
     522,   527,   532,   537,   542,   547,   552,   557,   562,   567,
     572,   577,   582,   590,   593
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const pd_yytname[] =
{
  "$end", "error", "$undefined", "T_AGO", "T_DST", "T_ID", "T_DAY",
  "T_DAY_UNIT", "T_DAYZONE", "T_HOUR_UNIT", "T_MINUTE_UNIT", "T_MONTH",
  "T_MONTH_UNIT", "T_SEC_UNIT", "T_SNUMBER", "T_UNUMBER", "T_YEAR_UNIT",
  "T_ZONE", "T_MERIDIAN", "':'", "','", "'/'", "$accept", "input", "spec",
  "item", "time", "zone", "day", "date", "rel", "relspec", "relunit",
  "o_merid", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const pd_yytype_uint16 pd_yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,    58,
      44,    47
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const pd_yytype_uint8 pd_yyr1[] =
{
       0,    22,    23,    24,    24,    24,    24,    25,    25,    25,
      25,    26,    26,    26,    26,    26,    27,    27,    27,    28,
      28,    28,    29,    29,    29,    29,    29,    29,    29,    29,
      29,    30,    30,    31,    31,    32,    32,    32,    32,    32,
      32,    32,    32,    32,    32,    32,    32,    32,    32,    32,
      32,    32,    32,    33,    33
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const pd_yytype_uint8 pd_yyr2[] =
{
       0,     2,     1,     0,     2,     2,     2,     1,     1,     1,
       1,     2,     4,     4,     6,     6,     1,     1,     2,     1,
       2,     2,     3,     5,     3,     3,     2,     4,     2,     3,
       5,     2,     1,     1,     2,     2,     2,     1,     2,     2,
       1,     2,     2,     1,     2,     2,     1,     2,     2,     1,
       2,     2,     1,     0,     1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const pd_yytype_uint8 pd_yydefact[] =
{
       3,     0,     2,     1,    19,    43,    17,    46,    49,     0,
      40,    52,     0,     6,    37,    16,     4,     7,     8,    10,
       9,     5,    32,    33,     0,    20,    26,    42,    45,    48,
      39,    51,    36,    21,    41,    44,    47,    28,    38,    50,
       0,    35,    11,     0,     0,    18,    31,     0,    34,     0,
       0,    25,    29,    24,    53,    22,     0,     0,    27,    13,
      54,     0,    12,     0,    30,    53,    23,    15,    14
};

/* YYDEFGOTO[NTERM-NUM].  */
static const pd_yytype_int8 pd_yydefgoto[] =
{
      -1,     1,     2,    16,    17,    18,    19,    20,    21,    22,
      23,    62
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -10
static const pd_yytype_int8 pd_yypact[] =
{
     -10,     9,    22,   -10,    -9,   -10,   -10,   -10,   -10,     4,
     -10,   -10,    33,    -6,   -10,    18,   -10,   -10,   -10,   -10,
     -10,   -10,    11,   -10,    26,   -10,    24,   -10,   -10,   -10,
     -10,   -10,   -10,   -10,   -10,   -10,   -10,     2,   -10,   -10,
      38,   -10,   -10,    32,    40,   -10,   -10,    41,   -10,    50,
      51,   -10,   -10,   -10,    42,    46,    45,    53,   -10,   -10,
     -10,    54,   -10,    55,   -10,    44,   -10,   -10,   -10
};

/* YYPGOTO[NTERM-NUM].  */
static const pd_yytype_int8 pd_yypgoto[] =
{
     -10,   -10,   -10,   -10,    10,   -10,   -10,   -10,   -10,   -10,
      49,     7
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const pd_yytype_uint8 pd_yytable[] =
{
      33,    34,    24,    35,    36,    37,    38,    39,    40,     3,
      41,    25,    42,    43,    46,    44,    51,    52,     5,    26,
       7,     8,    45,    10,    11,    12,    47,    14,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      27,    49,    28,    29,    50,    30,    31,    54,    34,    32,
      35,    36,    53,    38,    39,    55,    59,    41,    67,    57,
      60,    61,    60,    42,    43,    56,    58,    63,    64,    65,
      66,    48,    68
};

static const pd_yytype_uint8 pd_yycheck[] =
{
       6,     7,    11,     9,    10,    11,    12,    13,    14,     0,
      16,    20,    18,    19,     3,    21,    14,    15,     7,    15,
       9,    10,     4,    12,    13,    14,    15,    16,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
       7,    15,     9,    10,    20,    12,    13,    15,     7,    16,
       9,    10,    14,    12,    13,    15,    14,    16,    14,    49,
      18,    19,    18,    18,    19,    15,    15,    21,    15,    15,
      15,    22,    65
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const pd_yytype_uint8 pd_yystos[] =
{
       0,    23,    24,     0,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    25,    26,    27,    28,
      29,    30,    31,    32,    11,    20,    15,     7,     9,    10,
      12,    13,    16,     6,     7,     9,    10,    11,    12,    13,
      14,    16,    18,    19,    21,     4,     3,    15,    32,    15,
      20,    14,    15,    14,    15,    15,    15,    26,    15,    14,
      18,    19,    33,    21,    15,    15,    15,    14,    33
};

#define pd_yyerrok		(pd_yyerrstatus = 0)
#define pd_yyclearin	(pd_yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto pd_yyacceptlab
#define YYABORT		goto pd_yyabortlab
#define YYERROR		goto pd_yyerrorlab


/* Like YYERROR except do call pd_yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto pd_yyerrlab

#define YYRECOVERING()  (!!pd_yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (pd_yychar == YYEMPTY && pd_yylen == 1)				\
    {								\
      pd_yychar = (Token);						\
      pd_yylval = (Value);						\
      pd_yytoken = YYTRANSLATE (pd_yychar);				\
      YYPOPSTACK (1);						\
      goto pd_yybackup;						\
    }								\
  else								\
    {								\
      pd_yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `pd_yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX pd_yylex (YYLEX_PARAM)
#else
# define YYLEX pd_yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (pd_yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (pd_yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      pd_yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
pd_yy_symbol_value_print (FILE *pd_yyoutput, int pd_yytype, YYSTYPE const * const pd_yyvaluep)
#else
static void
pd_yy_symbol_value_print (pd_yyoutput, pd_yytype, pd_yyvaluep)
    FILE *pd_yyoutput;
    int pd_yytype;
    YYSTYPE const * const pd_yyvaluep;
#endif
{
  if (!pd_yyvaluep)
    return;
# ifdef YYPRINT
  if (pd_yytype < YYNTOKENS)
    YYPRINT (pd_yyoutput, pd_yytoknum[pd_yytype], *pd_yyvaluep);
# else
  YYUSE (pd_yyoutput);
# endif
  switch (pd_yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
pd_yy_symbol_print (FILE *pd_yyoutput, int pd_yytype, YYSTYPE const * const pd_yyvaluep)
#else
static void
pd_yy_symbol_print (pd_yyoutput, pd_yytype, pd_yyvaluep)
    FILE *pd_yyoutput;
    int pd_yytype;
    YYSTYPE const * const pd_yyvaluep;
#endif
{
  if (pd_yytype < YYNTOKENS)
    YYFPRINTF (pd_yyoutput, "token %s (", pd_yytname[pd_yytype]);
  else
    YYFPRINTF (pd_yyoutput, "nterm %s (", pd_yytname[pd_yytype]);

  pd_yy_symbol_value_print (pd_yyoutput, pd_yytype, pd_yyvaluep);
  YYFPRINTF (pd_yyoutput, ")");
}

/*------------------------------------------------------------------.
| pd_yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
pd_yy_stack_print (pd_yytype_int16 *bottom, pd_yytype_int16 *top)
#else
static void
pd_yy_stack_print (bottom, top)
    pd_yytype_int16 *bottom;
    pd_yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (pd_yydebug)							\
    pd_yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
pd_yy_reduce_print (YYSTYPE *pd_yyvsp, int pd_yyrule)
#else
static void
pd_yy_reduce_print (pd_yyvsp, pd_yyrule)
    YYSTYPE *pd_yyvsp;
    int pd_yyrule;
#endif
{
  int pd_yynrhs = pd_yyr2[pd_yyrule];
  int pd_yyi;
  unsigned long int pd_yylno = pd_yyrline[pd_yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     pd_yyrule - 1, pd_yylno);
  /* The symbols being reduced.  */
  for (pd_yyi = 0; pd_yyi < pd_yynrhs; pd_yyi++)
    {
      fprintf (stderr, "   $%d = ", pd_yyi + 1);
      pd_yy_symbol_print (stderr, pd_yyrhs[pd_yyprhs[pd_yyrule] + pd_yyi],
		       &(pd_yyvsp[(pd_yyi + 1) - (pd_yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (pd_yydebug)				\
    pd_yy_reduce_print (pd_yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int pd_yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef pd_yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define pd_yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
pd_yystrlen (const char *pd_yystr)
#else
static YYSIZE_T
pd_yystrlen (pd_yystr)
    const char *pd_yystr;
#endif
{
  YYSIZE_T pd_yylen;
  for (pd_yylen = 0; pd_yystr[pd_yylen]; pd_yylen++)
    continue;
  return pd_yylen;
}
#  endif
# endif

# ifndef pd_yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define pd_yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
pd_yystpcpy (char *pd_yydest, const char *pd_yysrc)
#else
static char *
pd_yystpcpy (pd_yydest, pd_yysrc)
    char *pd_yydest;
    const char *pd_yysrc;
#endif
{
  char *pd_yyd = pd_yydest;
  const char *pd_yys = pd_yysrc;

  while ((*pd_yyd++ = *pd_yys++) != '\0')
    continue;

  return pd_yyd - 1;
}
#  endif
# endif

# ifndef pd_yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for pd_yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from pd_yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
pd_yytnamerr (char *pd_yyres, const char *pd_yystr)
{
  if (*pd_yystr == '"')
    {
      YYSIZE_T pd_yyn = 0;
      char const *pd_yyp = pd_yystr;

      for (;;)
	switch (*++pd_yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++pd_yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (pd_yyres)
	      pd_yyres[pd_yyn] = *pd_yyp;
	    pd_yyn++;
	    break;

	  case '"':
	    if (pd_yyres)
	      pd_yyres[pd_yyn] = '\0';
	    return pd_yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! pd_yyres)
    return pd_yystrlen (pd_yystr);

  return pd_yystpcpy (pd_yyres, pd_yystr) - pd_yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
pd_yysyntax_error (char *pd_yyresult, int pd_yystate, int pd_yychar)
{
  int pd_yyn = pd_yypact[pd_yystate];

  if (! (YYPACT_NINF < pd_yyn && pd_yyn <= YYLAST))
    return 0;
  else
    {
      int pd_yytype = YYTRANSLATE (pd_yychar);
      YYSIZE_T pd_yysize0 = pd_yytnamerr (0, pd_yytname[pd_yytype]);
      YYSIZE_T pd_yysize = pd_yysize0;
      YYSIZE_T pd_yysize1;
      int pd_yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *pd_yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int pd_yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *pd_yyfmt;
      char const *pd_yyf;
      static char const pd_yyunexpected[] = "syntax error, unexpected %s";
      static char const pd_yyexpecting[] = ", expecting %s";
      static char const pd_yyor[] = " or %s";
      char pd_yyformat[sizeof pd_yyunexpected
		    + sizeof pd_yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof pd_yyor - 1))];
      char const *pd_yyprefix = pd_yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int pd_yyxbegin = pd_yyn < 0 ? -pd_yyn : 0;

      /* Stay within bounds of both pd_yycheck and pd_yytname.  */
      int pd_yychecklim = YYLAST - pd_yyn + 1;
      int pd_yyxend = pd_yychecklim < YYNTOKENS ? pd_yychecklim : YYNTOKENS;
      int pd_yycount = 1;

      pd_yyarg[0] = pd_yytname[pd_yytype];
      pd_yyfmt = pd_yystpcpy (pd_yyformat, pd_yyunexpected);

      for (pd_yyx = pd_yyxbegin; pd_yyx < pd_yyxend; ++pd_yyx)
	if (pd_yycheck[pd_yyx + pd_yyn] == pd_yyx && pd_yyx != YYTERROR)
	  {
	    if (pd_yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		pd_yycount = 1;
		pd_yysize = pd_yysize0;
		pd_yyformat[sizeof pd_yyunexpected - 1] = '\0';
		break;
	      }
	    pd_yyarg[pd_yycount++] = pd_yytname[pd_yyx];
	    pd_yysize1 = pd_yysize + pd_yytnamerr (0, pd_yytname[pd_yyx]);
	    pd_yysize_overflow |= (pd_yysize1 < pd_yysize);
	    pd_yysize = pd_yysize1;
	    pd_yyfmt = pd_yystpcpy (pd_yyfmt, pd_yyprefix);
	    pd_yyprefix = pd_yyor;
	  }

      pd_yyf = YY_(pd_yyformat);
      pd_yysize1 = pd_yysize + pd_yystrlen (pd_yyf);
      pd_yysize_overflow |= (pd_yysize1 < pd_yysize);
      pd_yysize = pd_yysize1;

      if (pd_yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (pd_yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *pd_yyp = pd_yyresult;
	  int pd_yyi = 0;
	  while ((*pd_yyp = *pd_yyf) != '\0')
	    {
	      if (*pd_yyp == '%' && pd_yyf[1] == 's' && pd_yyi < pd_yycount)
		{
		  pd_yyp += pd_yytnamerr (pd_yyp, pd_yyarg[pd_yyi++]);
		  pd_yyf += 2;
		}
	      else
		{
		  pd_yyp++;
		  pd_yyf++;
		}
	    }
	}
      return pd_yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
pd_yydestruct (const char *pd_yymsg, int pd_yytype, YYSTYPE *pd_yyvaluep)
#else
static void
pd_yydestruct (pd_yymsg, pd_yytype, pd_yyvaluep)
    const char *pd_yymsg;
    int pd_yytype;
    YYSTYPE *pd_yyvaluep;
#endif
{
  YYUSE (pd_yyvaluep);

  if (!pd_yymsg)
    pd_yymsg = "Deleting";
  YY_SYMBOL_PRINT (pd_yymsg, pd_yytype, pd_yyvaluep, pd_yylocationp);

  switch (pd_yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int pd_yyparse (void *YYPARSE_PARAM);
#else
int pd_yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int pd_yyparse (void);
#else
int pd_yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int pd_yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE pd_yylval;

/* Number of syntax errors so far.  */
int pd_yynerrs;



/*----------.
| pd_yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
pd_yyparse (void *YYPARSE_PARAM)
#else
int
pd_yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
pd_yyparse (void)
#else
int
pd_yyparse ()

#endif
#endif
{
  
  int pd_yystate;
  int pd_yyn;
  int pd_yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int pd_yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int pd_yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char pd_yymsgbuf[128];
  char *pd_yymsg = pd_yymsgbuf;
  YYSIZE_T pd_yymsg_alloc = sizeof pd_yymsgbuf;
#endif

  /* Three stacks and their tools:
     `pd_yyss': related to states,
     `pd_yyvs': related to semantic values,
     `pd_yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow pd_yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  pd_yytype_int16 pd_yyssa[YYINITDEPTH];
  pd_yytype_int16 *pd_yyss = pd_yyssa;
  pd_yytype_int16 *pd_yyssp;

  /* The semantic value stack.  */
  YYSTYPE pd_yyvsa[YYINITDEPTH];
  YYSTYPE *pd_yyvs = pd_yyvsa;
  YYSTYPE *pd_yyvsp;



#define YYPOPSTACK(N)   (pd_yyvsp -= (N), pd_yyssp -= (N))

  YYSIZE_T pd_yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE pd_yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int pd_yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  pd_yystate = 0;
  pd_yyerrstatus = 0;
  pd_yynerrs = 0;
  pd_yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  pd_yyssp = pd_yyss;
  pd_yyvsp = pd_yyvs;

  goto pd_yysetstate;

/*------------------------------------------------------------.
| pd_yynewstate -- Push a new state, which is found in pd_yystate.  |
`------------------------------------------------------------*/
 pd_yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  pd_yyssp++;

 pd_yysetstate:
  *pd_yyssp = pd_yystate;

  if (pd_yyss + pd_yystacksize - 1 <= pd_yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T pd_yysize = pd_yyssp - pd_yyss + 1;

#ifdef pd_yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *pd_yyvs1 = pd_yyvs;
	pd_yytype_int16 *pd_yyss1 = pd_yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if pd_yyoverflow is a macro.  */
	pd_yyoverflow (YY_("memory exhausted"),
		    &pd_yyss1, pd_yysize * sizeof (*pd_yyssp),
		    &pd_yyvs1, pd_yysize * sizeof (*pd_yyvsp),

		    &pd_yystacksize);

	pd_yyss = pd_yyss1;
	pd_yyvs = pd_yyvs1;
      }
#else /* no pd_yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto pd_yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= pd_yystacksize)
	goto pd_yyexhaustedlab;
      pd_yystacksize *= 2;
      if (YYMAXDEPTH < pd_yystacksize)
	pd_yystacksize = YYMAXDEPTH;

      {
	pd_yytype_int16 *pd_yyss1 = pd_yyss;
	union pd_yyalloc *pd_yyptr =
	  (union pd_yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (pd_yystacksize));
	if (! pd_yyptr)
	  goto pd_yyexhaustedlab;
	YYSTACK_RELOCATE (pd_yyss);
	YYSTACK_RELOCATE (pd_yyvs);

#  undef YYSTACK_RELOCATE
	if (pd_yyss1 != pd_yyssa)
	  YYSTACK_FREE (pd_yyss1);
      }
# endif
#endif /* no pd_yyoverflow */

      pd_yyssp = pd_yyss + pd_yysize - 1;
      pd_yyvsp = pd_yyvs + pd_yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) pd_yystacksize));

      if (pd_yyss + pd_yystacksize - 1 <= pd_yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", pd_yystate));

  goto pd_yybackup;

/*-----------.
| pd_yybackup.  |
`-----------*/
pd_yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  pd_yyn = pd_yypact[pd_yystate];
  if (pd_yyn == YYPACT_NINF)
    goto pd_yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (pd_yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      pd_yychar = YYLEX;
    }

  if (pd_yychar <= YYEOF)
    {
      pd_yychar = pd_yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      pd_yytoken = YYTRANSLATE (pd_yychar);
      YY_SYMBOL_PRINT ("Next token is", pd_yytoken, &pd_yylval, &pd_yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  pd_yyn += pd_yytoken;
  if (pd_yyn < 0 || YYLAST < pd_yyn || pd_yycheck[pd_yyn] != pd_yytoken)
    goto pd_yydefault;
  pd_yyn = pd_yytable[pd_yyn];
  if (pd_yyn <= 0)
    {
      if (pd_yyn == 0 || pd_yyn == YYTABLE_NINF)
	goto pd_yyerrlab;
      pd_yyn = -pd_yyn;
      goto pd_yyreduce;
    }

  if (pd_yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (pd_yyerrstatus)
    pd_yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", pd_yytoken, &pd_yylval, &pd_yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (pd_yychar != YYEOF)
    pd_yychar = YYEMPTY;

  pd_yystate = pd_yyn;
  *++pd_yyvsp = pd_yylval;

  goto pd_yynewstate;


/*-----------------------------------------------------------.
| pd_yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
pd_yydefault:
  pd_yyn = pd_yydefact[pd_yystate];
  if (pd_yyn == 0)
    goto pd_yyerrlab;
  goto pd_yyreduce;


/*-----------------------------.
| pd_yyreduce -- Do a reduction.  |
`-----------------------------*/
pd_yyreduce:
  /* pd_yyn is the number of a rule to reduce with.  */
  pd_yylen = pd_yyr2[pd_yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  pd_yyval = pd_yyvsp[1-pd_yylen];


  YY_REDUCE_PRINT (pd_yyn);
  switch (pd_yyn)
    {
        case 2:
#line 236 "parsedate.y"
    {
	    pd = (pd_yyvsp[(1) - (1)].datespec);
	  }
    break;

  case 3:
#line 242 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.datespec).date);
	    DATE_INIT ((pd_yyval.datespec).rel);
	  }
    break;

  case 4:
#line 247 "parsedate.y"
    {
	    if (pd_date_union (&(pd_yyvsp[(1) - (2)].datespec).date, &(pd_yyvsp[(2) - (2)].date)))
	      YYERROR;
	    (pd_yyval.datespec) = (pd_yyvsp[(1) - (2)].datespec);
	  }
    break;

  case 5:
#line 253 "parsedate.y"
    {
	    if (pd_date_union (&(pd_yyvsp[(1) - (2)].datespec).rel, &(pd_yyvsp[(2) - (2)].date)))
	      YYERROR;
	    (pd_yyval.datespec) = (pd_yyvsp[(1) - (2)].datespec);
	  }
    break;

  case 6:
#line 259 "parsedate.y"
    {
	    if (MASK_IS_SET ((pd_yyvsp[(1) - (2)].datespec).date.mask, (PD_MASK_TIME|PD_MASK_DATE))
		&& !(pd_yyvsp[(1) - (2)].datespec).rel.mask)
	      SET_YEAR ((pd_yyvsp[(1) - (2)].datespec).date, (pd_yyvsp[(2) - (2)].number));
	    else
	      {
		if ((pd_yyvsp[(2) - (2)].number) > 10000)
		  {
		    SET_DAY ((pd_yyvsp[(1) - (2)].datespec).date, (pd_yyvsp[(2) - (2)].number) % 100);
		    SET_MONTH ((pd_yyvsp[(1) - (2)].datespec).date, ((pd_yyvsp[(2) - (2)].number) / 100) %100);
		    SET_YEAR ((pd_yyvsp[(1) - (2)].datespec).date, (pd_yyvsp[(2) - (2)].number) / 10000);
		  }
		else
		  {
		    if ((pd_yyvsp[(2) - (2)].number) < 100)
		      {
			SET_YEAR ((pd_yyvsp[(1) - (2)].datespec).date, (pd_yyvsp[(2) - (2)].number));
		      }
		    else
		      {
		    	SET_HOUR ((pd_yyvsp[(1) - (2)].datespec).date, (pd_yyvsp[(2) - (2)].number) / 100);
		    	SET_MINUTE ((pd_yyvsp[(1) - (2)].datespec).date, (pd_yyvsp[(2) - (2)].number) % 100);
		      }
		    SET_MERIDIAN ((pd_yyvsp[(1) - (2)].datespec).date, MER24);
		  }
	      }
	    (pd_yyval.datespec) = (pd_yyvsp[(1) - (2)].datespec);
	  }
    break;

  case 11:
#line 296 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_HOUR ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number));
	    SET_MERIDIAN ((pd_yyval.date), (pd_yyvsp[(2) - (2)].meridian));
	  }
    break;

  case 12:
#line 302 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_HOUR ((pd_yyval.date), (pd_yyvsp[(1) - (4)].number));
	    SET_MINUTE ((pd_yyval.date), (pd_yyvsp[(3) - (4)].number));
	    SET_MERIDIAN ((pd_yyval.date), (pd_yyvsp[(4) - (4)].meridian));
	  }
    break;

  case 13:
#line 309 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_HOUR ((pd_yyval.date), (pd_yyvsp[(1) - (4)].number));
	    SET_MINUTE ((pd_yyval.date), (pd_yyvsp[(3) - (4)].number));
	    SET_MERIDIAN ((pd_yyval.date), MER24);
	    SET_TZ ((pd_yyval.date), ((pd_yyvsp[(4) - (4)].number) < 0
			   ? -(pd_yyvsp[(4) - (4)].number) % 100 + (-(pd_yyvsp[(4) - (4)].number) / 100) * 60
			   : - ((pd_yyvsp[(4) - (4)].number) % 100 + ((pd_yyvsp[(4) - (4)].number) / 100) * 60)));

	  }
    break;

  case 14:
#line 320 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_HOUR ((pd_yyval.date), (pd_yyvsp[(1) - (6)].number));
	    SET_MINUTE ((pd_yyval.date), (pd_yyvsp[(3) - (6)].number));
	    SET_SECOND ((pd_yyval.date), (pd_yyvsp[(5) - (6)].number));
	    SET_MERIDIAN ((pd_yyval.date), (pd_yyvsp[(6) - (6)].meridian));
	  }
    break;

  case 15:
#line 328 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_HOUR ((pd_yyval.date), (pd_yyvsp[(1) - (6)].number));
	    SET_MINUTE ((pd_yyval.date), (pd_yyvsp[(3) - (6)].number));
	    SET_SECOND ((pd_yyval.date), (pd_yyvsp[(5) - (6)].number));
	    SET_MERIDIAN ((pd_yyval.date), MER24);
	    SET_TZ ((pd_yyval.date), ((pd_yyvsp[(6) - (6)].number) < 0
			 ? -(pd_yyvsp[(6) - (6)].number) % 100 + (-(pd_yyvsp[(6) - (6)].number) / 100) * 60
			 : - ((pd_yyvsp[(6) - (6)].number) % 100 + ((pd_yyvsp[(6) - (6)].number) / 100) * 60)));
	  }
    break;

  case 16:
#line 341 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_TZ ((pd_yyval.date), (pd_yyvsp[(1) - (1)].number));
	  }
    break;

  case 17:
#line 346 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_TZ ((pd_yyval.date), (pd_yyvsp[(1) - (1)].number) - 60);
	  }
    break;

  case 18:
#line 351 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_TZ ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) - 60);
	  }
    break;

  case 19:
#line 358 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_ORDINAL ((pd_yyval.date), 1);
	    SET_NUMBER ((pd_yyval.date), (pd_yyvsp[(1) - (1)].number));
	  }
    break;

  case 20:
#line 364 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_ORDINAL ((pd_yyval.date), 1);
	    SET_NUMBER ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number));
	  }
    break;

  case 21:
#line 370 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_ORDINAL ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number));
	    SET_NUMBER ((pd_yyval.date), (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 22:
#line 378 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MONTH ((pd_yyval.date), (pd_yyvsp[(1) - (3)].number));
	    SET_DAY ((pd_yyval.date), (pd_yyvsp[(3) - (3)].number));
	  }
    break;

  case 23:
#line 384 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    /* Interpret as YYYY/MM/DD if $1 >= 1000, otherwise as MM/DD/YY.
	       The goal in recognizing YYYY/MM/DD is solely to support legacy
	       machine-generated dates like those in an RCS log listing.  If
	       you want portability, use the ISO 8601 format.  */
	    if ((pd_yyvsp[(1) - (5)].number) >= 1000)
	      {
		SET_YEAR ((pd_yyval.date), (pd_yyvsp[(1) - (5)].number));
		SET_MONTH ((pd_yyval.date), (pd_yyvsp[(3) - (5)].number));
		SET_DAY ((pd_yyval.date), (pd_yyvsp[(5) - (5)].number));
	      }
	    else
	      {
		SET_MONTH ((pd_yyval.date), (pd_yyvsp[(1) - (5)].number));
		SET_DAY ((pd_yyval.date), (pd_yyvsp[(3) - (5)].number));
		SET_YEAR ((pd_yyval.date), (pd_yyvsp[(5) - (5)].number));
	      }
	  }
    break;

  case 24:
#line 404 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    /* ISO 8601 format.  pd_yypd_yy-mm-dd.  */
	    SET_YEAR ((pd_yyval.date), (pd_yyvsp[(1) - (3)].number));
	    SET_MONTH ((pd_yyval.date), -(pd_yyvsp[(2) - (3)].number));
	    SET_DAY ((pd_yyval.date), -(pd_yyvsp[(3) - (3)].number));
	  }
    break;

  case 25:
#line 412 "parsedate.y"
    {
	    /* either 17-JUN-1992 or 1992-JUN-17 */
	    DATE_INIT ((pd_yyval.date));
	    if ((pd_yyvsp[(1) - (3)].number) < 32)
	      {
		SET_DAY ((pd_yyval.date), (pd_yyvsp[(1) - (3)].number));
		SET_MONTH ((pd_yyval.date), (pd_yyvsp[(2) - (3)].number));
		SET_YEAR ((pd_yyval.date), -(pd_yyvsp[(3) - (3)].number));
	      }
	    else
	      {
		SET_DAY ((pd_yyval.date), -(pd_yyvsp[(3) - (3)].number));
		SET_MONTH ((pd_yyval.date), (pd_yyvsp[(2) - (3)].number));
		SET_YEAR ((pd_yyval.date), (pd_yyvsp[(1) - (3)].number));
	      }
	  }
    break;

  case 26:
#line 429 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MONTH ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number));
	    SET_DAY ((pd_yyval.date), (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 27:
#line 435 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MONTH ((pd_yyval.date), (pd_yyvsp[(1) - (4)].number));
	    SET_DAY ((pd_yyval.date), (pd_yyvsp[(2) - (4)].number));
	    SET_YEAR ((pd_yyval.date), (pd_yyvsp[(4) - (4)].number));
	  }
    break;

  case 28:
#line 442 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MONTH ((pd_yyval.date), (pd_yyvsp[(2) - (2)].number));
	    SET_DAY ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number));
	  }
    break;

  case 29:
#line 448 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MONTH ((pd_yyval.date), (pd_yyvsp[(2) - (3)].number));
	    SET_DAY ((pd_yyval.date), (pd_yyvsp[(1) - (3)].number));
	    SET_YEAR ((pd_yyval.date), (pd_yyvsp[(3) - (3)].number));
	  }
    break;

  case 30:
#line 455 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));

	    SET_ORDINAL ((pd_yyval.date), 1);
	    SET_NUMBER ((pd_yyval.date), (pd_yyvsp[(1) - (5)].number));

	    SET_MONTH ((pd_yyval.date), (pd_yyvsp[(2) - (5)].number));
	    SET_DAY ((pd_yyval.date), (pd_yyvsp[(3) - (5)].number));
	    SET_YEAR ((pd_yyval.date), (pd_yyvsp[(5) - (5)].number));
	    if (pd_date_union (&(pd_yyval.date), &(pd_yyvsp[(4) - (5)].date)))
	      YYERROR;
	  }
    break;

  case 31:
#line 470 "parsedate.y"
    {
	    (pd_yyvsp[(1) - (2)].date).second = - (pd_yyvsp[(1) - (2)].date).second;
	    (pd_yyvsp[(1) - (2)].date).minute = - (pd_yyvsp[(1) - (2)].date).minute;
	    (pd_yyvsp[(1) - (2)].date).hour = - (pd_yyvsp[(1) - (2)].date).hour;
	    (pd_yyvsp[(1) - (2)].date).day = - (pd_yyvsp[(1) - (2)].date).day;
	    (pd_yyvsp[(1) - (2)].date).month = - (pd_yyvsp[(1) - (2)].date).month;
	    (pd_yyvsp[(1) - (2)].date).year = - (pd_yyvsp[(1) - (2)].date).year;
	    (pd_yyval.date) = (pd_yyvsp[(1) - (2)].date);
	  }
    break;

  case 33:
#line 484 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    if (pd_date_union (&(pd_yyval.date), &(pd_yyvsp[(1) - (1)].date)))
	      YYERROR;
	  }
    break;

  case 34:
#line 490 "parsedate.y"
    {
	    if (pd_date_union (&(pd_yyvsp[(1) - (2)].date), &(pd_yyvsp[(2) - (2)].date)))
	      YYERROR;
	    (pd_yyval.date) = (pd_yyvsp[(1) - (2)].date);
	  }
    break;

  case 35:
#line 498 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_YEAR ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 36:
#line 503 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_YEAR ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 37:
#line 508 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_YEAR ((pd_yyval.date), (pd_yyvsp[(1) - (1)].number));
	  }
    break;

  case 38:
#line 513 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MONTH ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 39:
#line 518 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MONTH ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 40:
#line 523 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MONTH ((pd_yyval.date), (pd_yyvsp[(1) - (1)].number));
	  }
    break;

  case 41:
#line 528 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_DAY ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 42:
#line 533 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_DAY ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 43:
#line 538 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_DAY ((pd_yyval.date), (pd_yyvsp[(1) - (1)].number));
	  }
    break;

  case 44:
#line 543 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_HOUR ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 45:
#line 548 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_HOUR ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 46:
#line 553 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_HOUR ((pd_yyval.date), (pd_yyvsp[(1) - (1)].number));
	  }
    break;

  case 47:
#line 558 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MINUTE ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 48:
#line 563 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MINUTE ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 49:
#line 568 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_MINUTE ((pd_yyval.date), (pd_yyvsp[(1) - (1)].number));
	  }
    break;

  case 50:
#line 573 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_SECOND ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 51:
#line 578 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_SECOND ((pd_yyval.date), (pd_yyvsp[(1) - (2)].number) * (pd_yyvsp[(2) - (2)].number));
	  }
    break;

  case 52:
#line 583 "parsedate.y"
    {
	    DATE_INIT ((pd_yyval.date));
	    SET_SECOND ((pd_yyval.date), (pd_yyvsp[(1) - (1)].number));
	  }
    break;

  case 53:
#line 590 "parsedate.y"
    {
	    (pd_yyval.meridian) = MER24;
	  }
    break;

  case 54:
#line 594 "parsedate.y"
    {
	    (pd_yyval.meridian) = (pd_yyvsp[(1) - (1)].meridian);
	  }
    break;


/* Line 1267 of yacc.c.  */
#line 2102 "parsedate.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", pd_yyr1[pd_yyn], &pd_yyval, &pd_yyloc);

  YYPOPSTACK (pd_yylen);
  pd_yylen = 0;
  YY_STACK_PRINT (pd_yyss, pd_yyssp);

  *++pd_yyvsp = pd_yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  pd_yyn = pd_yyr1[pd_yyn];

  pd_yystate = pd_yypgoto[pd_yyn - YYNTOKENS] + *pd_yyssp;
  if (0 <= pd_yystate && pd_yystate <= YYLAST && pd_yycheck[pd_yystate] == *pd_yyssp)
    pd_yystate = pd_yytable[pd_yystate];
  else
    pd_yystate = pd_yydefgoto[pd_yyn - YYNTOKENS];

  goto pd_yynewstate;


/*------------------------------------.
| pd_yyerrlab -- here on detecting error |
`------------------------------------*/
pd_yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!pd_yyerrstatus)
    {
      ++pd_yynerrs;
#if ! YYERROR_VERBOSE
      pd_yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T pd_yysize = pd_yysyntax_error (0, pd_yystate, pd_yychar);
	if (pd_yymsg_alloc < pd_yysize && pd_yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T pd_yyalloc = 2 * pd_yysize;
	    if (! (pd_yysize <= pd_yyalloc && pd_yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      pd_yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (pd_yymsg != pd_yymsgbuf)
	      YYSTACK_FREE (pd_yymsg);
	    pd_yymsg = (char *) YYSTACK_ALLOC (pd_yyalloc);
	    if (pd_yymsg)
	      pd_yymsg_alloc = pd_yyalloc;
	    else
	      {
		pd_yymsg = pd_yymsgbuf;
		pd_yymsg_alloc = sizeof pd_yymsgbuf;
	      }
	  }

	if (0 < pd_yysize && pd_yysize <= pd_yymsg_alloc)
	  {
	    (void) pd_yysyntax_error (pd_yymsg, pd_yystate, pd_yychar);
	    pd_yyerror (pd_yymsg);
	  }
	else
	  {
	    pd_yyerror (YY_("syntax error"));
	    if (pd_yysize != 0)
	      goto pd_yyexhaustedlab;
	  }
      }
#endif
    }



  if (pd_yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (pd_yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (pd_yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  pd_yydestruct ("Error: discarding",
		      pd_yytoken, &pd_yylval);
	  pd_yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto pd_yyerrlab1;


/*---------------------------------------------------.
| pd_yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
pd_yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label pd_yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto pd_yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (pd_yylen);
  pd_yylen = 0;
  YY_STACK_PRINT (pd_yyss, pd_yyssp);
  pd_yystate = *pd_yyssp;
  goto pd_yyerrlab1;


/*-------------------------------------------------------------.
| pd_yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
pd_yyerrlab1:
  pd_yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      pd_yyn = pd_yypact[pd_yystate];
      if (pd_yyn != YYPACT_NINF)
	{
	  pd_yyn += YYTERROR;
	  if (0 <= pd_yyn && pd_yyn <= YYLAST && pd_yycheck[pd_yyn] == YYTERROR)
	    {
	      pd_yyn = pd_yytable[pd_yyn];
	      if (0 < pd_yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (pd_yyssp == pd_yyss)
	YYABORT;


      pd_yydestruct ("Error: popping",
		  pd_yystos[pd_yystate], pd_yyvsp);
      YYPOPSTACK (1);
      pd_yystate = *pd_yyssp;
      YY_STACK_PRINT (pd_yyss, pd_yyssp);
    }

  if (pd_yyn == YYFINAL)
    YYACCEPT;

  *++pd_yyvsp = pd_yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", pd_yystos[pd_yyn], pd_yyvsp, pd_yylsp);

  pd_yystate = pd_yyn;
  goto pd_yynewstate;


/*-------------------------------------.
| pd_yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
pd_yyacceptlab:
  pd_yyresult = 0;
  goto pd_yyreturn;

/*-----------------------------------.
| pd_yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
pd_yyabortlab:
  pd_yyresult = 1;
  goto pd_yyreturn;

#ifndef pd_yyoverflow
/*-------------------------------------------------.
| pd_yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
pd_yyexhaustedlab:
  pd_yyerror (YY_("memory exhausted"));
  pd_yyresult = 2;
  /* Fall through.  */
#endif

pd_yyreturn:
  if (pd_yychar != YYEOF && pd_yychar != YYEMPTY)
     pd_yydestruct ("Cleanup: discarding lookahead",
		 pd_yytoken, &pd_yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (pd_yylen);
  YY_STACK_PRINT (pd_yyss, pd_yyssp);
  while (pd_yyssp != pd_yyss)
    {
      pd_yydestruct ("Cleanup: popping",
		  pd_yystos[*pd_yyssp], pd_yyvsp);
      YYPOPSTACK (1);
    }
#ifndef pd_yyoverflow
  if (pd_yyss != pd_yyssa)
    YYSTACK_FREE (pd_yyss);
#endif
#if YYERROR_VERBOSE
  if (pd_yymsg != pd_yymsgbuf)
    YYSTACK_FREE (pd_yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (pd_yyresult);
}


#line 599 "parsedate.y"


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
pd_yyerror (char *s MU_ARG_UNUSED)
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
      pd_yylval.meridian = MERam;
      return T_MERIDIAN;
    }
  if (strcmp (buff, "pm") == 0 || strcmp (buff, "p.m.") == 0)
    {
      pd_yylval.meridian = MERpm;
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
	      pd_yylval.number = tp->value;
	      return tp->type;
	    }
	}
      else if (strcmp (buff, tp->name) == 0)
	{
	  pd_yylval.number = tp->value;
	  return tp->type;
	}
    }

  for (tp = tz_tab; tp->name; tp++)
    if (strcmp (buff, tp->name) == 0)
      {
	pd_yylval.number = tp->value;
	return tp->type;
      }

  if (strcmp (buff, "dst") == 0)
    return T_DST;

  for (tp = units_tab; tp->name; tp++)
    if (strcmp (buff, tp->name) == 0)
      {
	pd_yylval.number = tp->value;
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
	    pd_yylval.number = tp->value;
	    return tp->type;
	  }
      buff[i] = 's';		/* Put back for "this" in other_tab. */
    }

  for (tp = other_tab; tp->name; tp++)
    if (strcmp (buff, tp->name) == 0)
      {
	pd_yylval.number = tp->value;
	return tp->type;
      }

  /* Military timezones. */
  if (buff[1] == '\0' && ISALPHA ((unsigned char) *buff))
    {
      for (tp = mil_tz_tab; tp->name; tp++)
	if (strcmp (buff, tp->name) == 0)
	  {
	    pd_yylval.number = tp->value;
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
	  pd_yylval.number = tp->value;
	  return tp->type;
	}

  return T_ID;
}

static int
pd_yylex ()
{
  register unsigned char c;
  register char *p;
  char buff[20];
  int count;
  int sign;

  for (;;)
    {
      while (ISSPACE ((unsigned char) *pd_yyinput))
	pd_yyinput++;

      if (ISDIGIT (c = *pd_yyinput) || c == '-' || c == '+')
	{
	  if (c == '-' || c == '+')
	    {
	      sign = c == '-' ? -1 : 1;
	      if (!ISDIGIT (*++pd_yyinput))
		/* skip the '-' sign */
		continue;
	    }
	  else
	    sign = 0;
	  for (pd_yylval.number = 0; ISDIGIT (c = *pd_yyinput++);)
	    pd_yylval.number = 10 * pd_yylval.number + c - '0';
	  pd_yyinput--;
	  if (sign < 0)
	    pd_yylval.number = -pd_yylval.number;
	  return sign ? T_SNUMBER : T_UNUMBER;
	}
      if (ISALPHA (c))
	{
	  for (p = buff; (c = *pd_yyinput++, ISALPHA (c)) || c == '.';)
	    if (p < &buff[sizeof buff - 1])
	      *p++ = c;
	  *p = '\0';
	  pd_yyinput--;
	  return sym_lookup (buff);
	}
      if (c != '(')
	return *pd_yyinput++;
      count = 0;
      do
	{
	  c = *pd_yyinput++;
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

  pd_yyinput = p;
  start = now ? *now : time ((time_t *) NULL);
  tmp = localtime (&start);
  if (!tmp)
    return -1;

  memset (&tm, 0, sizeof tm);
  tm.tm_isdst = tmp->tm_isdst;

  if (pd_yyparse ())
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
    pd_yydebug++;
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

