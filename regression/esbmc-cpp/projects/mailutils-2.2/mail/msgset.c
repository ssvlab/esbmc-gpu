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

/* All symbols defined below should begin with yy or YY, to avoid
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
   enum yytokentype {
     TYPE = 258,
     IDENT = 259,
     REGEXP = 260,
     HEADER = 261,
     BODY = 262,
     NUMBER = 263
   };
#endif
/* Tokens.  */
#define TYPE 258
#define IDENT 259
#define REGEXP 260
#define HEADER 261
#define BODY 262
#define NUMBER 263




/* Copy the first part of user declarations.  */
#line 20 "msgset.y"

#include "mail.h"

#include <stdio.h>
#include <stdlib.h>

#include <xalloc.h>

/* Defined in <limits.h> on some systems, but redefined in <regex.h>
   if we are using GNU's regex. So, undef it to avoid duplicate definition
   warnings. */

#ifdef RE_DUP_MAX
# undef RE_DUP_MAX
#endif
#include <regex.h>

struct header_data
{
  char *header;
  char *expr;
};

static msgset_t *msgset_select (int (*sel) (mu_message_t, void *),
				     void *closure, int rev,
				     unsigned int max_matches);
static int select_header (mu_message_t msg, void *closure);
static int select_body (mu_message_t msg, void *closure);
static int select_type (mu_message_t msg, void *closure);
static int select_sender (mu_message_t msg, void *closure);
static int select_deleted (mu_message_t msg, void *closure);
static int check_set (msgset_t **pset);
 
int yyerror (const char *);
int yylex  (void);

static int msgset_flags = MSG_NODELETED;
static msgset_t *result;


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
#line 60 "msgset.y"
{
  char *string;
  int number;
  int type;
  msgset_t *mset;
}
/* Line 187 of yacc.c.  */
#line 159 "msgset.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 172 "msgset.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
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

#if ! defined yyoverflow || YYERROR_VERBOSE

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
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
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
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
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
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  29
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   87

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  23
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  11
/* YYNRULES -- Number of rules.  */
#define YYNRULES  34
/* YYNRULES -- Number of states.  */
#define YYNSTATES  48

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   263

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    18,     2,     2,    11,     2,     2,     2,
      21,    22,    12,    14,    15,    13,     9,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    19,     2,    20,    10,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    16,     2,    17,     2,     2,     2,     2,
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
       5,     6,     7,     8
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint8 yyprhs[] =
{
       0,     0,     3,     4,     6,     8,    10,    12,    14,    16,
      18,    20,    24,    27,    29,    33,    36,    38,    43,    45,
      48,    50,    52,    54,    55,    57,    59,    63,    66,    68,
      72,    76,    78,    83,    85
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      24,     0,    -1,    -1,     9,    -1,    25,    -1,    10,    -1,
      11,    -1,    12,    -1,    13,    -1,    14,    -1,    26,    -1,
      25,    15,    26,    -1,    25,    26,    -1,    27,    -1,    16,
      25,    17,    -1,    18,    26,    -1,    28,    -1,    28,    19,
      30,    20,    -1,    31,    -1,    29,     5,    -1,     7,    -1,
       3,    -1,     4,    -1,    -1,     6,    -1,    31,    -1,    30,
      15,    31,    -1,    30,    31,    -1,    32,    -1,     8,    13,
      32,    -1,     8,    13,    12,    -1,    33,    -1,    33,    19,
      30,    20,    -1,     8,    -1,    21,    30,    22,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint8 yyrline[] =
{
       0,    76,    76,    79,    83,    87,    91,    95,    99,   103,
     109,   110,   114,   120,   126,   130,   136,   137,   143,   146,
     156,   160,   169,   177,   180,   186,   187,   191,   197,   198,
     212,   218,   219,   227,   231
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "TYPE", "IDENT", "REGEXP", "HEADER",
  "BODY", "NUMBER", "'.'", "'^'", "'$'", "'*'", "'-'", "'+'", "','", "'{'",
  "'}'", "'!'", "'['", "']'", "'('", "')'", "$accept", "input", "msgset",
  "msgexpr", "msgspec", "msg", "header", "rangeset", "range", "number",
  "partno", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,    46,
      94,    36,    42,    45,    43,    44,   123,   125,    33,    91,
      93,    40,    41
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    23,    24,    24,    24,    24,    24,    24,    24,    24,
      25,    25,    25,    26,    26,    26,    27,    27,    27,    28,
      28,    28,    28,    29,    29,    30,    30,    30,    31,    31,
      31,    32,    32,    33,    33
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     2,     1,     3,     2,     1,     4,     1,     2,
       1,     1,     1,     0,     1,     1,     3,     2,     1,     3,
       3,     1,     4,     1,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,    21,    22,    24,    20,    33,     3,     5,     6,     7,
       8,     9,    23,    23,     0,     0,     4,    10,    13,    16,
       0,    18,    28,    31,     0,    23,    15,     0,    25,     1,
      23,    12,     0,    19,     0,    33,    30,    29,    14,     0,
      34,    27,    11,     0,     0,    26,    17,    32
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,    15,    16,    17,    18,    19,    20,    27,    21,    22,
      23
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -15
static const yytype_int8 yypact[] =
{
      -2,   -15,   -15,   -15,   -15,     4,   -15,   -15,   -15,   -15,
     -15,   -15,    50,    50,    16,    26,    28,   -15,   -15,    19,
      34,   -15,   -15,    21,    66,    44,   -15,    62,   -15,   -15,
      50,   -15,    16,   -15,    16,   -15,   -15,   -15,   -15,    16,
     -15,   -15,   -15,     7,    65,   -15,   -15,   -15
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -15,   -15,    29,    51,   -15,   -15,   -15,   -11,   -14,    18,
     -15
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -24
static const yytype_int8 yytable[] =
{
      28,     1,     2,   -23,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    41,    12,     5,    13,    24,    28,    14,
      28,    43,    39,    44,     5,    45,    29,    46,    14,    41,
      41,     1,     2,   -23,     3,     4,     5,    14,    32,    33,
      34,    25,    37,    30,    12,     0,    13,     1,     2,    14,
       3,     4,     5,     1,     2,     0,     3,     4,     5,    30,
      12,    38,    13,     0,    26,    14,    12,    31,    13,     0,
       5,    14,     0,     5,    35,     0,    31,    39,    36,     0,
      39,    42,     0,    14,    40,    47,    14,    14
};

static const yytype_int8 yycheck[] =
{
      14,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    27,    16,     8,    18,    13,    32,    21,
      34,    32,    15,    34,     8,    39,     0,    20,    21,    43,
      44,     3,     4,     5,     6,     7,     8,    21,    19,     5,
      19,    12,    24,    15,    16,    -1,    18,     3,     4,    21,
       6,     7,     8,     3,     4,    -1,     6,     7,     8,    15,
      16,    17,    18,    -1,    13,    21,    16,    16,    18,    -1,
       8,    21,    -1,     8,     8,    -1,    25,    15,    12,    -1,
      15,    30,    -1,    21,    22,    20,    21,    21
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    16,    18,    21,    24,    25,    26,    27,    28,
      29,    31,    32,    33,    13,    25,    26,    30,    31,     0,
      15,    26,    19,     5,    19,     8,    12,    32,    17,    15,
      22,    31,    26,    30,    30,    31,    20,    20
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
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


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
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
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
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
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
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

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
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
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
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
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 76 "msgset.y"
    {
	     result = msgset_make_1 (get_cursor ());
	   }
    break;

  case 3:
#line 80 "msgset.y"
    {
	     result = msgset_make_1 (get_cursor ());
	   }
    break;

  case 4:
#line 84 "msgset.y"
    {
	     result = (yyvsp[(1) - (1)].mset);
	   }
    break;

  case 5:
#line 88 "msgset.y"
    {
	     result = msgset_select (select_deleted, NULL, 0, 1);
	   }
    break;

  case 6:
#line 92 "msgset.y"
    {
	     result = msgset_select (select_deleted, NULL, 1, 1);
	   }
    break;

  case 7:
#line 96 "msgset.y"
    {
	     result = msgset_select (select_deleted, NULL, 0, total);
	   }
    break;

  case 8:
#line 100 "msgset.y"
    {
	     result = msgset_select (select_deleted, NULL, 1, 1);
	   }
    break;

  case 9:
#line 104 "msgset.y"
    {
	     result = msgset_select (select_deleted, NULL, 0, 1);
	   }
    break;

  case 11:
#line 111 "msgset.y"
    {
	     (yyval.mset) = msgset_append ((yyvsp[(1) - (3)].mset), (yyvsp[(3) - (3)].mset));
	   }
    break;

  case 12:
#line 115 "msgset.y"
    {
	     (yyval.mset) = msgset_append ((yyvsp[(1) - (2)].mset), (yyvsp[(2) - (2)].mset));
	   }
    break;

  case 13:
#line 121 "msgset.y"
    {
	     (yyval.mset) = (yyvsp[(1) - (1)].mset);
	     if (check_set (&(yyval.mset)))
	       YYABORT;
	   }
    break;

  case 14:
#line 127 "msgset.y"
    {
	     (yyval.mset) = (yyvsp[(2) - (3)].mset);
	   }
    break;

  case 15:
#line 131 "msgset.y"
    {
	     (yyval.mset) = msgset_negate ((yyvsp[(2) - (2)].mset));
	   }
    break;

  case 17:
#line 138 "msgset.y"
    {
	     (yyval.mset) = msgset_expand ((yyvsp[(1) - (4)].mset), (yyvsp[(3) - (4)].mset));
	     msgset_free ((yyvsp[(1) - (4)].mset));
	     msgset_free ((yyvsp[(3) - (4)].mset));
	   }
    break;

  case 19:
#line 147 "msgset.y"
    {
	     struct header_data hd;
	     hd.header = (yyvsp[(1) - (2)].string);
	     hd.expr   = (yyvsp[(2) - (2)].string);
	     (yyval.mset) = msgset_select (select_header, &hd, 0, 0);
	     if ((yyvsp[(1) - (2)].string))
	       free ((yyvsp[(1) - (2)].string));
	     free ((yyvsp[(2) - (2)].string));
	   }
    break;

  case 20:
#line 157 "msgset.y"
    {
	     (yyval.mset) = msgset_select (select_body, (yyvsp[(1) - (1)].string), 0, 0);
	   }
    break;

  case 21:
#line 161 "msgset.y"
    {
	     if (strchr ("dnorTtu", (yyvsp[(1) - (1)].type)) == NULL)
	       {
		 yyerror (_("unknown message type"));
		 YYERROR;
	       }
	     (yyval.mset) = msgset_select (select_type, (void *)&(yyvsp[(1) - (1)].type), 0, 0);
	   }
    break;

  case 22:
#line 170 "msgset.y"
    {
	     (yyval.mset) = msgset_select (select_sender, (void *)(yyvsp[(1) - (1)].string), 0, 0);
	     free ((yyvsp[(1) - (1)].string));
	   }
    break;

  case 23:
#line 177 "msgset.y"
    {
	     (yyval.string) = NULL;
	   }
    break;

  case 24:
#line 181 "msgset.y"
    {
	     (yyval.string) = (yyvsp[(1) - (1)].string);
	   }
    break;

  case 26:
#line 188 "msgset.y"
    {
	     (yyval.mset) = msgset_append ((yyvsp[(1) - (3)].mset), (yyvsp[(3) - (3)].mset));
	   }
    break;

  case 27:
#line 192 "msgset.y"
    {
	     (yyval.mset) = msgset_append ((yyvsp[(1) - (2)].mset), (yyvsp[(2) - (2)].mset));
	   }
    break;

  case 29:
#line 199 "msgset.y"
    {
	     if ((yyvsp[(3) - (3)].mset)->npart == 1)
	       {
		 (yyval.mset) = msgset_range ((yyvsp[(1) - (3)].number), (yyvsp[(3) - (3)].mset)->msg_part[0]);
	       }
	     else
	       {
		 (yyval.mset) = msgset_range ((yyvsp[(1) - (3)].number), (yyvsp[(3) - (3)].mset)->msg_part[0]-1);
		 if (!(yyval.mset))
		   YYERROR;
		 msgset_append ((yyval.mset), (yyvsp[(3) - (3)].mset));
	       }
	   }
    break;

  case 30:
#line 213 "msgset.y"
    {
	     (yyval.mset) = msgset_range ((yyvsp[(1) - (3)].number), total);
	   }
    break;

  case 32:
#line 220 "msgset.y"
    {
	     (yyval.mset) = msgset_expand ((yyvsp[(1) - (4)].mset), (yyvsp[(3) - (4)].mset));
	     msgset_free ((yyvsp[(1) - (4)].mset));
	     msgset_free ((yyvsp[(3) - (4)].mset));
	   }
    break;

  case 33:
#line 228 "msgset.y"
    {
	     (yyval.mset) = msgset_make_1 ((yyvsp[(1) - (1)].number));
	   }
    break;

  case 34:
#line 232 "msgset.y"
    {
	     (yyval.mset) = (yyvsp[(2) - (3)].mset);
	   }
    break;


/* Line 1267 of yacc.c.  */
#line 1624 "msgset.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 236 "msgset.y"


static int xargc;
static char **xargv;
static int cur_ind;
static char *cur_p;

int
yyerror (const char *s)
{
  fprintf (stderr, "%s: ", xargv[0]);
  fprintf (stderr, "%s", s);
  if (!cur_p)
    fprintf (stderr, _(" near end"));
  else if (*cur_p == 0)
    {
      int i =  (*cur_p == 0) ? cur_ind + 1 : cur_ind;
      if (i == xargc)
	fprintf (stderr, _(" near end"));
      else
	fprintf (stderr, _(" near %s"), xargv[i]);
    }
  else
    fprintf (stderr, _(" near %s"), cur_p);
  fprintf (stderr, "\n");
  return 0;
}

int
yylex()
{
  if (cur_ind == xargc)
    return 0;
  if (!cur_p)
    cur_p = xargv[cur_ind];
  if (*cur_p == 0)
    {
      cur_ind++;
      cur_p = NULL;
      return yylex ();
    }

  if (mu_isdigit (*cur_p))
    {
      yylval.number = strtoul (cur_p, &cur_p, 10);
      return NUMBER;
    }

  if (mu_isalpha (*cur_p))
    {
      char *p = cur_p;
      int len;

      while (*cur_p && *cur_p != ',' && *cur_p != ':') 
	cur_p++;
      len = cur_p - p + 1;
      yylval.string = xmalloc (len);
      memcpy (yylval.string, p, len-1);
      yylval.string[len-1] = 0;
      if (*cur_p == ':')
	{
	  ++cur_p;
	  return HEADER;
	}
      return IDENT;
    }

  if (*cur_p == '/')
    {
      char *p = ++cur_p;
      int len;

      while (*cur_p && *cur_p != '/')
	cur_p++;
      len = cur_p - p + 1;
      cur_p++;
      yylval.string = xmalloc (len);
      memcpy (yylval.string, p, len-1);
      yylval.string[len-1] = 0;
      return REGEXP;
    }

  if (*cur_p == ':')
    {
      cur_p++;
      if (*cur_p == '/')
	{
	  char *p = ++cur_p;
	  int len;

	  while (*cur_p && *cur_p != '/')
	    cur_p++;
	  len = cur_p - p + 1;
	  cur_p++;
	  yylval.string = xmalloc (len);
	  memcpy (yylval.string, p, len-1);
	  yylval.string[len-1] = 0;
	  return BODY;
	}
      yylval.type = *cur_p++;
      return TYPE;
    }

  return *cur_p++;
}

int
msgset_parse (const int argc, char **argv, int flags, msgset_t **mset)
{
  int rc;
  xargc = argc;
  xargv = argv;
  msgset_flags = flags;
  cur_ind = 1;
  cur_p = NULL;
  result = NULL;
  rc = yyparse ();
  if (rc == 0)
    {
      if (result == NULL)
	{
	  util_noapp ();
	  rc = 1;
	}
      else
	*mset = result;
    }
  return rc;
}

void
msgset_free (msgset_t *msg_set)
{
  msgset_t *next;

  if (!msg_set)
    return;
  while (msg_set)
    {
      next = msg_set->next;
      if (msg_set->msg_part)
	free (msg_set->msg_part);
      free (msg_set);
      msg_set = next;
    }
}

size_t
msgset_count (msgset_t *set)
{
  size_t count = 0;
  for (; set; set = set->next)
    count++;
  return count;
}

/* Create a message set consisting of a single msg_num and no subparts */
msgset_t *
msgset_make_1 (size_t number)
{
  msgset_t *mp;

  if (number == 0)
    return NULL;
  mp = xmalloc (sizeof (*mp));
  mp->next = NULL;
  mp->npart = 1;
  mp->msg_part = xmalloc (sizeof mp->msg_part[0]);
  mp->msg_part[0] = number;
  return mp;
}

msgset_t *
msgset_dup (const msgset_t *set)
{
  msgset_t *mp;
  mp = xmalloc (sizeof (*mp));
  mp->next = NULL;
  mp->npart = set->npart;
  mp->msg_part = xcalloc (mp->npart, sizeof mp->msg_part[0]);
  memcpy (mp->msg_part, set->msg_part, mp->npart * sizeof mp->msg_part[0]);
  return mp;
}

msgset_t *
msgset_append (msgset_t *one, msgset_t *two)
{
  msgset_t *last;

  if (!one)
    return two;
  for (last = one; last->next; last = last->next)
    ;
  last->next = two;
  return one;
}

int
msgset_member (msgset_t *set, size_t n)
{
  for (; set; set = set->next)
    if (set->msg_part[0] == n)
      return 1;
  return 0;
}

msgset_t *
msgset_negate (msgset_t *set)
{
  size_t i;
  msgset_t *first = NULL, *last = NULL;

  for (i = 1; i <= total; i++)
    {
      if (!msgset_member (set, i))
	{
	  msgset_t *mp = msgset_make_1 (i);
	  if (!first)
	    first = mp;
	  else
	    last->next = mp;
	  last = mp;
	}
    }
  return first;
}

msgset_t *
msgset_range (int low, int high)
{
  int i;
  msgset_t *mp, *first = NULL, *last = NULL;

  if (low == high)
    return msgset_make_1 (low);

  if (low >= high)
    {
      yyerror (_("range error"));
      return NULL;
    }

  for (i = 0; low <= high; i++, low++)
    {
      mp = msgset_make_1 (low);
      if (!first)
	first = mp;
      else
	last->next = mp;
      last = mp;
    }
  return first;
}

msgset_t *
msgset_expand (msgset_t *set, msgset_t *expand_by)
{
  msgset_t *i, *j;
  msgset_t *first = NULL, *last = NULL, *mp;

  for (i = set; i; i = i->next)
    for (j = expand_by; j; j = j->next)
      {
	mp = xmalloc (sizeof *mp);
	mp->next = NULL;
	mp->npart = i->npart + j->npart;
	mp->msg_part = xcalloc (mp->npart, sizeof mp->msg_part[0]);
	memcpy (mp->msg_part, i->msg_part, i->npart * sizeof i->msg_part[0]);
	memcpy (mp->msg_part + i->npart, j->msg_part,
		j->npart * sizeof j->msg_part[0]);

	if (!first)
	  first = mp;
	else
	  last->next = mp;
	last = mp;
      }
  return first;
}

msgset_t *
msgset_select (int (*sel) (mu_message_t, void *), void *closure, int rev,
	       unsigned int max_matches)
{
  size_t i, match_count = 0;
  msgset_t *first = NULL, *last = NULL, *mp;
  mu_message_t msg = NULL;

  if (max_matches == 0)
    max_matches = total;

  if (rev)
    {
      for (i = total; i > 0; i--)
	{
	  mu_mailbox_get_message (mbox, i, &msg);
	  if ((*sel)(msg, closure))
	    {
	      mp = msgset_make_1 (i);
	      if (!first)
		first = mp;
	      else
		last->next = mp;
	      last = mp;
	      if (++match_count == max_matches)
		break;
	    }
	}
    }
  else
    {
      for (i = 1; i <= total; i++)
	{
	  mu_mailbox_get_message (mbox, i, &msg);
	  if ((*sel)(msg, closure))
	    {
	      mp = msgset_make_1 (i);
	      if (!first)
		first = mp;
	      else
		last->next = mp;
	      last = mp;
	      if (++match_count == max_matches)
		break;
	    }
	}
    }
  return first;
}

int
select_header (mu_message_t msg, void *closure)
{
  struct header_data *hd = (struct header_data *)closure;
  mu_header_t hdr;
  char *contents;
  const char *header = hd->header ? hd->header : MU_HEADER_SUBJECT;

  mu_message_get_header (msg, &hdr);
  if (mu_header_aget_value (hdr, header, &contents) == 0)
    {
      if (mailvar_get (NULL, "regex", mailvar_type_boolean, 0) == 0)
	{
	  /* Match string against the extended regular expression(ignoring
	     case) in pattern, treating errors as no match.
	     Return 1 for match, 0 for no match.
	  */
          regex_t re;
          int status;
	  int flags = REG_EXTENDED;

	  if (mu_islower (header[0]))
	    flags |= REG_ICASE;
          if (regcomp (&re, hd->expr, flags) != 0)
	    {
	      free (contents);
	      return 0;
	    }
          status = regexec (&re, contents, 0, NULL, 0);
          free (contents);
	  regfree (&re);
          return status == 0;
	}
      else
	{
	  int rc;
	  mu_strupper (contents);
	  rc = strstr (contents, hd->expr) != NULL;
	  free (contents);
	  return rc;
	}
    }
  return 0;
}

int
select_body (mu_message_t msg, void *closure)
{
  char *expr = closure;
  int noregex = mailvar_get (NULL, "regex", mailvar_type_boolean, 0);
  regex_t re;
  int status;
  mu_body_t body = NULL;
  mu_stream_t stream = NULL;
  size_t size = 0, lines = 0;
  char buffer[128];
  size_t n = 0;
  off_t offset = 0;

  if (noregex)
    mu_strupper (expr);
  else if (regcomp (&re, expr, REG_EXTENDED | REG_ICASE) != 0)
    return 0;

  mu_message_get_body (msg, &body);
  mu_body_size (body, &size);
  mu_body_lines (body, &lines);
  mu_body_get_stream (body, &stream);
  status = 0;
  while (status == 0
	 && mu_stream_readline (stream, buffer, sizeof(buffer)-1, offset, &n) == 0
	 && n > 0)
    {
      offset += n;
      if (noregex)
	{
	  mu_strupper (buffer);
	  status = strstr (buffer, expr) != NULL;
	}
      else
	status = regexec (&re, buffer, 0, NULL, 0);
    }

  if (!noregex)
    regfree (&re);

  return status;
}

int
select_sender (mu_message_t msg MU_ARG_UNUSED, void *closure MU_ARG_UNUSED)
{
  /* char *sender = (char*) closure; */
  /* FIXME: all messages from sender argv[i] */
  /* Annoying we can use mu_address_create() for that
     but to compare against what? The email ?  */
  return 0;
}

int
select_type (mu_message_t msg, void *closure)
{
  int type = *(int*) closure;
  mu_attribute_t attr= NULL;

  mu_message_get_attribute (msg, &attr);

  switch (type)
    {
    case 'd':
      return mu_attribute_is_deleted (attr);
    case 'n':
      return mu_attribute_is_recent (attr);
    case 'o':
      return mu_attribute_is_seen (attr);
    case 'r':
      return mu_attribute_is_read (attr);
    case 'u':
      return !mu_attribute_is_read (attr);
    case 't':
      return mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_TAGGED);
    case 'T':
      return !mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_TAGGED);
    }
  return 0;
}

int
select_deleted (mu_message_t msg, void *closure MU_ARG_UNUSED)
{
  mu_attribute_t attr= NULL;
  int rc;

  mu_message_get_attribute (msg, &attr);
  rc = mu_attribute_is_deleted (attr);
  return strcmp (xargv[0], "undelete") == 0 ? rc : !rc;
}

int
check_set (msgset_t **pset)
{
  int flags = msgset_flags;
  int rc = 0;
  
  if (msgset_count (*pset) == 1)
    flags ^= MSG_SILENT;
  if (flags & MSG_NODELETED)
    {
      msgset_t *p = *pset, *prev = NULL;
      msgset_t *delset = NULL;

      while (p)
	{
	  msgset_t *next = p->next;
	  if (util_isdeleted (p->msg_part[0]))
	    {
	      if ((flags & MSG_SILENT) && (prev || next))
		{
		  /* Mark subset as deleted */
		  p->next = delset;
		  delset = p;
		  /* Remove it from the set */
		  if (prev)
		    prev->next = next;
		  else
		    *pset = next;
		}
	      else
		{
		  util_error (_("%lu: Inappropriate message (has been deleted)"),
			      (unsigned long) p->msg_part[0]);
		  /* Delete entire set */
		  delset = *pset;
		  *pset = NULL;
		  rc = 1;
		  break;
		}
	    }
	  else
	    prev = p;
	  p = next;
	}

      if (delset)
	msgset_free (delset);

      if (!*pset)
	rc = 1;
    }

  return rc;
}

#if 0
void
msgset_print (msgset_t *mset)
{
  int i;
  printf ("(");
  printf ("%d .", mset->msg_part[0]);
  for (i = 1; i < mset->npart; i++)
    {
      printf (" %d", mset->msg_part[i]);
    }
  printf (")\n");
}

int
main(int argc, char **argv)
{
  msgset_t *mset = NULL;
  int rc = msgset_parse (argc, argv, &mset);

  for (; mset; mset = mset->next)
    msgset_print (mset);
  return 0;
}
#endif

