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

/* All symbols defined below should begin with fmt_yy or YY, to avoid
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
   enum fmt_yytokentype {
     NUMBER = 258,
     STRING = 259,
     FUNCTION = 260,
     IF = 261,
     ELIF = 262,
     ELSE = 263,
     FI = 264,
     OBRACE = 265,
     CBRACE = 266,
     OCURLY = 267,
     CCURLY = 268,
     FMTSPEC = 269,
     BOGUS = 270
   };
#endif
/* Tokens.  */
#define NUMBER 258
#define STRING 259
#define FUNCTION 260
#define IF 261
#define ELIF 262
#define ELSE 263
#define FI 264
#define OBRACE 265
#define CBRACE 266
#define OCURLY 267
#define CCURLY 268
#define FMTSPEC 269
#define BOGUS 270




/* Copy the first part of user declarations.  */
#line 1 "mh_fmtgram.y"

/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2004, 2007, 2009, 2010 Free
   Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#include <mh.h>
#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
#include <obstack.h>

int fmt_yyerror (const char *s);
int fmt_yylex ();
 
static mh_format_t format;     /* Format structure being built */
static size_t pc;              /* Program counter. Poins to current
				  cell in format.prog */
static struct obstack stack;   /* Temporary token storage */

#define FORMAT_INC 64          /* Increase format.prog by that many
				  cells each time pc reaches
				  format.progsize */

static size_t mh_code_op (mh_opcode_t op);
static size_t mh_code_string (char *string);
static size_t mh_code_number (int num);
static size_t mh_code_builtin (mh_builtin_t *bp, int argtype);
static void branch_fixup (size_t pc, size_t tgt); 

  /* Lexical tie-ins */
static int in_escape;       /* Set when inside an escape sequence */
static int want_function;   /* Set when expecting function name */
static int want_arg;        /* Expecting function argument */


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
#line 50 "mh_fmtgram.y"
{
  char *str;
  int num;
  int type;
  struct {
    size_t cond;
    size_t end;
  } elif_list;
  size_t pc;
  mh_builtin_t *builtin;
}
/* Line 187 of yacc.c.  */
#line 187 "mh_fmtgram.c"
	YYSTYPE;
# define fmt_yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 200 "mh_fmtgram.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 fmt_yytype_uint8;
#else
typedef unsigned char fmt_yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 fmt_yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char fmt_yytype_int8;
#else
typedef short int fmt_yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 fmt_yytype_uint16;
#else
typedef unsigned short int fmt_yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 fmt_yytype_int16;
#else
typedef short int fmt_yytype_int16;
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

#if ! defined fmt_yyoverflow || YYERROR_VERBOSE

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
#endif /* ! defined fmt_yyoverflow || YYERROR_VERBOSE */


#if (! defined fmt_yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union fmt_yyalloc
{
  fmt_yytype_int16 fmt_yyss;
  YYSTYPE fmt_yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union fmt_yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (fmt_yytype_int16) + sizeof (YYSTYPE)) \
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
	  YYSIZE_T fmt_yyi;				\
	  for (fmt_yyi = 0; fmt_yyi < (Count); fmt_yyi++)	\
	    (To)[fmt_yyi] = (From)[fmt_yyi];		\
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
	YYSIZE_T fmt_yynewbytes;						\
	YYCOPY (&fmt_yyptr->Stack, Stack, fmt_yysize);				\
	Stack = &fmt_yyptr->Stack;						\
	fmt_yynewbytes = fmt_yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	fmt_yyptr += fmt_yynewbytes / sizeof (*fmt_yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  16
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   64

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  16
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  28
/* YYNRULES -- Number of rules.  */
#define YYNRULES  42
/* YYNRULES -- Number of states.  */
#define YYNSTATES  56

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   270

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? fmt_yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const fmt_yytype_uint8 fmt_yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
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
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const fmt_yytype_uint8 fmt_yyprhs[] =
{
       0,     0,     3,     5,     7,    10,    12,    14,    16,    18,
      20,    22,    24,    26,    31,    33,    35,    36,    37,    45,
      46,    48,    50,    52,    53,    55,    57,    65,    66,    68,
      70,    72,    74,    75,    77,    79,    81,    82,    85,    89,
      95,    96,    99
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const fmt_yytype_int8 fmt_yyrhs[] =
{
      17,     0,    -1,    18,    -1,    19,    -1,    18,    19,    -1,
      20,    -1,    21,    -1,    22,    -1,     4,    -1,     3,    -1,
      23,    -1,    26,    -1,    32,    -1,    29,    12,     4,    13,
      -1,    10,    -1,    11,    -1,    -1,    -1,    29,    24,    27,
      30,    28,    31,    25,    -1,    -1,    14,    -1,     5,    -1,
       4,    -1,    -1,    21,    -1,    22,    -1,    34,    38,    33,
      37,    40,    42,    35,    -1,    -1,    18,    -1,     6,    -1,
       9,    -1,     7,    -1,    -1,    39,    -1,    23,    -1,    26,
      -1,    -1,    41,    37,    -1,    36,    38,    33,    -1,    41,
      37,    36,    38,    33,    -1,    -1,    43,    18,    -1,     8,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const fmt_yytype_uint16 fmt_yyrline[] =
{
       0,    75,    75,    81,    82,    85,   110,   111,   117,   122,
     129,   130,   131,   137,   152,   158,   164,   164,   164,   192,
     193,   200,   201,   217,   220,   221,   225,   258,   261,   264,
     270,   282,   290,   296,   307,   308,   312,   316,   324,   330,
     340,   343,   346
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const fmt_yytname[] =
{
  "$end", "error", "$undefined", "NUMBER", "STRING", "FUNCTION", "IF",
  "ELIF", "ELSE", "FI", "OBRACE", "CBRACE", "OCURLY", "CCURLY", "FMTSPEC",
  "BOGUS", "$accept", "input", "list", "pitem", "item", "literal",
  "escape", "component", "obrace", "cbrace", "funcall", "@1", "@2",
  "fmtspec", "function", "argument", "cntl", "zlist", "if", "fi", "elif",
  "end", "cond", "cond_expr", "elif_part", "elif_list", "else_part",
  "else", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const fmt_yytype_uint16 fmt_yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const fmt_yytype_uint8 fmt_yyr1[] =
{
       0,    16,    17,    18,    18,    19,    20,    20,    21,    21,
      22,    22,    22,    23,    24,    25,    27,    28,    26,    29,
      29,    30,    30,    31,    31,    31,    32,    33,    33,    34,
      35,    36,    37,    38,    39,    39,    40,    40,    41,    41,
      42,    42,    43
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const fmt_yytype_uint8 fmt_yyr2[] =
{
       0,     2,     1,     1,     2,     1,     1,     1,     1,     1,
       1,     1,     1,     4,     1,     1,     0,     0,     7,     0,
       1,     1,     1,     0,     1,     1,     7,     0,     1,     1,
       1,     1,     0,     1,     1,     1,     0,     2,     3,     5,
       0,     2,     1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const fmt_yytype_uint8 fmt_yydefact[] =
{
      19,     9,     8,    29,    20,     0,    19,     3,     5,     6,
       7,    10,    11,     0,    12,    19,     1,     4,    14,     0,
      16,    34,    35,    27,    33,     0,     0,    28,    32,    13,
      22,    21,    17,    36,    19,    31,    19,    40,    32,    24,
      25,     0,    27,    42,     0,    19,    37,    15,    18,    38,
      30,    26,    19,    19,    27,    39
};

/* YYDEFGOTO[NTERM-NUM].  */
static const fmt_yytype_int8 fmt_yydefgoto[] =
{
      -1,     5,    27,     7,     8,     9,    10,    11,    20,    48,
      12,    26,    34,    13,    32,    41,    14,    28,    15,    51,
      36,    33,    23,    24,    37,    38,    44,    45
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -38
static const fmt_yytype_int8 fmt_yypact[] =
{
      50,   -38,   -38,   -38,   -38,    14,     4,   -38,   -38,   -38,
     -38,   -38,   -38,    -1,   -38,     1,   -38,   -38,   -38,    12,
     -38,   -38,   -38,    22,   -38,     6,     8,    22,   -38,   -38,
     -38,   -38,   -38,    13,    34,   -38,     1,    19,   -38,   -38,
     -38,    18,    22,   -38,    21,    50,    13,   -38,   -38,   -38,
     -38,   -38,    46,     1,    22,   -38
};

/* YYPGOTO[NTERM-NUM].  */
static const fmt_yytype_int8 fmt_yypgoto[] =
{
     -38,   -38,     2,    -6,   -38,    -3,     9,   -14,   -38,   -38,
     -12,   -38,   -38,   -38,   -38,   -38,   -38,   -37,   -38,   -38,
     -13,    20,   -30,   -38,   -38,   -38,   -38,   -38
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -42
static const fmt_yytype_int8 fmt_yytable[] =
{
      17,    21,     6,    22,    -2,    49,    42,     1,     2,    18,
       3,    19,    30,    31,    16,     4,    25,    55,     4,    29,
      35,    17,    21,    54,    22,     1,     2,    43,     3,    47,
      50,    39,   -19,    53,   -19,     0,     4,     1,     2,    21,
       3,    22,     0,    40,     0,   -23,    17,    52,     4,     1,
       2,     0,     3,     1,     2,   -41,     3,     0,    46,     0,
       4,     0,     0,     0,     4
};

static const fmt_yytype_int8 fmt_yycheck[] =
{
       6,    15,     0,    15,     0,    42,    36,     3,     4,    10,
       6,    12,     4,     5,     0,    14,     4,    54,    14,    13,
       7,    27,    36,    53,    36,     3,     4,     8,     6,    11,
       9,    34,    10,    46,    12,    -1,    14,     3,     4,    53,
       6,    53,    -1,    34,    -1,    11,    52,    45,    14,     3,
       4,    -1,     6,     3,     4,     9,     6,    -1,    38,    -1,
      14,    -1,    -1,    -1,    14
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const fmt_yytype_uint8 fmt_yystos[] =
{
       0,     3,     4,     6,    14,    17,    18,    19,    20,    21,
      22,    23,    26,    29,    32,    34,     0,    19,    10,    12,
      24,    23,    26,    38,    39,     4,    27,    18,    33,    13,
       4,     5,    30,    37,    28,     7,    36,    40,    41,    21,
      22,    31,    38,     8,    42,    43,    37,    11,    25,    33,
       9,    35,    18,    36,    38,    33
};

#define fmt_yyerrok		(fmt_yyerrstatus = 0)
#define fmt_yyclearin	(fmt_yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto fmt_yyacceptlab
#define YYABORT		goto fmt_yyabortlab
#define YYERROR		goto fmt_yyerrorlab


/* Like YYERROR except do call fmt_yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto fmt_yyerrlab

#define YYRECOVERING()  (!!fmt_yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (fmt_yychar == YYEMPTY && fmt_yylen == 1)				\
    {								\
      fmt_yychar = (Token);						\
      fmt_yylval = (Value);						\
      fmt_yytoken = YYTRANSLATE (fmt_yychar);				\
      YYPOPSTACK (1);						\
      goto fmt_yybackup;						\
    }								\
  else								\
    {								\
      fmt_yyerror (YY_("syntax error: cannot back up")); \
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


/* YYLEX -- calling `fmt_yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX fmt_yylex (YYLEX_PARAM)
#else
# define YYLEX fmt_yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (fmt_yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (fmt_yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      fmt_yy_symbol_print (stderr,						  \
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
fmt_yy_symbol_value_print (FILE *fmt_yyoutput, int fmt_yytype, YYSTYPE const * const fmt_yyvaluep)
#else
static void
fmt_yy_symbol_value_print (fmt_yyoutput, fmt_yytype, fmt_yyvaluep)
    FILE *fmt_yyoutput;
    int fmt_yytype;
    YYSTYPE const * const fmt_yyvaluep;
#endif
{
  if (!fmt_yyvaluep)
    return;
# ifdef YYPRINT
  if (fmt_yytype < YYNTOKENS)
    YYPRINT (fmt_yyoutput, fmt_yytoknum[fmt_yytype], *fmt_yyvaluep);
# else
  YYUSE (fmt_yyoutput);
# endif
  switch (fmt_yytype)
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
fmt_yy_symbol_print (FILE *fmt_yyoutput, int fmt_yytype, YYSTYPE const * const fmt_yyvaluep)
#else
static void
fmt_yy_symbol_print (fmt_yyoutput, fmt_yytype, fmt_yyvaluep)
    FILE *fmt_yyoutput;
    int fmt_yytype;
    YYSTYPE const * const fmt_yyvaluep;
#endif
{
  if (fmt_yytype < YYNTOKENS)
    YYFPRINTF (fmt_yyoutput, "token %s (", fmt_yytname[fmt_yytype]);
  else
    YYFPRINTF (fmt_yyoutput, "nterm %s (", fmt_yytname[fmt_yytype]);

  fmt_yy_symbol_value_print (fmt_yyoutput, fmt_yytype, fmt_yyvaluep);
  YYFPRINTF (fmt_yyoutput, ")");
}

/*------------------------------------------------------------------.
| fmt_yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
fmt_yy_stack_print (fmt_yytype_int16 *bottom, fmt_yytype_int16 *top)
#else
static void
fmt_yy_stack_print (bottom, top)
    fmt_yytype_int16 *bottom;
    fmt_yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (fmt_yydebug)							\
    fmt_yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
fmt_yy_reduce_print (YYSTYPE *fmt_yyvsp, int fmt_yyrule)
#else
static void
fmt_yy_reduce_print (fmt_yyvsp, fmt_yyrule)
    YYSTYPE *fmt_yyvsp;
    int fmt_yyrule;
#endif
{
  int fmt_yynrhs = fmt_yyr2[fmt_yyrule];
  int fmt_yyi;
  unsigned long int fmt_yylno = fmt_yyrline[fmt_yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     fmt_yyrule - 1, fmt_yylno);
  /* The symbols being reduced.  */
  for (fmt_yyi = 0; fmt_yyi < fmt_yynrhs; fmt_yyi++)
    {
      fprintf (stderr, "   $%d = ", fmt_yyi + 1);
      fmt_yy_symbol_print (stderr, fmt_yyrhs[fmt_yyprhs[fmt_yyrule] + fmt_yyi],
		       &(fmt_yyvsp[(fmt_yyi + 1) - (fmt_yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (fmt_yydebug)				\
    fmt_yy_reduce_print (fmt_yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int fmt_yydebug;
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

# ifndef fmt_yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define fmt_yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
fmt_yystrlen (const char *fmt_yystr)
#else
static YYSIZE_T
fmt_yystrlen (fmt_yystr)
    const char *fmt_yystr;
#endif
{
  YYSIZE_T fmt_yylen;
  for (fmt_yylen = 0; fmt_yystr[fmt_yylen]; fmt_yylen++)
    continue;
  return fmt_yylen;
}
#  endif
# endif

# ifndef fmt_yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define fmt_yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
fmt_yystpcpy (char *fmt_yydest, const char *fmt_yysrc)
#else
static char *
fmt_yystpcpy (fmt_yydest, fmt_yysrc)
    char *fmt_yydest;
    const char *fmt_yysrc;
#endif
{
  char *fmt_yyd = fmt_yydest;
  const char *fmt_yys = fmt_yysrc;

  while ((*fmt_yyd++ = *fmt_yys++) != '\0')
    continue;

  return fmt_yyd - 1;
}
#  endif
# endif

# ifndef fmt_yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for fmt_yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from fmt_yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
fmt_yytnamerr (char *fmt_yyres, const char *fmt_yystr)
{
  if (*fmt_yystr == '"')
    {
      YYSIZE_T fmt_yyn = 0;
      char const *fmt_yyp = fmt_yystr;

      for (;;)
	switch (*++fmt_yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++fmt_yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (fmt_yyres)
	      fmt_yyres[fmt_yyn] = *fmt_yyp;
	    fmt_yyn++;
	    break;

	  case '"':
	    if (fmt_yyres)
	      fmt_yyres[fmt_yyn] = '\0';
	    return fmt_yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! fmt_yyres)
    return fmt_yystrlen (fmt_yystr);

  return fmt_yystpcpy (fmt_yyres, fmt_yystr) - fmt_yyres;
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
fmt_yysyntax_error (char *fmt_yyresult, int fmt_yystate, int fmt_yychar)
{
  int fmt_yyn = fmt_yypact[fmt_yystate];

  if (! (YYPACT_NINF < fmt_yyn && fmt_yyn <= YYLAST))
    return 0;
  else
    {
      int fmt_yytype = YYTRANSLATE (fmt_yychar);
      YYSIZE_T fmt_yysize0 = fmt_yytnamerr (0, fmt_yytname[fmt_yytype]);
      YYSIZE_T fmt_yysize = fmt_yysize0;
      YYSIZE_T fmt_yysize1;
      int fmt_yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *fmt_yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int fmt_yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *fmt_yyfmt;
      char const *fmt_yyf;
      static char const fmt_yyunexpected[] = "syntax error, unexpected %s";
      static char const fmt_yyexpecting[] = ", expecting %s";
      static char const fmt_yyor[] = " or %s";
      char fmt_yyformat[sizeof fmt_yyunexpected
		    + sizeof fmt_yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof fmt_yyor - 1))];
      char const *fmt_yyprefix = fmt_yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int fmt_yyxbegin = fmt_yyn < 0 ? -fmt_yyn : 0;

      /* Stay within bounds of both fmt_yycheck and fmt_yytname.  */
      int fmt_yychecklim = YYLAST - fmt_yyn + 1;
      int fmt_yyxend = fmt_yychecklim < YYNTOKENS ? fmt_yychecklim : YYNTOKENS;
      int fmt_yycount = 1;

      fmt_yyarg[0] = fmt_yytname[fmt_yytype];
      fmt_yyfmt = fmt_yystpcpy (fmt_yyformat, fmt_yyunexpected);

      for (fmt_yyx = fmt_yyxbegin; fmt_yyx < fmt_yyxend; ++fmt_yyx)
	if (fmt_yycheck[fmt_yyx + fmt_yyn] == fmt_yyx && fmt_yyx != YYTERROR)
	  {
	    if (fmt_yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		fmt_yycount = 1;
		fmt_yysize = fmt_yysize0;
		fmt_yyformat[sizeof fmt_yyunexpected - 1] = '\0';
		break;
	      }
	    fmt_yyarg[fmt_yycount++] = fmt_yytname[fmt_yyx];
	    fmt_yysize1 = fmt_yysize + fmt_yytnamerr (0, fmt_yytname[fmt_yyx]);
	    fmt_yysize_overflow |= (fmt_yysize1 < fmt_yysize);
	    fmt_yysize = fmt_yysize1;
	    fmt_yyfmt = fmt_yystpcpy (fmt_yyfmt, fmt_yyprefix);
	    fmt_yyprefix = fmt_yyor;
	  }

      fmt_yyf = YY_(fmt_yyformat);
      fmt_yysize1 = fmt_yysize + fmt_yystrlen (fmt_yyf);
      fmt_yysize_overflow |= (fmt_yysize1 < fmt_yysize);
      fmt_yysize = fmt_yysize1;

      if (fmt_yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (fmt_yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *fmt_yyp = fmt_yyresult;
	  int fmt_yyi = 0;
	  while ((*fmt_yyp = *fmt_yyf) != '\0')
	    {
	      if (*fmt_yyp == '%' && fmt_yyf[1] == 's' && fmt_yyi < fmt_yycount)
		{
		  fmt_yyp += fmt_yytnamerr (fmt_yyp, fmt_yyarg[fmt_yyi++]);
		  fmt_yyf += 2;
		}
	      else
		{
		  fmt_yyp++;
		  fmt_yyf++;
		}
	    }
	}
      return fmt_yysize;
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
fmt_yydestruct (const char *fmt_yymsg, int fmt_yytype, YYSTYPE *fmt_yyvaluep)
#else
static void
fmt_yydestruct (fmt_yymsg, fmt_yytype, fmt_yyvaluep)
    const char *fmt_yymsg;
    int fmt_yytype;
    YYSTYPE *fmt_yyvaluep;
#endif
{
  YYUSE (fmt_yyvaluep);

  if (!fmt_yymsg)
    fmt_yymsg = "Deleting";
  YY_SYMBOL_PRINT (fmt_yymsg, fmt_yytype, fmt_yyvaluep, fmt_yylocationp);

  switch (fmt_yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int fmt_yyparse (void *YYPARSE_PARAM);
#else
int fmt_yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int fmt_yyparse (void);
#else
int fmt_yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int fmt_yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE fmt_yylval;

/* Number of syntax errors so far.  */
int fmt_yynerrs;



/*----------.
| fmt_yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
fmt_yyparse (void *YYPARSE_PARAM)
#else
int
fmt_yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
fmt_yyparse (void)
#else
int
fmt_yyparse ()

#endif
#endif
{
  
  int fmt_yystate;
  int fmt_yyn;
  int fmt_yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int fmt_yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int fmt_yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char fmt_yymsgbuf[128];
  char *fmt_yymsg = fmt_yymsgbuf;
  YYSIZE_T fmt_yymsg_alloc = sizeof fmt_yymsgbuf;
#endif

  /* Three stacks and their tools:
     `fmt_yyss': related to states,
     `fmt_yyvs': related to semantic values,
     `fmt_yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow fmt_yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  fmt_yytype_int16 fmt_yyssa[YYINITDEPTH];
  fmt_yytype_int16 *fmt_yyss = fmt_yyssa;
  fmt_yytype_int16 *fmt_yyssp;

  /* The semantic value stack.  */
  YYSTYPE fmt_yyvsa[YYINITDEPTH];
  YYSTYPE *fmt_yyvs = fmt_yyvsa;
  YYSTYPE *fmt_yyvsp;



#define YYPOPSTACK(N)   (fmt_yyvsp -= (N), fmt_yyssp -= (N))

  YYSIZE_T fmt_yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE fmt_yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int fmt_yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  fmt_yystate = 0;
  fmt_yyerrstatus = 0;
  fmt_yynerrs = 0;
  fmt_yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  fmt_yyssp = fmt_yyss;
  fmt_yyvsp = fmt_yyvs;

  goto fmt_yysetstate;

/*------------------------------------------------------------.
| fmt_yynewstate -- Push a new state, which is found in fmt_yystate.  |
`------------------------------------------------------------*/
 fmt_yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  fmt_yyssp++;

 fmt_yysetstate:
  *fmt_yyssp = fmt_yystate;

  if (fmt_yyss + fmt_yystacksize - 1 <= fmt_yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T fmt_yysize = fmt_yyssp - fmt_yyss + 1;

#ifdef fmt_yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *fmt_yyvs1 = fmt_yyvs;
	fmt_yytype_int16 *fmt_yyss1 = fmt_yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if fmt_yyoverflow is a macro.  */
	fmt_yyoverflow (YY_("memory exhausted"),
		    &fmt_yyss1, fmt_yysize * sizeof (*fmt_yyssp),
		    &fmt_yyvs1, fmt_yysize * sizeof (*fmt_yyvsp),

		    &fmt_yystacksize);

	fmt_yyss = fmt_yyss1;
	fmt_yyvs = fmt_yyvs1;
      }
#else /* no fmt_yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto fmt_yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= fmt_yystacksize)
	goto fmt_yyexhaustedlab;
      fmt_yystacksize *= 2;
      if (YYMAXDEPTH < fmt_yystacksize)
	fmt_yystacksize = YYMAXDEPTH;

      {
	fmt_yytype_int16 *fmt_yyss1 = fmt_yyss;
	union fmt_yyalloc *fmt_yyptr =
	  (union fmt_yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (fmt_yystacksize));
	if (! fmt_yyptr)
	  goto fmt_yyexhaustedlab;
	YYSTACK_RELOCATE (fmt_yyss);
	YYSTACK_RELOCATE (fmt_yyvs);

#  undef YYSTACK_RELOCATE
	if (fmt_yyss1 != fmt_yyssa)
	  YYSTACK_FREE (fmt_yyss1);
      }
# endif
#endif /* no fmt_yyoverflow */

      fmt_yyssp = fmt_yyss + fmt_yysize - 1;
      fmt_yyvsp = fmt_yyvs + fmt_yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) fmt_yystacksize));

      if (fmt_yyss + fmt_yystacksize - 1 <= fmt_yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", fmt_yystate));

  goto fmt_yybackup;

/*-----------.
| fmt_yybackup.  |
`-----------*/
fmt_yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  fmt_yyn = fmt_yypact[fmt_yystate];
  if (fmt_yyn == YYPACT_NINF)
    goto fmt_yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (fmt_yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      fmt_yychar = YYLEX;
    }

  if (fmt_yychar <= YYEOF)
    {
      fmt_yychar = fmt_yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      fmt_yytoken = YYTRANSLATE (fmt_yychar);
      YY_SYMBOL_PRINT ("Next token is", fmt_yytoken, &fmt_yylval, &fmt_yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  fmt_yyn += fmt_yytoken;
  if (fmt_yyn < 0 || YYLAST < fmt_yyn || fmt_yycheck[fmt_yyn] != fmt_yytoken)
    goto fmt_yydefault;
  fmt_yyn = fmt_yytable[fmt_yyn];
  if (fmt_yyn <= 0)
    {
      if (fmt_yyn == 0 || fmt_yyn == YYTABLE_NINF)
	goto fmt_yyerrlab;
      fmt_yyn = -fmt_yyn;
      goto fmt_yyreduce;
    }

  if (fmt_yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (fmt_yyerrstatus)
    fmt_yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", fmt_yytoken, &fmt_yylval, &fmt_yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (fmt_yychar != YYEOF)
    fmt_yychar = YYEMPTY;

  fmt_yystate = fmt_yyn;
  *++fmt_yyvsp = fmt_yylval;

  goto fmt_yynewstate;


/*-----------------------------------------------------------.
| fmt_yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
fmt_yydefault:
  fmt_yyn = fmt_yydefact[fmt_yystate];
  if (fmt_yyn == 0)
    goto fmt_yyerrlab;
  goto fmt_yyreduce;


/*-----------------------------.
| fmt_yyreduce -- Do a reduction.  |
`-----------------------------*/
fmt_yyreduce:
  /* fmt_yyn is the number of a rule to reduce with.  */
  fmt_yylen = fmt_yyr2[fmt_yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  fmt_yyval = fmt_yyvsp[1-fmt_yylen];


  YY_REDUCE_PRINT (fmt_yyn);
  switch (fmt_yyn)
    {
        case 2:
#line 76 "mh_fmtgram.y"
    {
	      /* nothing: to shut bison up */
	    }
    break;

  case 5:
#line 86 "mh_fmtgram.y"
    {
	      switch ((fmt_yyvsp[(1) - (1)].type))
		{
		case mhtype_none:
		  break;
		  
		case mhtype_num:
		  mh_code_op (mhop_num_asgn);
		  mh_code_op (mhop_num_print);
		  break;
		  
		case mhtype_str:
		  mh_code_op (mhop_str_asgn);
		  mh_code_op (mhop_str_print);
		  break;
		  
		default:
		  fmt_yyerror (_("INTERNAL ERROR: unexpected item type (please report)"));
		  abort ();
		}
	      (fmt_yyval.pc) = pc;
	    }
    break;

  case 7:
#line 112 "mh_fmtgram.y"
    {
	      in_escape = 0;
	    }
    break;

  case 8:
#line 118 "mh_fmtgram.y"
    {
	      mh_code_string ((fmt_yyvsp[(1) - (1)].str));
	      (fmt_yyval.type) = mhtype_str;
	    }
    break;

  case 9:
#line 123 "mh_fmtgram.y"
    {
	      mh_code_number ((fmt_yyvsp[(1) - (1)].num));
	      (fmt_yyval.type) = mhtype_num;
	    }
    break;

  case 12:
#line 132 "mh_fmtgram.y"
    {
	      (fmt_yyval.type) = mhtype_none;
	    }
    break;

  case 13:
#line 138 "mh_fmtgram.y"
    {
	      if (mu_c_strcasecmp ((fmt_yyvsp[(3) - (4)].str), "body") == 0)
		{
		  mh_code_op (mhop_body);
		}
	      else
		{
		  mh_code_string ((fmt_yyvsp[(3) - (4)].str));
		  mh_code_op (mhop_header);
		}
	      (fmt_yyval.type) = mhtype_str;
	    }
    break;

  case 14:
#line 153 "mh_fmtgram.y"
    {
	      in_escape++;
	    }
    break;

  case 15:
#line 159 "mh_fmtgram.y"
    {
	      in_escape--;
	    }
    break;

  case 16:
#line 164 "mh_fmtgram.y"
    { want_function = 1;}
    break;

  case 17:
#line 164 "mh_fmtgram.y"
    { want_function = 0; want_arg = 1;}
    break;

  case 18:
#line 165 "mh_fmtgram.y"
    {
	      if ((fmt_yyvsp[(4) - (7)].builtin))
		{
		  if (!mh_code_builtin ((fmt_yyvsp[(4) - (7)].builtin), (fmt_yyvsp[(6) - (7)].type)))
		    YYERROR;
		  (fmt_yyval.type) = (fmt_yyvsp[(4) - (7)].builtin)->type;
		}
	      else
		{
		  switch ((fmt_yyvsp[(6) - (7)].type))
		    {
		    default:
		      break;
		  
		    case mhtype_num:
		      mh_code_op (mhop_num_asgn);
		      break;
		  
		    case mhtype_str:
		      mh_code_op (mhop_str_asgn);
		      break;
		    }
		  (fmt_yyval.type) = mhtype_none;
		}
	    }
    break;

  case 20:
#line 194 "mh_fmtgram.y"
    {
	      mh_code_op (mhop_fmtspec);
	      mh_code_op ((fmt_yyvsp[(1) - (1)].num));
	    }
    break;

  case 22:
#line 202 "mh_fmtgram.y"
    {
	      if (strcmp ((fmt_yyvsp[(1) - (1)].str), "void") == 0)
		{
		  (fmt_yyval.builtin) = NULL;
		}
	      else
		{
		  fmt_yyerror (_("undefined function"));
		  mu_error ((fmt_yyvsp[(1) - (1)].str));
		  YYERROR;
		}
	    }
    break;

  case 23:
#line 217 "mh_fmtgram.y"
    {
	      (fmt_yyval.type) = mhtype_none;
	    }
    break;

  case 26:
#line 226 "mh_fmtgram.y"
    {
	      size_t start_pc = 0, end_pc = 0;

	      /* Fixup first condition */
	      if ((fmt_yyvsp[(5) - (7)].elif_list).cond)
		MHI_NUM(format.prog[(fmt_yyvsp[(2) - (7)].pc)]) = (fmt_yyvsp[(5) - (7)].elif_list).cond - (fmt_yyvsp[(2) - (7)].pc);
	      else if ((fmt_yyvsp[(6) - (7)].pc))
		MHI_NUM(format.prog[(fmt_yyvsp[(2) - (7)].pc)]) = (fmt_yyvsp[(6) - (7)].pc) - (fmt_yyvsp[(2) - (7)].pc);
	      else
		MHI_NUM(format.prog[(fmt_yyvsp[(2) - (7)].pc)]) = (fmt_yyvsp[(7) - (7)].elif_list).cond - (fmt_yyvsp[(2) - (7)].pc);

	      /* Link all "false" lists */
	      if ((fmt_yyvsp[(5) - (7)].elif_list).cond)
		{
		  start_pc = (fmt_yyvsp[(5) - (7)].elif_list).end;
		  end_pc = (fmt_yyvsp[(5) - (7)].elif_list).end;
		  while (MHI_NUM(format.prog[end_pc]))
		    end_pc = MHI_NUM(format.prog[end_pc]);
		}

	      if (start_pc)
		MHI_NUM(format.prog[end_pc]) = (fmt_yyvsp[(4) - (7)].pc);
	      else
		start_pc = (fmt_yyvsp[(4) - (7)].pc);

	      /* Now, fixup the end branches */
	      branch_fixup (start_pc, (fmt_yyvsp[(7) - (7)].elif_list).end);
	      MHI_NUM(format.prog[start_pc]) = (fmt_yyvsp[(7) - (7)].elif_list).end - start_pc;
	    }
    break;

  case 27:
#line 258 "mh_fmtgram.y"
    {
	      (fmt_yyval.pc) = pc;
	    }
    break;

  case 29:
#line 265 "mh_fmtgram.y"
    {
	      in_escape++;
	    }
    break;

  case 30:
#line 271 "mh_fmtgram.y"
    {
	      /* False branch of an if-block */
	      (fmt_yyval.elif_list).cond = mh_code_op (mhop_num_asgn);
	      /* Jump over the true branch */
	      mh_code_op (mhop_branch);
	      mh_code_op (2);
	      /* True branch */
	      (fmt_yyval.elif_list).end = mh_code_op (mhop_num_asgn);
	    }
    break;

  case 31:
#line 283 "mh_fmtgram.y"
    {
	      in_escape++;
	      (fmt_yyval.pc) = pc;
	    }
    break;

  case 32:
#line 290 "mh_fmtgram.y"
    {
	      mh_code_op (mhop_branch);
	      (fmt_yyval.pc) = mh_code_op (0);
	    }
    break;

  case 33:
#line 297 "mh_fmtgram.y"
    {
	      in_escape--;
	      if ((fmt_yyvsp[(1) - (1)].type) == mhtype_str)
		mh_code_op (mhop_str_branch);
	      else
		mh_code_op (mhop_num_branch);
	      (fmt_yyval.pc) = mh_code_op (0);
	    }
    break;

  case 36:
#line 312 "mh_fmtgram.y"
    {
	      (fmt_yyval.elif_list).cond = 0;
	      (fmt_yyval.elif_list).end = 0;
	    }
    break;

  case 37:
#line 317 "mh_fmtgram.y"
    {
	      (fmt_yyval.elif_list).cond = (fmt_yyvsp[(1) - (2)].elif_list).cond;
	      MHI_NUM(format.prog[(fmt_yyvsp[(2) - (2)].pc)]) = (fmt_yyvsp[(1) - (2)].elif_list).end;
	      (fmt_yyval.elif_list).end = (fmt_yyvsp[(2) - (2)].pc);
	    }
    break;

  case 38:
#line 325 "mh_fmtgram.y"
    {
	      (fmt_yyval.elif_list).cond = (fmt_yyvsp[(1) - (3)].pc);
	      MHI_NUM(format.prog[(fmt_yyvsp[(2) - (3)].pc)]) = pc - (fmt_yyvsp[(2) - (3)].pc) + 2;
	      (fmt_yyval.elif_list).end = 0;
	    }
    break;

  case 39:
#line 331 "mh_fmtgram.y"
    {
	      MHI_NUM(format.prog[(fmt_yyvsp[(4) - (5)].pc)]) = pc - (fmt_yyvsp[(4) - (5)].pc) + 2;
	      (fmt_yyval.elif_list).cond = (fmt_yyvsp[(1) - (5)].elif_list).cond;
	      MHI_NUM(format.prog[(fmt_yyvsp[(2) - (5)].pc)]) = (fmt_yyvsp[(1) - (5)].elif_list).end;
	      (fmt_yyval.elif_list).end = (fmt_yyvsp[(2) - (5)].pc);
	    }
    break;

  case 40:
#line 340 "mh_fmtgram.y"
    {
	      (fmt_yyval.pc) = 0;
	    }
    break;

  case 42:
#line 347 "mh_fmtgram.y"
    {
	      (fmt_yyval.pc) = pc;
	    }
    break;


/* Line 1267 of yacc.c.  */
#line 1745 "mh_fmtgram.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", fmt_yyr1[fmt_yyn], &fmt_yyval, &fmt_yyloc);

  YYPOPSTACK (fmt_yylen);
  fmt_yylen = 0;
  YY_STACK_PRINT (fmt_yyss, fmt_yyssp);

  *++fmt_yyvsp = fmt_yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  fmt_yyn = fmt_yyr1[fmt_yyn];

  fmt_yystate = fmt_yypgoto[fmt_yyn - YYNTOKENS] + *fmt_yyssp;
  if (0 <= fmt_yystate && fmt_yystate <= YYLAST && fmt_yycheck[fmt_yystate] == *fmt_yyssp)
    fmt_yystate = fmt_yytable[fmt_yystate];
  else
    fmt_yystate = fmt_yydefgoto[fmt_yyn - YYNTOKENS];

  goto fmt_yynewstate;


/*------------------------------------.
| fmt_yyerrlab -- here on detecting error |
`------------------------------------*/
fmt_yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!fmt_yyerrstatus)
    {
      ++fmt_yynerrs;
#if ! YYERROR_VERBOSE
      fmt_yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T fmt_yysize = fmt_yysyntax_error (0, fmt_yystate, fmt_yychar);
	if (fmt_yymsg_alloc < fmt_yysize && fmt_yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T fmt_yyalloc = 2 * fmt_yysize;
	    if (! (fmt_yysize <= fmt_yyalloc && fmt_yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      fmt_yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (fmt_yymsg != fmt_yymsgbuf)
	      YYSTACK_FREE (fmt_yymsg);
	    fmt_yymsg = (char *) YYSTACK_ALLOC (fmt_yyalloc);
	    if (fmt_yymsg)
	      fmt_yymsg_alloc = fmt_yyalloc;
	    else
	      {
		fmt_yymsg = fmt_yymsgbuf;
		fmt_yymsg_alloc = sizeof fmt_yymsgbuf;
	      }
	  }

	if (0 < fmt_yysize && fmt_yysize <= fmt_yymsg_alloc)
	  {
	    (void) fmt_yysyntax_error (fmt_yymsg, fmt_yystate, fmt_yychar);
	    fmt_yyerror (fmt_yymsg);
	  }
	else
	  {
	    fmt_yyerror (YY_("syntax error"));
	    if (fmt_yysize != 0)
	      goto fmt_yyexhaustedlab;
	  }
      }
#endif
    }



  if (fmt_yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (fmt_yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (fmt_yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  fmt_yydestruct ("Error: discarding",
		      fmt_yytoken, &fmt_yylval);
	  fmt_yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto fmt_yyerrlab1;


/*---------------------------------------------------.
| fmt_yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
fmt_yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label fmt_yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto fmt_yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (fmt_yylen);
  fmt_yylen = 0;
  YY_STACK_PRINT (fmt_yyss, fmt_yyssp);
  fmt_yystate = *fmt_yyssp;
  goto fmt_yyerrlab1;


/*-------------------------------------------------------------.
| fmt_yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
fmt_yyerrlab1:
  fmt_yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      fmt_yyn = fmt_yypact[fmt_yystate];
      if (fmt_yyn != YYPACT_NINF)
	{
	  fmt_yyn += YYTERROR;
	  if (0 <= fmt_yyn && fmt_yyn <= YYLAST && fmt_yycheck[fmt_yyn] == YYTERROR)
	    {
	      fmt_yyn = fmt_yytable[fmt_yyn];
	      if (0 < fmt_yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (fmt_yyssp == fmt_yyss)
	YYABORT;


      fmt_yydestruct ("Error: popping",
		  fmt_yystos[fmt_yystate], fmt_yyvsp);
      YYPOPSTACK (1);
      fmt_yystate = *fmt_yyssp;
      YY_STACK_PRINT (fmt_yyss, fmt_yyssp);
    }

  if (fmt_yyn == YYFINAL)
    YYACCEPT;

  *++fmt_yyvsp = fmt_yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", fmt_yystos[fmt_yyn], fmt_yyvsp, fmt_yylsp);

  fmt_yystate = fmt_yyn;
  goto fmt_yynewstate;


/*-------------------------------------.
| fmt_yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
fmt_yyacceptlab:
  fmt_yyresult = 0;
  goto fmt_yyreturn;

/*-----------------------------------.
| fmt_yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
fmt_yyabortlab:
  fmt_yyresult = 1;
  goto fmt_yyreturn;

#ifndef fmt_yyoverflow
/*-------------------------------------------------.
| fmt_yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
fmt_yyexhaustedlab:
  fmt_yyerror (YY_("memory exhausted"));
  fmt_yyresult = 2;
  /* Fall through.  */
#endif

fmt_yyreturn:
  if (fmt_yychar != YYEOF && fmt_yychar != YYEMPTY)
     fmt_yydestruct ("Cleanup: discarding lookahead",
		 fmt_yytoken, &fmt_yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (fmt_yylen);
  YY_STACK_PRINT (fmt_yyss, fmt_yyssp);
  while (fmt_yyssp != fmt_yyss)
    {
      fmt_yydestruct ("Cleanup: popping",
		  fmt_yystos[*fmt_yyssp], fmt_yyvsp);
      YYPOPSTACK (1);
    }
#ifndef fmt_yyoverflow
  if (fmt_yyss != fmt_yyssa)
    YYSTACK_FREE (fmt_yyss);
#endif
#if YYERROR_VERBOSE
  if (fmt_yymsg != fmt_yymsgbuf)
    YYSTACK_FREE (fmt_yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (fmt_yyresult);
}


#line 352 "mh_fmtgram.y"


static char *start;
static char *curp;

int
fmt_yyerror (const char *s)
{
  int len;
  mu_error ("%s: %s", start, s);
  len = curp - start;
  mu_error ("%*.*s^", len, len, "");
  return 0;
}

#define isdelim(c) (strchr("%<>?|(){} ",c) != NULL)

static int percent;
static int backslash(int c);

int
fmt_yylex ()
{
  /* Reset the tie-in */
  int expect_arg = want_arg;
  want_arg = 0;
  
  if (fmt_yydebug)
    fprintf (stderr, "[lex at %10.10s]\n", curp);
  if (*curp == '%')
    {
      curp++;
      percent = 1;
      if (mu_isdigit (*curp) || *curp == '-')
	{
	  int num = 0;
	  int flags = 0;

	  if (*curp == '-')
	    {
	      curp++;
	      flags = MH_FMT_RALIGN;
	    }
	  if (*curp == '0')
	    flags |= MH_FMT_ZEROPAD;
	  while (*curp && mu_isdigit (*curp))
	    num = num * 10 + *curp++ - '0';
	  fmt_yylval.num = num | flags;
	  return FMTSPEC;
	}
    }

  if (percent)
    {
      percent = 0;
      switch (*curp++)
	{
	case '<':
	  return IF;
	case '>':
	  return FI;
	case '?':
	  return ELIF;
	case '|':
	  return ELSE;
	case '%':
	  return '%';
	case '(':
	  return OBRACE;
	case '{':
	  return OCURLY;
	default:
	  return BOGUS;
      }
    }

  if (in_escape)
    {
      while (*curp && (*curp == ' ' || *curp == '\n'))
	curp++;
      switch (*curp)
	{
	case '(':
	  curp++;
	  return OBRACE;
	case '{':
	  curp++;
	  return OCURLY;
	case '0':case '1':case '2':case '3':case '4':
	case '5':case '6':case '7':case '8':case '9':
	  fmt_yylval.num = strtol (curp, &curp, 0);
	  return NUMBER;
	}
    }
  
  switch (*curp)
    {
    case ')':
      curp++;
      return CBRACE;
    case '}':
      curp++;
      return CCURLY;
    case 0:
      return 0;
    }

  do
    {
      if (*curp == '\\')
	{
	  int c = backslash (*++curp);
	  obstack_1grow (&stack, c);
	}
      else
	obstack_1grow (&stack, *curp);
      curp++;
    }
  while (*curp && (expect_arg ? *curp != ')' : !isdelim(*curp)));

  obstack_1grow (&stack, 0);
  fmt_yylval.str = obstack_finish (&stack);

  if (want_function)
    {
      int rest;
      mh_builtin_t *bp = mh_lookup_builtin (fmt_yylval.str, &rest);
      if (bp)
	{
	  curp -= rest;
	  fmt_yylval.builtin = bp;
	  while (*curp && mu_isspace (*curp))
	    curp++;
	  return FUNCTION;
	}
    }
  
  return STRING;
}

void
mh_format_debug (int val)
{
  fmt_yydebug = val;
}

int
mh_format_parse (char *format_str, mh_format_t *fmt)
{
  int rc;
  char *p = getenv ("MHFORMAT_DEBUG");

  if (p)
    fmt_yydebug = 1;
  start = curp = format_str;
  obstack_init (&stack);
  format.prog = NULL;
  format.progsize = 0;
  pc = 0;
  mh_code_op (mhop_stop);

  in_escape = 0; 
  percent = 0;

  rc = fmt_yyparse ();
  mh_code_op (mhop_stop);
  obstack_free (&stack, NULL);
  if (rc)
    {
      mh_format_free (&format);
      return 1;
    }
  *fmt = format;
  return 0;
}

int
backslash(int c)
{
  static char transtab[] = "b\bf\fn\nr\rt\t";
  char *p;
  
  for (p = transtab; *p; p += 2)
    {
      if (*p == c)
	return p[1];
    }
  return c;
}

void
branch_fixup (size_t epc, size_t tgt)
{
  size_t prev = MHI_NUM(format.prog[epc]);
  if (!prev)
    return;
  branch_fixup (prev, tgt);
  MHI_NUM(format.prog[prev]) = tgt - prev;
}


/* Make sure there are at least `count' entries available in the prog
   buffer */
void
prog_reserve (size_t count)
{
  if (pc + count >= format.progsize)
    {
      size_t inc = (count + 1) / FORMAT_INC + 1;
      format.progsize += inc * FORMAT_INC;
      format.prog = xrealloc (format.prog,
			      format.progsize * sizeof format.prog[0]);
    }
}

size_t
mh_code_string (char *string)
{
  int length = strlen (string) + 1;
  size_t count = (length + sizeof (mh_instr_t)) / sizeof (mh_instr_t);
  size_t start_pc = pc;
  
  mh_code_op (mhop_str_arg);
  prog_reserve (count);
  MHI_NUM(format.prog[pc++]) = count;
  memcpy (MHI_STR(format.prog[pc]), string, length);
  pc += count;
  return start_pc;
}
 
size_t
mh_code (mh_instr_t *instr)
{
  prog_reserve (1);
  format.prog[pc] = *instr;
  return pc++;
}

size_t
mh_code_op (mh_opcode_t op)
{
  mh_instr_t instr;
  MHI_OPCODE(instr) = op;
  return mh_code(&instr);
}

size_t
mh_code_number (int num)
{
  mh_instr_t instr;
  size_t ret = mh_code_op (mhop_num_arg);
  MHI_NUM(instr) = num;
  mh_code (&instr);
  return ret;
}

size_t
mh_code_builtin (mh_builtin_t *bp, int argtype)
{
  mh_instr_t instr;
  size_t start_pc = pc;
  if (bp->argtype != argtype)
    {
      if (argtype == mhtype_none)
	{
	  if (bp->optarg)
	    {
	      switch (bp->argtype)
		{
		case mhtype_num:
		  mh_code_op (mhop_num_to_arg);
		  break;
		  
		case mhtype_str:
		  if (bp->optarg == MHA_OPT_CLEAR)
		    mh_code_string ("");
		  /* mhtype_none means that the argument was an escape,
		     which has left its string value (if any) in the
		     arg_str register. Therefore, there's no need to
		     code mhop_str_to_arg */
		  break;
		  
		default:
		  fmt_yyerror (_("INTERNAL ERROR: unknown argtype (please report)"));
		  abort ();
		}
	    }
	  else
	    {
	      mu_error (_("missing argument for %s"), bp->name);
	      return 0;
	    }
	}
      else
	{
	  switch (bp->argtype)
	    {
	    case mhtype_none:
	      mu_error (_("extra arguments to %s"), bp->name);
	      return 0;
	      
	    case mhtype_num:
	      mh_code_op (mhop_str_to_num);
	      break;
	      
	    case mhtype_str:
	      mh_code_op (mhop_num_to_str);
	      break;
	    }
	}
    }

  mh_code_op (mhop_call);
  MHI_BUILTIN(instr) = bp->fun;
  mh_code (&instr);
  return start_pc;
}


