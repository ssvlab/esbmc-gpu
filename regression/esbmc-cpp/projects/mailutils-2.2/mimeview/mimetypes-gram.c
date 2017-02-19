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

/* All symbols defined below should begin with mimetypes_yy or YY, to avoid
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
   enum mimetypes_yytokentype {
     IDENT = 258,
     IDENT_L = 259,
     STRING = 260,
     EOL = 261,
     BOGUS = 262
   };
#endif
/* Tokens.  */
#define IDENT 258
#define IDENT_L 259
#define STRING 260
#define EOL 261
#define BOGUS 262




/* Copy the first part of user declarations.  */
#line 1 "mimetypes.y"

/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2007, 2009, 2010 Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
  
#include <mailutils/cctype.h>
#include <mimeview.h>
#include <mimetypes-decl.h>
  
static void
mimetypes_yyprint (FILE *output, unsigned short toknum, YYSTYPE val)
{
  switch (toknum)
    {
    case IDENT:
    case IDENT_L:
    case STRING:
      fprintf (output, "[%lu] %s", (unsigned long) val.string.len,
	       val.string.ptr);
      break;

    case EOL:
    default:
      break;
    }
}

#define YYPRINT mimetypes_yyprint

static mu_list_t arg_list; /* For error recovery */

#define L_OR  0
#define L_AND 1

enum node_type
  {
    functional_node,
    binary_node,
    negation_node,
    suffix_node
  };

union argument
{
  struct mimetypes_string *string;
  unsigned number;
  int c;
};

typedef int (*builtin_t) (union argument *args);

struct node
{
  enum node_type type;
  union
  {
    struct
    {
      builtin_t fun;
      union argument *args;
    } function;
    struct node *arg;
    struct
    {
      int op;
      struct node *arg1;
      struct node *arg2;
    } bin; 
    struct mimetypes_string suffix;
  } v;
};

static struct node *make_binary_node (int op,
				      struct node *left, struct node *rigth);
static struct node *make_negation_node (struct node *p);

static struct node *make_suffix_node (struct mimetypes_string *suffix);
static struct node *make_functional_node (char *ident, mu_list_t list);

static int eval_rule (struct node *root);

struct rule_tab
{
  char *type;
  struct node *node;
};

static mu_list_t rule_list;



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
#line 121 "mimetypes.y"
{
  struct mimetypes_string string;
  mu_list_t list;
  int result;
  struct node *node;
}
/* Line 187 of yacc.c.  */
#line 226 "mimetypes-gram.c"
	YYSTYPE;
# define mimetypes_yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 239 "mimetypes-gram.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 mimetypes_yytype_uint8;
#else
typedef unsigned char mimetypes_yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 mimetypes_yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char mimetypes_yytype_int8;
#else
typedef short int mimetypes_yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 mimetypes_yytype_uint16;
#else
typedef unsigned short int mimetypes_yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 mimetypes_yytype_int16;
#else
typedef short int mimetypes_yytype_int16;
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

#if ! defined mimetypes_yyoverflow || YYERROR_VERBOSE

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
#endif /* ! defined mimetypes_yyoverflow || YYERROR_VERBOSE */


#if (! defined mimetypes_yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union mimetypes_yyalloc
{
  mimetypes_yytype_int16 mimetypes_yyss;
  YYSTYPE mimetypes_yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union mimetypes_yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (mimetypes_yytype_int16) + sizeof (YYSTYPE)) \
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
	  YYSIZE_T mimetypes_yyi;				\
	  for (mimetypes_yyi = 0; mimetypes_yyi < (Count); mimetypes_yyi++)	\
	    (To)[mimetypes_yyi] = (From)[mimetypes_yyi];		\
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
	YYSIZE_T mimetypes_yynewbytes;						\
	YYCOPY (&mimetypes_yyptr->Stack, Stack, mimetypes_yysize);				\
	Stack = &mimetypes_yyptr->Stack;						\
	mimetypes_yynewbytes = mimetypes_yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	mimetypes_yyptr += mimetypes_yynewbytes / sizeof (*mimetypes_yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  10
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   54

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  14
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  12
/* YYNRULES -- Number of rules.  */
#define YYNRULES  24
/* YYNRULES -- Number of states.  */
#define YYNSTATES  38

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   262

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? mimetypes_yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const mimetypes_yytype_uint8 mimetypes_yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    11,     2,     2,     2,     2,     2,     2,
      12,    13,     2,     9,     8,     2,     2,    10,     2,     2,
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
       5,     6,     7
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const mimetypes_yytype_uint8 mimetypes_yyprhs[] =
{
       0,     0,     3,     5,     7,    11,    12,    15,    18,    20,
      23,    27,    29,    32,    36,    40,    43,    47,    49,    51,
      53,    55,    59,    61,    65
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const mimetypes_yytype_int8 mimetypes_yyrhs[] =
{
      15,     0,    -1,    16,    -1,    17,    -1,    16,    18,    17,
      -1,    -1,    19,    20,    -1,     1,    18,    -1,     6,    -1,
      18,     6,    -1,     3,    10,     3,    -1,    21,    -1,    20,
      20,    -1,    20,     8,    20,    -1,    20,     9,    20,    -1,
      11,    21,    -1,    12,    20,    13,    -1,    22,    -1,    23,
      -1,     5,    -1,     3,    -1,     4,    24,    13,    -1,    25,
      -1,    24,     8,    25,    -1,    22,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const mimetypes_yytype_uint8 mimetypes_yyrline[] =
{
       0,   130,   130,   133,   134,   137,   138,   147,   156,   157,
     160,   166,   167,   171,   175,   181,   185,   189,   193,   196,
     197,   200,   209,   215,   222
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const mimetypes_yytname[] =
{
  "$end", "error", "$undefined", "IDENT", "IDENT_L", "STRING", "EOL",
  "BOGUS", "','", "'+'", "'/'", "'!'", "'('", "')'", "$accept", "input",
  "list", "rule_line", "eol", "type", "rule", "stmt", "string", "function",
  "arglist", "arg", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const mimetypes_yytype_uint16 mimetypes_yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,    44,    43,
      47,    33,    40,    41
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const mimetypes_yytype_uint8 mimetypes_yyr1[] =
{
       0,    14,    15,    16,    16,    17,    17,    17,    18,    18,
      19,    20,    20,    20,    20,    21,    21,    21,    21,    22,
      22,    23,    24,    24,    25
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const mimetypes_yytype_uint8 mimetypes_yyr2[] =
{
       0,     2,     1,     1,     3,     0,     2,     2,     1,     2,
       3,     1,     2,     3,     3,     2,     3,     1,     1,     1,
       1,     3,     1,     3,     1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const mimetypes_yytype_uint8 mimetypes_yydefact[] =
{
       0,     0,     0,     0,     2,     3,     0,     8,     7,     0,
       1,     0,    20,     0,    19,     0,     0,     6,    11,    17,
      18,     9,    10,     4,    24,     0,    22,    15,     0,     0,
       0,    12,     0,    21,    16,    13,    14,    23
};

/* YYDEFGOTO[NTERM-NUM].  */
static const mimetypes_yytype_int8 mimetypes_yydefgoto[] =
{
      -1,     3,     4,     5,     8,     6,    31,    18,    19,    20,
      25,    26
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -13
static const mimetypes_yytype_int8 mimetypes_yypact[] =
{
      15,    -4,    -3,    17,    -4,   -13,    35,   -13,    16,     3,
     -13,    48,   -13,    47,   -13,    35,    35,    22,   -13,   -13,
     -13,   -13,   -13,   -13,   -13,     6,   -13,   -13,     0,    35,
      35,    32,    47,   -13,   -13,    32,    35,   -13
};

/* YYPGOTO[NTERM-NUM].  */
static const mimetypes_yytype_int8 mimetypes_yypgoto[] =
{
     -13,   -13,   -13,    18,    24,   -13,    -6,    27,   -12,   -13,
     -13,    13
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -6
static const mimetypes_yytype_int8 mimetypes_yytable[] =
{
      17,    24,     7,    12,    13,    14,    22,     9,    29,    30,
      28,    15,    16,    34,    32,    -5,     1,    10,     2,    33,
      24,    -5,    21,    35,    36,    12,    13,    14,    11,    23,
      29,    30,     0,    15,    16,    12,    13,    14,    12,    13,
      14,    30,    27,    15,    16,    37,    15,    16,    -5,     1,
      12,     2,    14,     0,    21
};

static const mimetypes_yytype_int8 mimetypes_yycheck[] =
{
       6,    13,     6,     3,     4,     5,     3,    10,     8,     9,
      16,    11,    12,    13,     8,     0,     1,     0,     3,    13,
      32,     6,     6,    29,    30,     3,     4,     5,     4,    11,
       8,     9,    -1,    11,    12,     3,     4,     5,     3,     4,
       5,     9,    15,    11,    12,    32,    11,    12,     0,     1,
       3,     3,     5,    -1,     6
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const mimetypes_yytype_uint8 mimetypes_yystos[] =
{
       0,     1,     3,    15,    16,    17,    19,     6,    18,    10,
       0,    18,     3,     4,     5,    11,    12,    20,    21,    22,
      23,     6,     3,    17,    22,    24,    25,    21,    20,     8,
       9,    20,     8,    13,    13,    20,    20,    25
};

#define mimetypes_yyerrok		(mimetypes_yyerrstatus = 0)
#define mimetypes_yyclearin	(mimetypes_yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto mimetypes_yyacceptlab
#define YYABORT		goto mimetypes_yyabortlab
#define YYERROR		goto mimetypes_yyerrorlab


/* Like YYERROR except do call mimetypes_yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto mimetypes_yyerrlab

#define YYRECOVERING()  (!!mimetypes_yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (mimetypes_yychar == YYEMPTY && mimetypes_yylen == 1)				\
    {								\
      mimetypes_yychar = (Token);						\
      mimetypes_yylval = (Value);						\
      mimetypes_yytoken = YYTRANSLATE (mimetypes_yychar);				\
      YYPOPSTACK (1);						\
      goto mimetypes_yybackup;						\
    }								\
  else								\
    {								\
      mimetypes_yyerror (YY_("syntax error: cannot back up")); \
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


/* YYLEX -- calling `mimetypes_yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX mimetypes_yylex (YYLEX_PARAM)
#else
# define YYLEX mimetypes_yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (mimetypes_yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (mimetypes_yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      mimetypes_yy_symbol_print (stderr,						  \
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
mimetypes_yy_symbol_value_print (FILE *mimetypes_yyoutput, int mimetypes_yytype, YYSTYPE const * const mimetypes_yyvaluep)
#else
static void
mimetypes_yy_symbol_value_print (mimetypes_yyoutput, mimetypes_yytype, mimetypes_yyvaluep)
    FILE *mimetypes_yyoutput;
    int mimetypes_yytype;
    YYSTYPE const * const mimetypes_yyvaluep;
#endif
{
  if (!mimetypes_yyvaluep)
    return;
# ifdef YYPRINT
  if (mimetypes_yytype < YYNTOKENS)
    YYPRINT (mimetypes_yyoutput, mimetypes_yytoknum[mimetypes_yytype], *mimetypes_yyvaluep);
# else
  YYUSE (mimetypes_yyoutput);
# endif
  switch (mimetypes_yytype)
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
mimetypes_yy_symbol_print (FILE *mimetypes_yyoutput, int mimetypes_yytype, YYSTYPE const * const mimetypes_yyvaluep)
#else
static void
mimetypes_yy_symbol_print (mimetypes_yyoutput, mimetypes_yytype, mimetypes_yyvaluep)
    FILE *mimetypes_yyoutput;
    int mimetypes_yytype;
    YYSTYPE const * const mimetypes_yyvaluep;
#endif
{
  if (mimetypes_yytype < YYNTOKENS)
    YYFPRINTF (mimetypes_yyoutput, "token %s (", mimetypes_yytname[mimetypes_yytype]);
  else
    YYFPRINTF (mimetypes_yyoutput, "nterm %s (", mimetypes_yytname[mimetypes_yytype]);

  mimetypes_yy_symbol_value_print (mimetypes_yyoutput, mimetypes_yytype, mimetypes_yyvaluep);
  YYFPRINTF (mimetypes_yyoutput, ")");
}

/*------------------------------------------------------------------.
| mimetypes_yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
mimetypes_yy_stack_print (mimetypes_yytype_int16 *bottom, mimetypes_yytype_int16 *top)
#else
static void
mimetypes_yy_stack_print (bottom, top)
    mimetypes_yytype_int16 *bottom;
    mimetypes_yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (mimetypes_yydebug)							\
    mimetypes_yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
mimetypes_yy_reduce_print (YYSTYPE *mimetypes_yyvsp, int mimetypes_yyrule)
#else
static void
mimetypes_yy_reduce_print (mimetypes_yyvsp, mimetypes_yyrule)
    YYSTYPE *mimetypes_yyvsp;
    int mimetypes_yyrule;
#endif
{
  int mimetypes_yynrhs = mimetypes_yyr2[mimetypes_yyrule];
  int mimetypes_yyi;
  unsigned long int mimetypes_yylno = mimetypes_yyrline[mimetypes_yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     mimetypes_yyrule - 1, mimetypes_yylno);
  /* The symbols being reduced.  */
  for (mimetypes_yyi = 0; mimetypes_yyi < mimetypes_yynrhs; mimetypes_yyi++)
    {
      fprintf (stderr, "   $%d = ", mimetypes_yyi + 1);
      mimetypes_yy_symbol_print (stderr, mimetypes_yyrhs[mimetypes_yyprhs[mimetypes_yyrule] + mimetypes_yyi],
		       &(mimetypes_yyvsp[(mimetypes_yyi + 1) - (mimetypes_yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (mimetypes_yydebug)				\
    mimetypes_yy_reduce_print (mimetypes_yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int mimetypes_yydebug;
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

# ifndef mimetypes_yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define mimetypes_yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
mimetypes_yystrlen (const char *mimetypes_yystr)
#else
static YYSIZE_T
mimetypes_yystrlen (mimetypes_yystr)
    const char *mimetypes_yystr;
#endif
{
  YYSIZE_T mimetypes_yylen;
  for (mimetypes_yylen = 0; mimetypes_yystr[mimetypes_yylen]; mimetypes_yylen++)
    continue;
  return mimetypes_yylen;
}
#  endif
# endif

# ifndef mimetypes_yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define mimetypes_yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
mimetypes_yystpcpy (char *mimetypes_yydest, const char *mimetypes_yysrc)
#else
static char *
mimetypes_yystpcpy (mimetypes_yydest, mimetypes_yysrc)
    char *mimetypes_yydest;
    const char *mimetypes_yysrc;
#endif
{
  char *mimetypes_yyd = mimetypes_yydest;
  const char *mimetypes_yys = mimetypes_yysrc;

  while ((*mimetypes_yyd++ = *mimetypes_yys++) != '\0')
    continue;

  return mimetypes_yyd - 1;
}
#  endif
# endif

# ifndef mimetypes_yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for mimetypes_yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from mimetypes_yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
mimetypes_yytnamerr (char *mimetypes_yyres, const char *mimetypes_yystr)
{
  if (*mimetypes_yystr == '"')
    {
      YYSIZE_T mimetypes_yyn = 0;
      char const *mimetypes_yyp = mimetypes_yystr;

      for (;;)
	switch (*++mimetypes_yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++mimetypes_yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (mimetypes_yyres)
	      mimetypes_yyres[mimetypes_yyn] = *mimetypes_yyp;
	    mimetypes_yyn++;
	    break;

	  case '"':
	    if (mimetypes_yyres)
	      mimetypes_yyres[mimetypes_yyn] = '\0';
	    return mimetypes_yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! mimetypes_yyres)
    return mimetypes_yystrlen (mimetypes_yystr);

  return mimetypes_yystpcpy (mimetypes_yyres, mimetypes_yystr) - mimetypes_yyres;
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
mimetypes_yysyntax_error (char *mimetypes_yyresult, int mimetypes_yystate, int mimetypes_yychar)
{
  int mimetypes_yyn = mimetypes_yypact[mimetypes_yystate];

  if (! (YYPACT_NINF < mimetypes_yyn && mimetypes_yyn <= YYLAST))
    return 0;
  else
    {
      int mimetypes_yytype = YYTRANSLATE (mimetypes_yychar);
      YYSIZE_T mimetypes_yysize0 = mimetypes_yytnamerr (0, mimetypes_yytname[mimetypes_yytype]);
      YYSIZE_T mimetypes_yysize = mimetypes_yysize0;
      YYSIZE_T mimetypes_yysize1;
      int mimetypes_yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *mimetypes_yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int mimetypes_yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *mimetypes_yyfmt;
      char const *mimetypes_yyf;
      static char const mimetypes_yyunexpected[] = "syntax error, unexpected %s";
      static char const mimetypes_yyexpecting[] = ", expecting %s";
      static char const mimetypes_yyor[] = " or %s";
      char mimetypes_yyformat[sizeof mimetypes_yyunexpected
		    + sizeof mimetypes_yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof mimetypes_yyor - 1))];
      char const *mimetypes_yyprefix = mimetypes_yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int mimetypes_yyxbegin = mimetypes_yyn < 0 ? -mimetypes_yyn : 0;

      /* Stay within bounds of both mimetypes_yycheck and mimetypes_yytname.  */
      int mimetypes_yychecklim = YYLAST - mimetypes_yyn + 1;
      int mimetypes_yyxend = mimetypes_yychecklim < YYNTOKENS ? mimetypes_yychecklim : YYNTOKENS;
      int mimetypes_yycount = 1;

      mimetypes_yyarg[0] = mimetypes_yytname[mimetypes_yytype];
      mimetypes_yyfmt = mimetypes_yystpcpy (mimetypes_yyformat, mimetypes_yyunexpected);

      for (mimetypes_yyx = mimetypes_yyxbegin; mimetypes_yyx < mimetypes_yyxend; ++mimetypes_yyx)
	if (mimetypes_yycheck[mimetypes_yyx + mimetypes_yyn] == mimetypes_yyx && mimetypes_yyx != YYTERROR)
	  {
	    if (mimetypes_yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		mimetypes_yycount = 1;
		mimetypes_yysize = mimetypes_yysize0;
		mimetypes_yyformat[sizeof mimetypes_yyunexpected - 1] = '\0';
		break;
	      }
	    mimetypes_yyarg[mimetypes_yycount++] = mimetypes_yytname[mimetypes_yyx];
	    mimetypes_yysize1 = mimetypes_yysize + mimetypes_yytnamerr (0, mimetypes_yytname[mimetypes_yyx]);
	    mimetypes_yysize_overflow |= (mimetypes_yysize1 < mimetypes_yysize);
	    mimetypes_yysize = mimetypes_yysize1;
	    mimetypes_yyfmt = mimetypes_yystpcpy (mimetypes_yyfmt, mimetypes_yyprefix);
	    mimetypes_yyprefix = mimetypes_yyor;
	  }

      mimetypes_yyf = YY_(mimetypes_yyformat);
      mimetypes_yysize1 = mimetypes_yysize + mimetypes_yystrlen (mimetypes_yyf);
      mimetypes_yysize_overflow |= (mimetypes_yysize1 < mimetypes_yysize);
      mimetypes_yysize = mimetypes_yysize1;

      if (mimetypes_yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (mimetypes_yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *mimetypes_yyp = mimetypes_yyresult;
	  int mimetypes_yyi = 0;
	  while ((*mimetypes_yyp = *mimetypes_yyf) != '\0')
	    {
	      if (*mimetypes_yyp == '%' && mimetypes_yyf[1] == 's' && mimetypes_yyi < mimetypes_yycount)
		{
		  mimetypes_yyp += mimetypes_yytnamerr (mimetypes_yyp, mimetypes_yyarg[mimetypes_yyi++]);
		  mimetypes_yyf += 2;
		}
	      else
		{
		  mimetypes_yyp++;
		  mimetypes_yyf++;
		}
	    }
	}
      return mimetypes_yysize;
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
mimetypes_yydestruct (const char *mimetypes_yymsg, int mimetypes_yytype, YYSTYPE *mimetypes_yyvaluep)
#else
static void
mimetypes_yydestruct (mimetypes_yymsg, mimetypes_yytype, mimetypes_yyvaluep)
    const char *mimetypes_yymsg;
    int mimetypes_yytype;
    YYSTYPE *mimetypes_yyvaluep;
#endif
{
  YYUSE (mimetypes_yyvaluep);

  if (!mimetypes_yymsg)
    mimetypes_yymsg = "Deleting";
  YY_SYMBOL_PRINT (mimetypes_yymsg, mimetypes_yytype, mimetypes_yyvaluep, mimetypes_yylocationp);

  switch (mimetypes_yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int mimetypes_yyparse (void *YYPARSE_PARAM);
#else
int mimetypes_yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int mimetypes_yyparse (void);
#else
int mimetypes_yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int mimetypes_yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE mimetypes_yylval;

/* Number of syntax errors so far.  */
int mimetypes_yynerrs;



/*----------.
| mimetypes_yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
mimetypes_yyparse (void *YYPARSE_PARAM)
#else
int
mimetypes_yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
mimetypes_yyparse (void)
#else
int
mimetypes_yyparse ()

#endif
#endif
{
  
  int mimetypes_yystate;
  int mimetypes_yyn;
  int mimetypes_yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int mimetypes_yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int mimetypes_yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char mimetypes_yymsgbuf[128];
  char *mimetypes_yymsg = mimetypes_yymsgbuf;
  YYSIZE_T mimetypes_yymsg_alloc = sizeof mimetypes_yymsgbuf;
#endif

  /* Three stacks and their tools:
     `mimetypes_yyss': related to states,
     `mimetypes_yyvs': related to semantic values,
     `mimetypes_yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow mimetypes_yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  mimetypes_yytype_int16 mimetypes_yyssa[YYINITDEPTH];
  mimetypes_yytype_int16 *mimetypes_yyss = mimetypes_yyssa;
  mimetypes_yytype_int16 *mimetypes_yyssp;

  /* The semantic value stack.  */
  YYSTYPE mimetypes_yyvsa[YYINITDEPTH];
  YYSTYPE *mimetypes_yyvs = mimetypes_yyvsa;
  YYSTYPE *mimetypes_yyvsp;



#define YYPOPSTACK(N)   (mimetypes_yyvsp -= (N), mimetypes_yyssp -= (N))

  YYSIZE_T mimetypes_yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE mimetypes_yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int mimetypes_yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  mimetypes_yystate = 0;
  mimetypes_yyerrstatus = 0;
  mimetypes_yynerrs = 0;
  mimetypes_yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  mimetypes_yyssp = mimetypes_yyss;
  mimetypes_yyvsp = mimetypes_yyvs;

  goto mimetypes_yysetstate;

/*------------------------------------------------------------.
| mimetypes_yynewstate -- Push a new state, which is found in mimetypes_yystate.  |
`------------------------------------------------------------*/
 mimetypes_yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  mimetypes_yyssp++;

 mimetypes_yysetstate:
  *mimetypes_yyssp = mimetypes_yystate;

  if (mimetypes_yyss + mimetypes_yystacksize - 1 <= mimetypes_yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T mimetypes_yysize = mimetypes_yyssp - mimetypes_yyss + 1;

#ifdef mimetypes_yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *mimetypes_yyvs1 = mimetypes_yyvs;
	mimetypes_yytype_int16 *mimetypes_yyss1 = mimetypes_yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if mimetypes_yyoverflow is a macro.  */
	mimetypes_yyoverflow (YY_("memory exhausted"),
		    &mimetypes_yyss1, mimetypes_yysize * sizeof (*mimetypes_yyssp),
		    &mimetypes_yyvs1, mimetypes_yysize * sizeof (*mimetypes_yyvsp),

		    &mimetypes_yystacksize);

	mimetypes_yyss = mimetypes_yyss1;
	mimetypes_yyvs = mimetypes_yyvs1;
      }
#else /* no mimetypes_yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto mimetypes_yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= mimetypes_yystacksize)
	goto mimetypes_yyexhaustedlab;
      mimetypes_yystacksize *= 2;
      if (YYMAXDEPTH < mimetypes_yystacksize)
	mimetypes_yystacksize = YYMAXDEPTH;

      {
	mimetypes_yytype_int16 *mimetypes_yyss1 = mimetypes_yyss;
	union mimetypes_yyalloc *mimetypes_yyptr =
	  (union mimetypes_yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (mimetypes_yystacksize));
	if (! mimetypes_yyptr)
	  goto mimetypes_yyexhaustedlab;
	YYSTACK_RELOCATE (mimetypes_yyss);
	YYSTACK_RELOCATE (mimetypes_yyvs);

#  undef YYSTACK_RELOCATE
	if (mimetypes_yyss1 != mimetypes_yyssa)
	  YYSTACK_FREE (mimetypes_yyss1);
      }
# endif
#endif /* no mimetypes_yyoverflow */

      mimetypes_yyssp = mimetypes_yyss + mimetypes_yysize - 1;
      mimetypes_yyvsp = mimetypes_yyvs + mimetypes_yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) mimetypes_yystacksize));

      if (mimetypes_yyss + mimetypes_yystacksize - 1 <= mimetypes_yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", mimetypes_yystate));

  goto mimetypes_yybackup;

/*-----------.
| mimetypes_yybackup.  |
`-----------*/
mimetypes_yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  mimetypes_yyn = mimetypes_yypact[mimetypes_yystate];
  if (mimetypes_yyn == YYPACT_NINF)
    goto mimetypes_yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (mimetypes_yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      mimetypes_yychar = YYLEX;
    }

  if (mimetypes_yychar <= YYEOF)
    {
      mimetypes_yychar = mimetypes_yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      mimetypes_yytoken = YYTRANSLATE (mimetypes_yychar);
      YY_SYMBOL_PRINT ("Next token is", mimetypes_yytoken, &mimetypes_yylval, &mimetypes_yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  mimetypes_yyn += mimetypes_yytoken;
  if (mimetypes_yyn < 0 || YYLAST < mimetypes_yyn || mimetypes_yycheck[mimetypes_yyn] != mimetypes_yytoken)
    goto mimetypes_yydefault;
  mimetypes_yyn = mimetypes_yytable[mimetypes_yyn];
  if (mimetypes_yyn <= 0)
    {
      if (mimetypes_yyn == 0 || mimetypes_yyn == YYTABLE_NINF)
	goto mimetypes_yyerrlab;
      mimetypes_yyn = -mimetypes_yyn;
      goto mimetypes_yyreduce;
    }

  if (mimetypes_yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (mimetypes_yyerrstatus)
    mimetypes_yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", mimetypes_yytoken, &mimetypes_yylval, &mimetypes_yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (mimetypes_yychar != YYEOF)
    mimetypes_yychar = YYEMPTY;

  mimetypes_yystate = mimetypes_yyn;
  *++mimetypes_yyvsp = mimetypes_yylval;

  goto mimetypes_yynewstate;


/*-----------------------------------------------------------.
| mimetypes_yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
mimetypes_yydefault:
  mimetypes_yyn = mimetypes_yydefact[mimetypes_yystate];
  if (mimetypes_yyn == 0)
    goto mimetypes_yyerrlab;
  goto mimetypes_yyreduce;


/*-----------------------------.
| mimetypes_yyreduce -- Do a reduction.  |
`-----------------------------*/
mimetypes_yyreduce:
  /* mimetypes_yyn is the number of a rule to reduce with.  */
  mimetypes_yylen = mimetypes_yyr2[mimetypes_yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  mimetypes_yyval = mimetypes_yyvsp[1-mimetypes_yylen];


  YY_REDUCE_PRINT (mimetypes_yyn);
  switch (mimetypes_yyn)
    {
        case 6:
#line 139 "mimetypes.y"
    {
	     struct rule_tab *p = mimetypes_malloc (sizeof (*p));
	     if (!rule_list)
	       mu_list_create (&rule_list);
	     p->type = (mimetypes_yyvsp[(1) - (2)].string).ptr;
	     p->node = (mimetypes_yyvsp[(2) - (2)].node);
	     mu_list_append (rule_list, p);
	   }
    break;

  case 7:
#line 148 "mimetypes.y"
    {
	     if (arg_list)
	       mu_list_destroy (&arg_list);
	     arg_list = NULL;
	     reset_lex ();
	   }
    break;

  case 10:
#line 161 "mimetypes.y"
    {
	     (mimetypes_yyval.string) = mimetypes_append_string2 (&(mimetypes_yyvsp[(1) - (3)].string), '/', &(mimetypes_yyvsp[(3) - (3)].string));
	   }
    break;

  case 12:
#line 168 "mimetypes.y"
    {
	     (mimetypes_yyval.node) = make_binary_node (L_OR, (mimetypes_yyvsp[(1) - (2)].node), (mimetypes_yyvsp[(2) - (2)].node));
	   }
    break;

  case 13:
#line 172 "mimetypes.y"
    {
	     (mimetypes_yyval.node) = make_binary_node (L_OR, (mimetypes_yyvsp[(1) - (3)].node), (mimetypes_yyvsp[(3) - (3)].node));
	   }
    break;

  case 14:
#line 176 "mimetypes.y"
    {
	     (mimetypes_yyval.node) = make_binary_node (L_AND, (mimetypes_yyvsp[(1) - (3)].node), (mimetypes_yyvsp[(3) - (3)].node));
	   }
    break;

  case 15:
#line 182 "mimetypes.y"
    {
	     (mimetypes_yyval.node) = make_negation_node ((mimetypes_yyvsp[(2) - (2)].node));
	   }
    break;

  case 16:
#line 186 "mimetypes.y"
    {
	     (mimetypes_yyval.node) = (mimetypes_yyvsp[(2) - (3)].node);
	   }
    break;

  case 17:
#line 190 "mimetypes.y"
    {
	     (mimetypes_yyval.node) = make_suffix_node (&(mimetypes_yyvsp[(1) - (1)].string));
	   }
    break;

  case 21:
#line 201 "mimetypes.y"
    {
	     reset_lex ();
	     (mimetypes_yyval.node) = make_functional_node ((mimetypes_yyvsp[(1) - (3)].string).ptr, (mimetypes_yyvsp[(2) - (3)].list));
	     if (!(mimetypes_yyval.node))
	       YYERROR;
	   }
    break;

  case 22:
#line 210 "mimetypes.y"
    {
	     mu_list_create (&arg_list);
	     (mimetypes_yyval.list) = arg_list;
	     mu_list_append ((mimetypes_yyval.list), mimetypes_string_dup (&(mimetypes_yyvsp[(1) - (1)].string)));
	   }
    break;

  case 23:
#line 216 "mimetypes.y"
    {
	     mu_list_append ((mimetypes_yyvsp[(1) - (3)].list), mimetypes_string_dup (&(mimetypes_yyvsp[(3) - (3)].string)));
	     (mimetypes_yyval.list) = (mimetypes_yyvsp[(1) - (3)].list);
	   }
    break;


/* Line 1267 of yacc.c.  */
#line 1555 "mimetypes-gram.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", mimetypes_yyr1[mimetypes_yyn], &mimetypes_yyval, &mimetypes_yyloc);

  YYPOPSTACK (mimetypes_yylen);
  mimetypes_yylen = 0;
  YY_STACK_PRINT (mimetypes_yyss, mimetypes_yyssp);

  *++mimetypes_yyvsp = mimetypes_yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  mimetypes_yyn = mimetypes_yyr1[mimetypes_yyn];

  mimetypes_yystate = mimetypes_yypgoto[mimetypes_yyn - YYNTOKENS] + *mimetypes_yyssp;
  if (0 <= mimetypes_yystate && mimetypes_yystate <= YYLAST && mimetypes_yycheck[mimetypes_yystate] == *mimetypes_yyssp)
    mimetypes_yystate = mimetypes_yytable[mimetypes_yystate];
  else
    mimetypes_yystate = mimetypes_yydefgoto[mimetypes_yyn - YYNTOKENS];

  goto mimetypes_yynewstate;


/*------------------------------------.
| mimetypes_yyerrlab -- here on detecting error |
`------------------------------------*/
mimetypes_yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!mimetypes_yyerrstatus)
    {
      ++mimetypes_yynerrs;
#if ! YYERROR_VERBOSE
      mimetypes_yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T mimetypes_yysize = mimetypes_yysyntax_error (0, mimetypes_yystate, mimetypes_yychar);
	if (mimetypes_yymsg_alloc < mimetypes_yysize && mimetypes_yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T mimetypes_yyalloc = 2 * mimetypes_yysize;
	    if (! (mimetypes_yysize <= mimetypes_yyalloc && mimetypes_yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      mimetypes_yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (mimetypes_yymsg != mimetypes_yymsgbuf)
	      YYSTACK_FREE (mimetypes_yymsg);
	    mimetypes_yymsg = (char *) YYSTACK_ALLOC (mimetypes_yyalloc);
	    if (mimetypes_yymsg)
	      mimetypes_yymsg_alloc = mimetypes_yyalloc;
	    else
	      {
		mimetypes_yymsg = mimetypes_yymsgbuf;
		mimetypes_yymsg_alloc = sizeof mimetypes_yymsgbuf;
	      }
	  }

	if (0 < mimetypes_yysize && mimetypes_yysize <= mimetypes_yymsg_alloc)
	  {
	    (void) mimetypes_yysyntax_error (mimetypes_yymsg, mimetypes_yystate, mimetypes_yychar);
	    mimetypes_yyerror (mimetypes_yymsg);
	  }
	else
	  {
	    mimetypes_yyerror (YY_("syntax error"));
	    if (mimetypes_yysize != 0)
	      goto mimetypes_yyexhaustedlab;
	  }
      }
#endif
    }



  if (mimetypes_yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (mimetypes_yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (mimetypes_yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  mimetypes_yydestruct ("Error: discarding",
		      mimetypes_yytoken, &mimetypes_yylval);
	  mimetypes_yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto mimetypes_yyerrlab1;


/*---------------------------------------------------.
| mimetypes_yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
mimetypes_yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label mimetypes_yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto mimetypes_yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (mimetypes_yylen);
  mimetypes_yylen = 0;
  YY_STACK_PRINT (mimetypes_yyss, mimetypes_yyssp);
  mimetypes_yystate = *mimetypes_yyssp;
  goto mimetypes_yyerrlab1;


/*-------------------------------------------------------------.
| mimetypes_yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
mimetypes_yyerrlab1:
  mimetypes_yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      mimetypes_yyn = mimetypes_yypact[mimetypes_yystate];
      if (mimetypes_yyn != YYPACT_NINF)
	{
	  mimetypes_yyn += YYTERROR;
	  if (0 <= mimetypes_yyn && mimetypes_yyn <= YYLAST && mimetypes_yycheck[mimetypes_yyn] == YYTERROR)
	    {
	      mimetypes_yyn = mimetypes_yytable[mimetypes_yyn];
	      if (0 < mimetypes_yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (mimetypes_yyssp == mimetypes_yyss)
	YYABORT;


      mimetypes_yydestruct ("Error: popping",
		  mimetypes_yystos[mimetypes_yystate], mimetypes_yyvsp);
      YYPOPSTACK (1);
      mimetypes_yystate = *mimetypes_yyssp;
      YY_STACK_PRINT (mimetypes_yyss, mimetypes_yyssp);
    }

  if (mimetypes_yyn == YYFINAL)
    YYACCEPT;

  *++mimetypes_yyvsp = mimetypes_yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", mimetypes_yystos[mimetypes_yyn], mimetypes_yyvsp, mimetypes_yylsp);

  mimetypes_yystate = mimetypes_yyn;
  goto mimetypes_yynewstate;


/*-------------------------------------.
| mimetypes_yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
mimetypes_yyacceptlab:
  mimetypes_yyresult = 0;
  goto mimetypes_yyreturn;

/*-----------------------------------.
| mimetypes_yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
mimetypes_yyabortlab:
  mimetypes_yyresult = 1;
  goto mimetypes_yyreturn;

#ifndef mimetypes_yyoverflow
/*-------------------------------------------------.
| mimetypes_yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
mimetypes_yyexhaustedlab:
  mimetypes_yyerror (YY_("memory exhausted"));
  mimetypes_yyresult = 2;
  /* Fall through.  */
#endif

mimetypes_yyreturn:
  if (mimetypes_yychar != YYEOF && mimetypes_yychar != YYEMPTY)
     mimetypes_yydestruct ("Cleanup: discarding lookahead",
		 mimetypes_yytoken, &mimetypes_yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (mimetypes_yylen);
  YY_STACK_PRINT (mimetypes_yyss, mimetypes_yyssp);
  while (mimetypes_yyssp != mimetypes_yyss)
    {
      mimetypes_yydestruct ("Cleanup: popping",
		  mimetypes_yystos[*mimetypes_yyssp], mimetypes_yyvsp);
      YYPOPSTACK (1);
    }
#ifndef mimetypes_yyoverflow
  if (mimetypes_yyss != mimetypes_yyssa)
    YYSTACK_FREE (mimetypes_yyss);
#endif
#if YYERROR_VERBOSE
  if (mimetypes_yymsg != mimetypes_yymsgbuf)
    YYSTACK_FREE (mimetypes_yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (mimetypes_yyresult);
}


#line 225 "mimetypes.y"


int
mimetypes_parse (const char *name)
{
  int rc;
  if (mimetypes_open (name))
    return 1;
  rc = mimetypes_yyparse ();
  mimetypes_close ();
  return rule_list == NULL;
}
  
void
mimetypes_gram_debug (int level)
{
  mimetypes_yydebug = level;
}


static struct node *
make_node (enum node_type type)
{
  struct node *p = mimetypes_malloc (sizeof *p);
  p->type = type;
  return p;
}

static struct node *
make_binary_node (int op, struct node *left, struct node *right)
{
  struct node *node = make_node (binary_node);

  node->v.bin.op = op;
  node->v.bin.arg1 = left;
  node->v.bin.arg2 = right;
  return node;
}

struct node *
make_negation_node (struct node *p)
{
  struct node *node = make_node (negation_node);
  node->v.arg = p;
  return node;
}

struct node *
make_suffix_node (struct mimetypes_string *suffix)
{
  struct node *node = make_node (suffix_node);
  node->v.suffix = *suffix;
  return node;
}

struct builtin_tab
{
  char *name;
  char *args;
  builtin_t handler;
};

/*        match("pattern")
            Pattern match on filename
*/
static int
b_match (union argument *args)
{
  return fnmatch (args[0].string->ptr, mimeview_file, 0) == 0;
}

/*       ascii(offset,length)
            True if bytes are valid printable ASCII (CR, NL, TAB,
            BS, 32-126)
*/
static int
b_ascii (union argument *args)
{
  int i;
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }

  for (i = 0; i < args[1].number; i++)
    {
      int c = getc (mimeview_fp);
      if (c == EOF)
	break;
      if (!mu_isascii (c))
	return 0;
    }
      
  return 1;
}

/*       printable(offset,length)
            True if bytes are printable 8-bit chars (CR, NL, TAB,
            BS, 32-126, 128-254)
*/
#define ISPRINT(c) ((c) &&\
                    (strchr ("\n\r\t\b",c) \
                     || (32<=(c) && (c)<=126) \
                     || (128<=(c) && (c)<=254)))
static int
b_printable (union argument *args)
{
  int i;

  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }

  for (i = 0; i < args[1].number; i++)
    {
      int c = getc (mimeview_fp);
      if (c == EOF)
	break;
      if (!ISPRINT ((unsigned)c))
	return 0;
    }
  return 1;
}

/*        string(offset,"string")
            True if bytes are identical to string
*/
static int
b_string (union argument *args)
{
  struct mimetypes_string *str = args[1].string;
  int i;
  
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }

  for (i = 0; i < str->len; i++)
    {
      int c = getc (mimeview_fp);
      if (c == EOF || (char)c != str->ptr[i])
	return 0;
    }
  return 1;
}

/*        istring(offset,"string")
            True if a case-insensitive comparison of the bytes is
            identical
*/
static int
b_istring (union argument *args)
{
  int i;
  struct mimetypes_string *str = args[1].string;
  
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }

  for (i = 0; i < str->len; i++)
    {
      int c = getc (mimeview_fp);
      if (c == EOF || mu_tolower (c) != mu_tolower (str->ptr[i]))
	return 0;
    }
  return 1;
}

/*       char(offset,value)
            True if byte is identical
*/
static int
b_char (union argument *args)
{
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }
  return getc (mimeview_fp) == args[1].number;
}

/*        short(offset,value)
            True if 16-bit integer is identical
	  FIXME: Byte order  
*/
static int
b_short (union argument *args)
{
  unsigned short val;
  int rc;
  
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }
  rc = fread (&val, sizeof val, 1, mimeview_fp);

  if (rc == -1)
    {
      mu_error ("fread: %s", mu_strerror (errno));
      return 0;
    }
  else if (rc == 0)
    return 0;
  return val == args[1].number;
}

/*        int(offset,value)
            True if 32-bit integer is identical
          FIXME: Byte order
*/
static int
b_int (union argument *args)
{
  unsigned int val;
  int rc;
  
  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }
  rc = fread (&val, sizeof val, 1, mimeview_fp);
  if (rc == -1)
    {
      mu_error ("fread: %s", mu_strerror (errno));
      return 0;
    }
  else if (rc == 0)
    return 0;
  return val == args[1].number;
}

/*        locale("string")
            True if current locale matches string
*/
static int
b_locale (union argument *args)
{
  abort (); /* FIXME */
  return 0;
}

/*        contains(offset,range,"string")
            True if the range contains the string
*/
static int
b_contains (union argument *args)
{
  int i, count;
  char *buf;
  struct mimetypes_string *str = args[2].string;

  if (fseek (mimeview_fp, args[0].number, SEEK_SET) == -1)
    {
      mu_error ("fseek: %s", mu_strerror (errno));
      return 0;
    }
  buf = xmalloc (args[1].number);
  count = fread (buf, 1, args[1].number, mimeview_fp);
  if (count == -1)
    {
      mu_error ("fread: %s", mu_strerror (errno));
    }
  else if (count > str->len)
    for (i = 0; i < count - str->len; i++)
      if (buf[i] == str->ptr[0] && memcmp (buf + i, str->ptr, str->len) == 0)
	{
	  free (buf);
	  return 1;
	}
  free (buf);
  return 0;
}

static struct builtin_tab builtin_tab[] = {
  { "match", "s", b_match },
  { "ascii", "dd", b_ascii },
  { "printable", "dd", b_printable },
  { "string", "ds", b_string },
  { "istring", "ds", b_istring },
  { "char", "dc", b_char },
  { "short", "dd", b_short },
  { "int", "dd", b_int },
  { "locale", "s", b_locale },
  { "contains", "dds", b_contains },
  { NULL }
};
  
struct node *
make_functional_node (char *ident, mu_list_t list)
{
  size_t count, i;
  struct builtin_tab *p;
  struct node *node;
  union argument *args;
  mu_iterator_t itr;
  
  for (p = builtin_tab; ; p++)
    {
      if (!p->name)
	{
	  char *s;
	  asprintf (&s, _("%s: unknown function"), ident);
	  mimetypes_yyerror (s);
	  free (s);
	  return NULL;
	}

      if (strcmp (ident, p->name) == 0)
	break;
    }

  mu_list_count (list, &count);
  i = strlen (p->args);

  if (count < i)
    {
      char *s;
      asprintf (&s, _("too few arguments in call to `%s'"), ident);
      mimetypes_yyerror (s);
      free (s);
      return NULL;
    }
  else if (count > i)
    {
      char *s;
      asprintf (&s, _("too many arguments in call to `%s'"), ident);
      mimetypes_yyerror (s);
      free (s);
      return NULL;
    }

  args = mimetypes_malloc (count * sizeof *args);
  
  mu_list_get_iterator (list, &itr);
  for (i = 0, mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr), i++)
    {
      struct mimetypes_string *data;
      char *tmp;
      
      mu_iterator_current (itr, (void **)&data);
      switch (p->args[i])
	{
	case 'd':
	  args[i].number = strtoul (data->ptr, &tmp, 0);
	  if (*tmp)
	    goto err;
	  break;
	  
	case 's':
	  args[i].string = data;
	  break;
	  
	case 'c':
	  args[i].c = strtoul (data->ptr, &tmp, 0);
	  if (*tmp)
	    goto err;
	  break;
	  
	default:
	  abort ();
	}
    }

  node = make_node (functional_node);
  node->v.function.fun = p->handler;
  node->v.function.args = args;
  return node;
  
 err:
  {
    char *s;
    asprintf (&s,
	      _("argument %lu has wrong type in call to `%s'"),
	      (unsigned long) i, ident);
    mimetypes_yyerror (s);
    free (s);
    return NULL;
  }
}

static int
check_suffix (char *suf)
{
  char *p = strrchr (mimeview_file, '.');
  if (!p)
    return 0;
  return strcmp (p+1, suf) == 0;
}

static int
eval_rule (struct node *root)
{
  int result;
  
  switch (root->type)
    {
    case functional_node:
      result = root->v.function.fun (root->v.function.args);
      break;
      
    case binary_node:
      result = eval_rule (root->v.bin.arg1);
      switch (root->v.bin.op)
	{
	case L_OR:
	  if (!result)
	    result |= eval_rule (root->v.bin.arg2);
	  break;
	  
	case L_AND:
	  if (result)
	    result &= eval_rule (root->v.bin.arg2);
	  break;
	  
	default:
	  abort ();
	}
      break;
      
    case negation_node:
      result = !eval_rule (root->v.arg);
      break;
      
    case suffix_node:
      result = check_suffix (root->v.suffix.ptr);
      break;

    default:
      abort ();
    }
  return result;
}

static int
evaluate (void *item, void *data)
{
  struct rule_tab *p = item;
  char **ptype = data;
    
  if (eval_rule (p->node))
    {
      *ptype = p->type;
      return 1;
    }
  return 0;
}

const char *
get_file_type ()
{
  const char *type = NULL;
  mu_list_do (rule_list, evaluate, &type);
  return type;
}
    

