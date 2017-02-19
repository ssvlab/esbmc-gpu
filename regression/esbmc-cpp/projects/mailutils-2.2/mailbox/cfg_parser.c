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

/* All symbols defined below should begin with mu_cfg_yy or YY, to avoid
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
   enum mu_cfg_yytokentype {
     MU_TOK_IDENT = 258,
     MU_TOK_STRING = 259,
     MU_TOK_QSTRING = 260,
     MU_TOK_MSTRING = 261
   };
#endif
/* Tokens.  */
#define MU_TOK_IDENT 258
#define MU_TOK_STRING 259
#define MU_TOK_QSTRING 260
#define MU_TOK_MSTRING 261




/* Copy the first part of user declarations.  */
#line 1 "cfg_parser.y"

/* cfg_parser.y -- general-purpose configuration file parser
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 3, or (at
   your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <netdb.h>
#include "intprops.h"
#include <mailutils/argcv.h>
#include <mailutils/nls.h>
#include <mailutils/cfg.h>
#include <mailutils/alloc.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/list.h>
#include <mailutils/iterator.h>
#include <mailutils/debug.h>
#include <mailutils/mutil.h>  

int mu_cfg_parser_verbose;
static mu_list_t /* of mu_cfg_node_t */ parse_node_list; 
mu_cfg_locus_t mu_cfg_locus;
size_t mu_cfg_error_count;

static int _mu_cfg_errcnt;
static mu_debug_t _mu_cfg_debug;

int mu_cfg_yylex ();

void _mu_line_begin (void);
void _mu_line_add (char *text, size_t len);
char *_mu_line_finish (void);

static int
mu_cfg_yyerror (char *s)
{
  mu_cfg_parse_error ("%s", s);
  return 0;
}

static mu_config_value_t *
config_value_dup (mu_config_value_t *src)
{
  if (!src)
    return NULL;
  else
    {
      /* FIXME: Use mu_opool_alloc */
      mu_config_value_t *val = mu_alloc (sizeof (*val));
      *val = *src;
      return val;
    }
}

static mu_cfg_node_t *
mu_cfg_alloc_node (enum mu_cfg_node_type type, mu_cfg_locus_t *loc,
		   const char *tag, mu_config_value_t *label,
		   mu_list_t nodelist)
{
  char *p;
  mu_cfg_node_t *np;
  size_t size = sizeof *np + strlen (tag) + 1;
  np = mu_alloc (size);
  np->type = type;
  np->locus = *loc;
  p = (char*) (np + 1);
  np->tag = p;
  strcpy (p, tag);
  np->label = label;
  np->nodes = nodelist;
  return np;
}

void
mu_cfg_free_node (mu_cfg_node_t *node)
{
  free (node->label);
  free (node);
}

void
mu_cfg_format_error (mu_debug_t debug, size_t level, const char *fmt, ...)
{
  va_list ap;

  if (!debug)
    mu_diag_get_debug (&debug);
  va_start (ap, fmt);
  mu_debug_vprintf (debug, 0, fmt, ap);
  mu_debug_printf (debug, 0, "\n");
  va_end (ap);
  if (level <= MU_DEBUG_ERROR)
    mu_cfg_error_count++;
}

static void
_mu_cfg_debug_set_locus (mu_debug_t debug, const mu_cfg_locus_t *loc)
{
  mu_debug_set_locus (debug, loc->file ? loc->file : _("unknown file"),
		      loc->line);
}

void
mu_cfg_vperror (mu_debug_t debug, const mu_cfg_locus_t *loc,
		 const char *fmt, va_list ap)
{
  if (!debug)
    mu_diag_get_debug (&debug);
  _mu_cfg_debug_set_locus (debug, loc);
  mu_debug_vprintf (debug, 0, fmt, ap);
  mu_debug_printf (debug, 0, "\n");
  mu_debug_set_locus (debug, NULL, 0);
  mu_cfg_error_count++;
}

void
mu_cfg_perror (mu_debug_t debug, const mu_cfg_locus_t *loc,
	       const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  mu_cfg_vperror (debug, loc, fmt, ap);
  va_end (ap);
}

void
mu_cfg_parse_error (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  mu_cfg_vperror (_mu_cfg_debug, &mu_cfg_locus, fmt, ap);
  va_end (ap);
}

#define node_type_str(t) (((t) == mu_cfg_node_statement) ? "stmt" : "param")

static void
debug_print_node (mu_cfg_node_t *node)
{
  if (mu_debug_check_level (_mu_cfg_debug, MU_DEBUG_TRACE0))
    {
      mu_debug_set_locus (_mu_cfg_debug,
			  node->locus.file, node->locus.line);
      if (node->type == mu_cfg_node_undefined)
	/* Stay on the safe side */
	mu_cfg_format_error (_mu_cfg_debug, MU_DEBUG_ERROR,
			     "unknown statement type!");
      else
	/* FIXME: How to print label? */
	mu_cfg_format_error (_mu_cfg_debug, MU_DEBUG_TRACE0,
			     "statement: %s, id: %s",
			     node_type_str (node->type),
			     node->tag ? node->tag : "(null)");

      mu_debug_set_locus (_mu_cfg_debug, NULL, 0);
    }
}

static void
free_node_item (void *item)
{
  mu_cfg_node_t *node = item;

  switch (node->type)
    {
    case mu_cfg_node_statement:
      mu_list_destroy (&node->nodes);
      break;
      
    case mu_cfg_node_undefined: /* hmm... */
    case mu_cfg_node_param:
      break;
    }
  mu_cfg_free_node (node);
}

int
mu_cfg_create_node_list (mu_list_t *plist)
{
  int rc;
  mu_list_t list;

  rc = mu_list_create (&list);
  if (rc)
    return rc;
  mu_list_set_destroy_item (list, free_node_item);
  *plist = list;
  return 0;
}



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
#line 215 "cfg_parser.y"
{
  mu_cfg_node_t node;
  mu_cfg_node_t *pnode;
  mu_list_t /* of mu_cfg_node_t */ nodelist;
  char *string;
  mu_config_value_t value, *pvalue;
  mu_list_t list;
  struct { const char *name; mu_cfg_locus_t locus; } ident;
}
/* Line 187 of yacc.c.  */
#line 332 "cfg_parser.c"
	YYSTYPE;
# define mu_cfg_yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 345 "cfg_parser.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 mu_cfg_yytype_uint8;
#else
typedef unsigned char mu_cfg_yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 mu_cfg_yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char mu_cfg_yytype_int8;
#else
typedef short int mu_cfg_yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 mu_cfg_yytype_uint16;
#else
typedef unsigned short int mu_cfg_yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 mu_cfg_yytype_int16;
#else
typedef short int mu_cfg_yytype_int16;
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

#if ! defined mu_cfg_yyoverflow || YYERROR_VERBOSE

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
#endif /* ! defined mu_cfg_yyoverflow || YYERROR_VERBOSE */


#if (! defined mu_cfg_yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union mu_cfg_yyalloc
{
  mu_cfg_yytype_int16 mu_cfg_yyss;
  YYSTYPE mu_cfg_yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union mu_cfg_yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (mu_cfg_yytype_int16) + sizeof (YYSTYPE)) \
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
	  YYSIZE_T mu_cfg_yyi;				\
	  for (mu_cfg_yyi = 0; mu_cfg_yyi < (Count); mu_cfg_yyi++)	\
	    (To)[mu_cfg_yyi] = (From)[mu_cfg_yyi];		\
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
	YYSIZE_T mu_cfg_yynewbytes;						\
	YYCOPY (&mu_cfg_yyptr->Stack, Stack, mu_cfg_yysize);				\
	Stack = &mu_cfg_yyptr->Stack;						\
	mu_cfg_yynewbytes = mu_cfg_yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	mu_cfg_yyptr += mu_cfg_yynewbytes / sizeof (*mu_cfg_yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  8
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   29

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  13
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  17
/* YYNRULES -- Number of rules.  */
#define YYNRULES  30
/* YYNRULES -- Number of states.  */
#define YYNSTATES  39

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   261

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? mu_cfg_yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const mu_cfg_yytype_uint8 mu_cfg_yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      10,    11,     2,     2,    12,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     7,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     8,     2,     9,     2,     2,     2,     2,
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
       5,     6
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const mu_cfg_yytype_uint8 mu_cfg_yyprhs[] =
{
       0,     0,     3,     5,     7,    10,    12,    14,    18,    24,
      31,    33,    34,    36,    38,    40,    43,    45,    47,    49,
      51,    53,    55,    57,    59,    62,    66,    71,    73,    77,
      78
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const mu_cfg_yytype_int8 mu_cfg_yyrhs[] =
{
      14,     0,    -1,    15,    -1,    16,    -1,    15,    16,    -1,
      17,    -1,    18,    -1,    19,    21,     7,    -1,    19,    20,
       8,     9,    29,    -1,    19,    20,     8,    15,     9,    29,
      -1,     3,    -1,    -1,    21,    -1,    22,    -1,    23,    -1,
      22,    23,    -1,    24,    -1,    27,    -1,     6,    -1,     4,
      -1,     3,    -1,    25,    -1,    26,    -1,     5,    -1,    26,
       5,    -1,    10,    28,    11,    -1,    10,    28,    12,    11,
      -1,    23,    -1,    28,    12,    23,    -1,    -1,     7,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const mu_cfg_yytype_uint16 mu_cfg_yyrline[] =
{
       0,   237,   237,   243,   248,   256,   257,   260,   268,   275,
     283,   291,   294,   297,   332,   343,   349,   354,   359,   366,
     367,   368,   371,   390,   395,   402,   406,   412,   417,   424,
     425
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const mu_cfg_yytname[] =
{
  "$end", "error", "$undefined", "MU_TOK_IDENT", "MU_TOK_STRING",
  "MU_TOK_QSTRING", "MU_TOK_MSTRING", "';'", "'{'", "'}'", "'('", "')'",
  "','", "$accept", "input", "stmtlist", "stmt", "simple", "block",
  "ident", "tag", "vallist", "vlist", "value", "string", "slist", "slist0",
  "list", "values", "opt_sc", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const mu_cfg_yytype_uint16 mu_cfg_yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,    59,   123,   125,
      40,    41,    44
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const mu_cfg_yytype_uint8 mu_cfg_yyr1[] =
{
       0,    13,    14,    15,    15,    16,    16,    17,    18,    18,
      19,    20,    20,    21,    22,    22,    23,    23,    23,    24,
      24,    24,    25,    26,    26,    27,    27,    28,    28,    29,
      29
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const mu_cfg_yytype_uint8 mu_cfg_yyr2[] =
{
       0,     2,     1,     1,     2,     1,     1,     3,     5,     6,
       1,     0,     1,     1,     1,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     3,     4,     1,     3,     0,
       1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const mu_cfg_yytype_uint8 mu_cfg_yydefact[] =
{
       0,    10,     0,     2,     3,     5,     6,    11,     1,     4,
      20,    19,    23,    18,     0,     0,    12,    13,    14,    16,
      21,    22,    17,    27,     0,     0,     7,    15,    24,    25,
       0,    29,     0,    26,    28,    30,     8,    29,     9
};

/* YYDEFGOTO[NTERM-NUM].  */
static const mu_cfg_yytype_int8 mu_cfg_yydefgoto[] =
{
      -1,     2,     3,     4,     5,     6,     7,    15,    16,    17,
      18,    19,    20,    21,    22,    24,    36
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -14
static const mu_cfg_yytype_int8 mu_cfg_yypact[] =
{
      -1,   -14,    11,    -1,   -14,   -14,   -14,    15,   -14,   -14,
     -14,   -14,   -14,   -14,    15,    14,    16,    15,   -14,   -14,
     -14,    19,   -14,   -14,     3,     0,   -14,   -14,   -14,   -14,
       2,    20,     7,   -14,   -14,   -14,   -14,    20,   -14
};

/* YYPGOTO[NTERM-NUM].  */
static const mu_cfg_yytype_int8 mu_cfg_yypgoto[] =
{
     -14,   -14,     1,    -3,   -14,   -14,   -14,   -14,   -14,   -14,
     -13,   -14,   -14,   -14,   -14,   -14,    -9
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const mu_cfg_yytype_uint8 mu_cfg_yytable[] =
{
       9,    23,     1,     1,    27,    10,    11,    12,    13,    31,
       1,     8,    14,    33,    29,    30,    37,    34,    10,    11,
      12,    13,    25,    26,    28,    14,    32,    35,    38,     9
};

static const mu_cfg_yytype_uint8 mu_cfg_yycheck[] =
{
       3,    14,     3,     3,    17,     3,     4,     5,     6,     9,
       3,     0,    10,    11,    11,    12,     9,    30,     3,     4,
       5,     6,     8,     7,     5,    10,    25,     7,    37,    32
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const mu_cfg_yytype_uint8 mu_cfg_yystos[] =
{
       0,     3,    14,    15,    16,    17,    18,    19,     0,    16,
       3,     4,     5,     6,    10,    20,    21,    22,    23,    24,
      25,    26,    27,    23,    28,     8,     7,    23,     5,    11,
      12,     9,    15,    11,    23,     7,    29,     9,    29
};

#define mu_cfg_yyerrok		(mu_cfg_yyerrstatus = 0)
#define mu_cfg_yyclearin	(mu_cfg_yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto mu_cfg_yyacceptlab
#define YYABORT		goto mu_cfg_yyabortlab
#define YYERROR		goto mu_cfg_yyerrorlab


/* Like YYERROR except do call mu_cfg_yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto mu_cfg_yyerrlab

#define YYRECOVERING()  (!!mu_cfg_yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (mu_cfg_yychar == YYEMPTY && mu_cfg_yylen == 1)				\
    {								\
      mu_cfg_yychar = (Token);						\
      mu_cfg_yylval = (Value);						\
      mu_cfg_yytoken = YYTRANSLATE (mu_cfg_yychar);				\
      YYPOPSTACK (1);						\
      goto mu_cfg_yybackup;						\
    }								\
  else								\
    {								\
      mu_cfg_yyerror (YY_("syntax error: cannot back up")); \
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


/* YYLEX -- calling `mu_cfg_yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX mu_cfg_yylex (YYLEX_PARAM)
#else
# define YYLEX mu_cfg_yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (mu_cfg_yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (mu_cfg_yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      mu_cfg_yy_symbol_print (stderr,						  \
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
mu_cfg_yy_symbol_value_print (FILE *mu_cfg_yyoutput, int mu_cfg_yytype, YYSTYPE const * const mu_cfg_yyvaluep)
#else
static void
mu_cfg_yy_symbol_value_print (mu_cfg_yyoutput, mu_cfg_yytype, mu_cfg_yyvaluep)
    FILE *mu_cfg_yyoutput;
    int mu_cfg_yytype;
    YYSTYPE const * const mu_cfg_yyvaluep;
#endif
{
  if (!mu_cfg_yyvaluep)
    return;
# ifdef YYPRINT
  if (mu_cfg_yytype < YYNTOKENS)
    YYPRINT (mu_cfg_yyoutput, mu_cfg_yytoknum[mu_cfg_yytype], *mu_cfg_yyvaluep);
# else
  YYUSE (mu_cfg_yyoutput);
# endif
  switch (mu_cfg_yytype)
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
mu_cfg_yy_symbol_print (FILE *mu_cfg_yyoutput, int mu_cfg_yytype, YYSTYPE const * const mu_cfg_yyvaluep)
#else
static void
mu_cfg_yy_symbol_print (mu_cfg_yyoutput, mu_cfg_yytype, mu_cfg_yyvaluep)
    FILE *mu_cfg_yyoutput;
    int mu_cfg_yytype;
    YYSTYPE const * const mu_cfg_yyvaluep;
#endif
{
  if (mu_cfg_yytype < YYNTOKENS)
    YYFPRINTF (mu_cfg_yyoutput, "token %s (", mu_cfg_yytname[mu_cfg_yytype]);
  else
    YYFPRINTF (mu_cfg_yyoutput, "nterm %s (", mu_cfg_yytname[mu_cfg_yytype]);

  mu_cfg_yy_symbol_value_print (mu_cfg_yyoutput, mu_cfg_yytype, mu_cfg_yyvaluep);
  YYFPRINTF (mu_cfg_yyoutput, ")");
}

/*------------------------------------------------------------------.
| mu_cfg_yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
mu_cfg_yy_stack_print (mu_cfg_yytype_int16 *bottom, mu_cfg_yytype_int16 *top)
#else
static void
mu_cfg_yy_stack_print (bottom, top)
    mu_cfg_yytype_int16 *bottom;
    mu_cfg_yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (mu_cfg_yydebug)							\
    mu_cfg_yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
mu_cfg_yy_reduce_print (YYSTYPE *mu_cfg_yyvsp, int mu_cfg_yyrule)
#else
static void
mu_cfg_yy_reduce_print (mu_cfg_yyvsp, mu_cfg_yyrule)
    YYSTYPE *mu_cfg_yyvsp;
    int mu_cfg_yyrule;
#endif
{
  int mu_cfg_yynrhs = mu_cfg_yyr2[mu_cfg_yyrule];
  int mu_cfg_yyi;
  unsigned long int mu_cfg_yylno = mu_cfg_yyrline[mu_cfg_yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     mu_cfg_yyrule - 1, mu_cfg_yylno);
  /* The symbols being reduced.  */
  for (mu_cfg_yyi = 0; mu_cfg_yyi < mu_cfg_yynrhs; mu_cfg_yyi++)
    {
      fprintf (stderr, "   $%d = ", mu_cfg_yyi + 1);
      mu_cfg_yy_symbol_print (stderr, mu_cfg_yyrhs[mu_cfg_yyprhs[mu_cfg_yyrule] + mu_cfg_yyi],
		       &(mu_cfg_yyvsp[(mu_cfg_yyi + 1) - (mu_cfg_yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (mu_cfg_yydebug)				\
    mu_cfg_yy_reduce_print (mu_cfg_yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int mu_cfg_yydebug;
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

# ifndef mu_cfg_yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define mu_cfg_yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
mu_cfg_yystrlen (const char *mu_cfg_yystr)
#else
static YYSIZE_T
mu_cfg_yystrlen (mu_cfg_yystr)
    const char *mu_cfg_yystr;
#endif
{
  YYSIZE_T mu_cfg_yylen;
  for (mu_cfg_yylen = 0; mu_cfg_yystr[mu_cfg_yylen]; mu_cfg_yylen++)
    continue;
  return mu_cfg_yylen;
}
#  endif
# endif

# ifndef mu_cfg_yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define mu_cfg_yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
mu_cfg_yystpcpy (char *mu_cfg_yydest, const char *mu_cfg_yysrc)
#else
static char *
mu_cfg_yystpcpy (mu_cfg_yydest, mu_cfg_yysrc)
    char *mu_cfg_yydest;
    const char *mu_cfg_yysrc;
#endif
{
  char *mu_cfg_yyd = mu_cfg_yydest;
  const char *mu_cfg_yys = mu_cfg_yysrc;

  while ((*mu_cfg_yyd++ = *mu_cfg_yys++) != '\0')
    continue;

  return mu_cfg_yyd - 1;
}
#  endif
# endif

# ifndef mu_cfg_yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for mu_cfg_yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from mu_cfg_yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
mu_cfg_yytnamerr (char *mu_cfg_yyres, const char *mu_cfg_yystr)
{
  if (*mu_cfg_yystr == '"')
    {
      YYSIZE_T mu_cfg_yyn = 0;
      char const *mu_cfg_yyp = mu_cfg_yystr;

      for (;;)
	switch (*++mu_cfg_yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++mu_cfg_yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (mu_cfg_yyres)
	      mu_cfg_yyres[mu_cfg_yyn] = *mu_cfg_yyp;
	    mu_cfg_yyn++;
	    break;

	  case '"':
	    if (mu_cfg_yyres)
	      mu_cfg_yyres[mu_cfg_yyn] = '\0';
	    return mu_cfg_yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! mu_cfg_yyres)
    return mu_cfg_yystrlen (mu_cfg_yystr);

  return mu_cfg_yystpcpy (mu_cfg_yyres, mu_cfg_yystr) - mu_cfg_yyres;
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
mu_cfg_yysyntax_error (char *mu_cfg_yyresult, int mu_cfg_yystate, int mu_cfg_yychar)
{
  int mu_cfg_yyn = mu_cfg_yypact[mu_cfg_yystate];

  if (! (YYPACT_NINF < mu_cfg_yyn && mu_cfg_yyn <= YYLAST))
    return 0;
  else
    {
      int mu_cfg_yytype = YYTRANSLATE (mu_cfg_yychar);
      YYSIZE_T mu_cfg_yysize0 = mu_cfg_yytnamerr (0, mu_cfg_yytname[mu_cfg_yytype]);
      YYSIZE_T mu_cfg_yysize = mu_cfg_yysize0;
      YYSIZE_T mu_cfg_yysize1;
      int mu_cfg_yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *mu_cfg_yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int mu_cfg_yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *mu_cfg_yyfmt;
      char const *mu_cfg_yyf;
      static char const mu_cfg_yyunexpected[] = "syntax error, unexpected %s";
      static char const mu_cfg_yyexpecting[] = ", expecting %s";
      static char const mu_cfg_yyor[] = " or %s";
      char mu_cfg_yyformat[sizeof mu_cfg_yyunexpected
		    + sizeof mu_cfg_yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof mu_cfg_yyor - 1))];
      char const *mu_cfg_yyprefix = mu_cfg_yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int mu_cfg_yyxbegin = mu_cfg_yyn < 0 ? -mu_cfg_yyn : 0;

      /* Stay within bounds of both mu_cfg_yycheck and mu_cfg_yytname.  */
      int mu_cfg_yychecklim = YYLAST - mu_cfg_yyn + 1;
      int mu_cfg_yyxend = mu_cfg_yychecklim < YYNTOKENS ? mu_cfg_yychecklim : YYNTOKENS;
      int mu_cfg_yycount = 1;

      mu_cfg_yyarg[0] = mu_cfg_yytname[mu_cfg_yytype];
      mu_cfg_yyfmt = mu_cfg_yystpcpy (mu_cfg_yyformat, mu_cfg_yyunexpected);

      for (mu_cfg_yyx = mu_cfg_yyxbegin; mu_cfg_yyx < mu_cfg_yyxend; ++mu_cfg_yyx)
	if (mu_cfg_yycheck[mu_cfg_yyx + mu_cfg_yyn] == mu_cfg_yyx && mu_cfg_yyx != YYTERROR)
	  {
	    if (mu_cfg_yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		mu_cfg_yycount = 1;
		mu_cfg_yysize = mu_cfg_yysize0;
		mu_cfg_yyformat[sizeof mu_cfg_yyunexpected - 1] = '\0';
		break;
	      }
	    mu_cfg_yyarg[mu_cfg_yycount++] = mu_cfg_yytname[mu_cfg_yyx];
	    mu_cfg_yysize1 = mu_cfg_yysize + mu_cfg_yytnamerr (0, mu_cfg_yytname[mu_cfg_yyx]);
	    mu_cfg_yysize_overflow |= (mu_cfg_yysize1 < mu_cfg_yysize);
	    mu_cfg_yysize = mu_cfg_yysize1;
	    mu_cfg_yyfmt = mu_cfg_yystpcpy (mu_cfg_yyfmt, mu_cfg_yyprefix);
	    mu_cfg_yyprefix = mu_cfg_yyor;
	  }

      mu_cfg_yyf = YY_(mu_cfg_yyformat);
      mu_cfg_yysize1 = mu_cfg_yysize + mu_cfg_yystrlen (mu_cfg_yyf);
      mu_cfg_yysize_overflow |= (mu_cfg_yysize1 < mu_cfg_yysize);
      mu_cfg_yysize = mu_cfg_yysize1;

      if (mu_cfg_yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (mu_cfg_yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *mu_cfg_yyp = mu_cfg_yyresult;
	  int mu_cfg_yyi = 0;
	  while ((*mu_cfg_yyp = *mu_cfg_yyf) != '\0')
	    {
	      if (*mu_cfg_yyp == '%' && mu_cfg_yyf[1] == 's' && mu_cfg_yyi < mu_cfg_yycount)
		{
		  mu_cfg_yyp += mu_cfg_yytnamerr (mu_cfg_yyp, mu_cfg_yyarg[mu_cfg_yyi++]);
		  mu_cfg_yyf += 2;
		}
	      else
		{
		  mu_cfg_yyp++;
		  mu_cfg_yyf++;
		}
	    }
	}
      return mu_cfg_yysize;
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
mu_cfg_yydestruct (const char *mu_cfg_yymsg, int mu_cfg_yytype, YYSTYPE *mu_cfg_yyvaluep)
#else
static void
mu_cfg_yydestruct (mu_cfg_yymsg, mu_cfg_yytype, mu_cfg_yyvaluep)
    const char *mu_cfg_yymsg;
    int mu_cfg_yytype;
    YYSTYPE *mu_cfg_yyvaluep;
#endif
{
  YYUSE (mu_cfg_yyvaluep);

  if (!mu_cfg_yymsg)
    mu_cfg_yymsg = "Deleting";
  YY_SYMBOL_PRINT (mu_cfg_yymsg, mu_cfg_yytype, mu_cfg_yyvaluep, mu_cfg_yylocationp);

  switch (mu_cfg_yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int mu_cfg_yyparse (void *YYPARSE_PARAM);
#else
int mu_cfg_yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int mu_cfg_yyparse (void);
#else
int mu_cfg_yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int mu_cfg_yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE mu_cfg_yylval;

/* Number of syntax errors so far.  */
int mu_cfg_yynerrs;



/*----------.
| mu_cfg_yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
mu_cfg_yyparse (void *YYPARSE_PARAM)
#else
int
mu_cfg_yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
mu_cfg_yyparse (void)
#else
int
mu_cfg_yyparse ()

#endif
#endif
{
  
  int mu_cfg_yystate;
  int mu_cfg_yyn;
  int mu_cfg_yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int mu_cfg_yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int mu_cfg_yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char mu_cfg_yymsgbuf[128];
  char *mu_cfg_yymsg = mu_cfg_yymsgbuf;
  YYSIZE_T mu_cfg_yymsg_alloc = sizeof mu_cfg_yymsgbuf;
#endif

  /* Three stacks and their tools:
     `mu_cfg_yyss': related to states,
     `mu_cfg_yyvs': related to semantic values,
     `mu_cfg_yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow mu_cfg_yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  mu_cfg_yytype_int16 mu_cfg_yyssa[YYINITDEPTH];
  mu_cfg_yytype_int16 *mu_cfg_yyss = mu_cfg_yyssa;
  mu_cfg_yytype_int16 *mu_cfg_yyssp;

  /* The semantic value stack.  */
  YYSTYPE mu_cfg_yyvsa[YYINITDEPTH];
  YYSTYPE *mu_cfg_yyvs = mu_cfg_yyvsa;
  YYSTYPE *mu_cfg_yyvsp;



#define YYPOPSTACK(N)   (mu_cfg_yyvsp -= (N), mu_cfg_yyssp -= (N))

  YYSIZE_T mu_cfg_yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE mu_cfg_yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int mu_cfg_yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  mu_cfg_yystate = 0;
  mu_cfg_yyerrstatus = 0;
  mu_cfg_yynerrs = 0;
  mu_cfg_yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  mu_cfg_yyssp = mu_cfg_yyss;
  mu_cfg_yyvsp = mu_cfg_yyvs;

  goto mu_cfg_yysetstate;

/*------------------------------------------------------------.
| mu_cfg_yynewstate -- Push a new state, which is found in mu_cfg_yystate.  |
`------------------------------------------------------------*/
 mu_cfg_yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  mu_cfg_yyssp++;

 mu_cfg_yysetstate:
  *mu_cfg_yyssp = mu_cfg_yystate;

  if (mu_cfg_yyss + mu_cfg_yystacksize - 1 <= mu_cfg_yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T mu_cfg_yysize = mu_cfg_yyssp - mu_cfg_yyss + 1;

#ifdef mu_cfg_yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *mu_cfg_yyvs1 = mu_cfg_yyvs;
	mu_cfg_yytype_int16 *mu_cfg_yyss1 = mu_cfg_yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if mu_cfg_yyoverflow is a macro.  */
	mu_cfg_yyoverflow (YY_("memory exhausted"),
		    &mu_cfg_yyss1, mu_cfg_yysize * sizeof (*mu_cfg_yyssp),
		    &mu_cfg_yyvs1, mu_cfg_yysize * sizeof (*mu_cfg_yyvsp),

		    &mu_cfg_yystacksize);

	mu_cfg_yyss = mu_cfg_yyss1;
	mu_cfg_yyvs = mu_cfg_yyvs1;
      }
#else /* no mu_cfg_yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto mu_cfg_yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= mu_cfg_yystacksize)
	goto mu_cfg_yyexhaustedlab;
      mu_cfg_yystacksize *= 2;
      if (YYMAXDEPTH < mu_cfg_yystacksize)
	mu_cfg_yystacksize = YYMAXDEPTH;

      {
	mu_cfg_yytype_int16 *mu_cfg_yyss1 = mu_cfg_yyss;
	union mu_cfg_yyalloc *mu_cfg_yyptr =
	  (union mu_cfg_yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (mu_cfg_yystacksize));
	if (! mu_cfg_yyptr)
	  goto mu_cfg_yyexhaustedlab;
	YYSTACK_RELOCATE (mu_cfg_yyss);
	YYSTACK_RELOCATE (mu_cfg_yyvs);

#  undef YYSTACK_RELOCATE
	if (mu_cfg_yyss1 != mu_cfg_yyssa)
	  YYSTACK_FREE (mu_cfg_yyss1);
      }
# endif
#endif /* no mu_cfg_yyoverflow */

      mu_cfg_yyssp = mu_cfg_yyss + mu_cfg_yysize - 1;
      mu_cfg_yyvsp = mu_cfg_yyvs + mu_cfg_yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) mu_cfg_yystacksize));

      if (mu_cfg_yyss + mu_cfg_yystacksize - 1 <= mu_cfg_yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", mu_cfg_yystate));

  goto mu_cfg_yybackup;

/*-----------.
| mu_cfg_yybackup.  |
`-----------*/
mu_cfg_yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  mu_cfg_yyn = mu_cfg_yypact[mu_cfg_yystate];
  if (mu_cfg_yyn == YYPACT_NINF)
    goto mu_cfg_yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (mu_cfg_yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      mu_cfg_yychar = YYLEX;
    }

  if (mu_cfg_yychar <= YYEOF)
    {
      mu_cfg_yychar = mu_cfg_yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      mu_cfg_yytoken = YYTRANSLATE (mu_cfg_yychar);
      YY_SYMBOL_PRINT ("Next token is", mu_cfg_yytoken, &mu_cfg_yylval, &mu_cfg_yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  mu_cfg_yyn += mu_cfg_yytoken;
  if (mu_cfg_yyn < 0 || YYLAST < mu_cfg_yyn || mu_cfg_yycheck[mu_cfg_yyn] != mu_cfg_yytoken)
    goto mu_cfg_yydefault;
  mu_cfg_yyn = mu_cfg_yytable[mu_cfg_yyn];
  if (mu_cfg_yyn <= 0)
    {
      if (mu_cfg_yyn == 0 || mu_cfg_yyn == YYTABLE_NINF)
	goto mu_cfg_yyerrlab;
      mu_cfg_yyn = -mu_cfg_yyn;
      goto mu_cfg_yyreduce;
    }

  if (mu_cfg_yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (mu_cfg_yyerrstatus)
    mu_cfg_yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", mu_cfg_yytoken, &mu_cfg_yylval, &mu_cfg_yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (mu_cfg_yychar != YYEOF)
    mu_cfg_yychar = YYEMPTY;

  mu_cfg_yystate = mu_cfg_yyn;
  *++mu_cfg_yyvsp = mu_cfg_yylval;

  goto mu_cfg_yynewstate;


/*-----------------------------------------------------------.
| mu_cfg_yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
mu_cfg_yydefault:
  mu_cfg_yyn = mu_cfg_yydefact[mu_cfg_yystate];
  if (mu_cfg_yyn == 0)
    goto mu_cfg_yyerrlab;
  goto mu_cfg_yyreduce;


/*-----------------------------.
| mu_cfg_yyreduce -- Do a reduction.  |
`-----------------------------*/
mu_cfg_yyreduce:
  /* mu_cfg_yyn is the number of a rule to reduce with.  */
  mu_cfg_yylen = mu_cfg_yyr2[mu_cfg_yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  mu_cfg_yyval = mu_cfg_yyvsp[1-mu_cfg_yylen];


  YY_REDUCE_PRINT (mu_cfg_yyn);
  switch (mu_cfg_yyn)
    {
        case 2:
#line 238 "cfg_parser.y"
    {
	    parse_node_list = (mu_cfg_yyvsp[(1) - (1)].nodelist);
	  }
    break;

  case 3:
#line 244 "cfg_parser.y"
    {
	    mu_cfg_create_node_list (&(mu_cfg_yyval.nodelist));
	    mu_list_append ((mu_cfg_yyval.nodelist), (mu_cfg_yyvsp[(1) - (1)].pnode));
	  }
    break;

  case 4:
#line 249 "cfg_parser.y"
    {
	    mu_list_append ((mu_cfg_yyvsp[(1) - (2)].nodelist), (mu_cfg_yyvsp[(2) - (2)].pnode));
	    (mu_cfg_yyval.nodelist) = (mu_cfg_yyvsp[(1) - (2)].nodelist);
	    debug_print_node ((mu_cfg_yyvsp[(2) - (2)].pnode));
	  }
    break;

  case 7:
#line 261 "cfg_parser.y"
    {
	    (mu_cfg_yyval.pnode) = mu_cfg_alloc_node (mu_cfg_node_param, &(mu_cfg_yyvsp[(1) - (3)].ident).locus,
				    (mu_cfg_yyvsp[(1) - (3)].ident).name, (mu_cfg_yyvsp[(2) - (3)].pvalue),
				    NULL);
	  }
    break;

  case 8:
#line 269 "cfg_parser.y"
    {
	    (mu_cfg_yyval.pnode) = mu_cfg_alloc_node (mu_cfg_node_statement, &(mu_cfg_yyvsp[(1) - (5)].ident).locus,
				    (mu_cfg_yyvsp[(1) - (5)].ident).name, (mu_cfg_yyvsp[(2) - (5)].pvalue),
				    NULL);

	  }
    break;

  case 9:
#line 276 "cfg_parser.y"
    {
	    (mu_cfg_yyval.pnode) = mu_cfg_alloc_node (mu_cfg_node_statement, &(mu_cfg_yyvsp[(1) - (6)].ident).locus,
				    (mu_cfg_yyvsp[(1) - (6)].ident).name, (mu_cfg_yyvsp[(2) - (6)].pvalue), (mu_cfg_yyvsp[(4) - (6)].nodelist));

	  }
    break;

  case 10:
#line 284 "cfg_parser.y"
    {
	    (mu_cfg_yyval.ident).name = (mu_cfg_yyvsp[(1) - (1)].string);
	    (mu_cfg_yyval.ident).locus = mu_cfg_locus;
	  }
    break;

  case 11:
#line 291 "cfg_parser.y"
    {
	    (mu_cfg_yyval.pvalue) = NULL;
	  }
    break;

  case 13:
#line 298 "cfg_parser.y"
    {
	    size_t n = 0;
	    mu_list_count((mu_cfg_yyvsp[(1) - (1)].list), &n);
	    if (n == 1)
	      {
		mu_list_get ((mu_cfg_yyvsp[(1) - (1)].list), 0, (void**) &(mu_cfg_yyval.pvalue));
	      }
	    else
	      {
		size_t i;
		mu_config_value_t val;

		val.type = MU_CFG_ARRAY;
		val.v.arg.c = n;
		/* FIXME: Use mu_opool_alloc */
		val.v.arg.v = mu_alloc (n * sizeof (val.v.arg.v[0]));
		if (!val.v.arg.v)
		  {
		    mu_cfg_parse_error (_("not enough memory"));
		    abort();
		  }

		for (i = 0; i < n; i++)
		  {
		    mu_config_value_t *v;
		    mu_list_get ((mu_cfg_yyvsp[(1) - (1)].list), i, (void **) &v);
		    val.v.arg.v[i] = *v;
		  }
		(mu_cfg_yyval.pvalue) = config_value_dup (&val);
	      }
	    mu_list_destroy (&(mu_cfg_yyvsp[(1) - (1)].list));
	  }
    break;

  case 14:
#line 333 "cfg_parser.y"
    {
	      int rc = mu_list_create (&(mu_cfg_yyval.list));
	      if (rc)
		{
		  mu_cfg_parse_error (_("cannot create list: %s"),
				      mu_strerror (rc));
		  abort ();
		}
	      mu_list_append ((mu_cfg_yyval.list), config_value_dup (&(mu_cfg_yyvsp[(1) - (1)].value))); /* FIXME */
	  }
    break;

  case 15:
#line 344 "cfg_parser.y"
    {
	    mu_list_append ((mu_cfg_yyvsp[(1) - (2)].list), config_value_dup (&(mu_cfg_yyvsp[(2) - (2)].value)));
	  }
    break;

  case 16:
#line 350 "cfg_parser.y"
    {
	      (mu_cfg_yyval.value).type = MU_CFG_STRING;
	      (mu_cfg_yyval.value).v.string = (mu_cfg_yyvsp[(1) - (1)].string);
	  }
    break;

  case 17:
#line 355 "cfg_parser.y"
    {
	      (mu_cfg_yyval.value).type = MU_CFG_LIST;
	      (mu_cfg_yyval.value).v.list = (mu_cfg_yyvsp[(1) - (1)].list);
	  }
    break;

  case 18:
#line 360 "cfg_parser.y"
    {
	      (mu_cfg_yyval.value).type = MU_CFG_STRING;
	      (mu_cfg_yyval.value).v.string = (mu_cfg_yyvsp[(1) - (1)].string);
	  }
    break;

  case 22:
#line 372 "cfg_parser.y"
    {
	    mu_iterator_t itr;
	    mu_list_get_iterator ((mu_cfg_yyvsp[(1) - (1)].list), &itr);

	    _mu_line_begin ();
	    for (mu_iterator_first (itr);
		 !mu_iterator_is_done (itr); mu_iterator_next (itr))
	      {
		char *p;
		mu_iterator_current (itr, (void**)&p);
		_mu_line_add (p, strlen (p));
	      }
	    (mu_cfg_yyval.string) = _mu_line_finish ();
	    mu_iterator_destroy (&itr);
	    mu_list_destroy(&(mu_cfg_yyvsp[(1) - (1)].list));
	  }
    break;

  case 23:
#line 391 "cfg_parser.y"
    {
	    mu_list_create (&(mu_cfg_yyval.list));
	    mu_list_append ((mu_cfg_yyval.list), (mu_cfg_yyvsp[(1) - (1)].string));
	  }
    break;

  case 24:
#line 396 "cfg_parser.y"
    {
	    mu_list_append ((mu_cfg_yyvsp[(1) - (2)].list), (mu_cfg_yyvsp[(2) - (2)].string));
	    (mu_cfg_yyval.list) = (mu_cfg_yyvsp[(1) - (2)].list);
	  }
    break;

  case 25:
#line 403 "cfg_parser.y"
    {
	      (mu_cfg_yyval.list) = (mu_cfg_yyvsp[(2) - (3)].list);
	  }
    break;

  case 26:
#line 407 "cfg_parser.y"
    {
	      (mu_cfg_yyval.list) = (mu_cfg_yyvsp[(2) - (4)].list);
	  }
    break;

  case 27:
#line 413 "cfg_parser.y"
    {
	    mu_list_create (&(mu_cfg_yyval.list));
	    mu_list_append ((mu_cfg_yyval.list), config_value_dup (&(mu_cfg_yyvsp[(1) - (1)].value)));
	  }
    break;

  case 28:
#line 418 "cfg_parser.y"
    {
	    mu_list_append ((mu_cfg_yyvsp[(1) - (3)].list), config_value_dup (&(mu_cfg_yyvsp[(3) - (3)].value)));
	    (mu_cfg_yyval.list) = (mu_cfg_yyvsp[(1) - (3)].list);
	  }
    break;


/* Line 1267 of yacc.c.  */
#line 1777 "cfg_parser.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", mu_cfg_yyr1[mu_cfg_yyn], &mu_cfg_yyval, &mu_cfg_yyloc);

  YYPOPSTACK (mu_cfg_yylen);
  mu_cfg_yylen = 0;
  YY_STACK_PRINT (mu_cfg_yyss, mu_cfg_yyssp);

  *++mu_cfg_yyvsp = mu_cfg_yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  mu_cfg_yyn = mu_cfg_yyr1[mu_cfg_yyn];

  mu_cfg_yystate = mu_cfg_yypgoto[mu_cfg_yyn - YYNTOKENS] + *mu_cfg_yyssp;
  if (0 <= mu_cfg_yystate && mu_cfg_yystate <= YYLAST && mu_cfg_yycheck[mu_cfg_yystate] == *mu_cfg_yyssp)
    mu_cfg_yystate = mu_cfg_yytable[mu_cfg_yystate];
  else
    mu_cfg_yystate = mu_cfg_yydefgoto[mu_cfg_yyn - YYNTOKENS];

  goto mu_cfg_yynewstate;


/*------------------------------------.
| mu_cfg_yyerrlab -- here on detecting error |
`------------------------------------*/
mu_cfg_yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!mu_cfg_yyerrstatus)
    {
      ++mu_cfg_yynerrs;
#if ! YYERROR_VERBOSE
      mu_cfg_yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T mu_cfg_yysize = mu_cfg_yysyntax_error (0, mu_cfg_yystate, mu_cfg_yychar);
	if (mu_cfg_yymsg_alloc < mu_cfg_yysize && mu_cfg_yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T mu_cfg_yyalloc = 2 * mu_cfg_yysize;
	    if (! (mu_cfg_yysize <= mu_cfg_yyalloc && mu_cfg_yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      mu_cfg_yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (mu_cfg_yymsg != mu_cfg_yymsgbuf)
	      YYSTACK_FREE (mu_cfg_yymsg);
	    mu_cfg_yymsg = (char *) YYSTACK_ALLOC (mu_cfg_yyalloc);
	    if (mu_cfg_yymsg)
	      mu_cfg_yymsg_alloc = mu_cfg_yyalloc;
	    else
	      {
		mu_cfg_yymsg = mu_cfg_yymsgbuf;
		mu_cfg_yymsg_alloc = sizeof mu_cfg_yymsgbuf;
	      }
	  }

	if (0 < mu_cfg_yysize && mu_cfg_yysize <= mu_cfg_yymsg_alloc)
	  {
	    (void) mu_cfg_yysyntax_error (mu_cfg_yymsg, mu_cfg_yystate, mu_cfg_yychar);
	    mu_cfg_yyerror (mu_cfg_yymsg);
	  }
	else
	  {
	    mu_cfg_yyerror (YY_("syntax error"));
	    if (mu_cfg_yysize != 0)
	      goto mu_cfg_yyexhaustedlab;
	  }
      }
#endif
    }



  if (mu_cfg_yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (mu_cfg_yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (mu_cfg_yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  mu_cfg_yydestruct ("Error: discarding",
		      mu_cfg_yytoken, &mu_cfg_yylval);
	  mu_cfg_yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto mu_cfg_yyerrlab1;


/*---------------------------------------------------.
| mu_cfg_yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
mu_cfg_yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label mu_cfg_yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto mu_cfg_yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (mu_cfg_yylen);
  mu_cfg_yylen = 0;
  YY_STACK_PRINT (mu_cfg_yyss, mu_cfg_yyssp);
  mu_cfg_yystate = *mu_cfg_yyssp;
  goto mu_cfg_yyerrlab1;


/*-------------------------------------------------------------.
| mu_cfg_yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
mu_cfg_yyerrlab1:
  mu_cfg_yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      mu_cfg_yyn = mu_cfg_yypact[mu_cfg_yystate];
      if (mu_cfg_yyn != YYPACT_NINF)
	{
	  mu_cfg_yyn += YYTERROR;
	  if (0 <= mu_cfg_yyn && mu_cfg_yyn <= YYLAST && mu_cfg_yycheck[mu_cfg_yyn] == YYTERROR)
	    {
	      mu_cfg_yyn = mu_cfg_yytable[mu_cfg_yyn];
	      if (0 < mu_cfg_yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (mu_cfg_yyssp == mu_cfg_yyss)
	YYABORT;


      mu_cfg_yydestruct ("Error: popping",
		  mu_cfg_yystos[mu_cfg_yystate], mu_cfg_yyvsp);
      YYPOPSTACK (1);
      mu_cfg_yystate = *mu_cfg_yyssp;
      YY_STACK_PRINT (mu_cfg_yyss, mu_cfg_yyssp);
    }

  if (mu_cfg_yyn == YYFINAL)
    YYACCEPT;

  *++mu_cfg_yyvsp = mu_cfg_yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", mu_cfg_yystos[mu_cfg_yyn], mu_cfg_yyvsp, mu_cfg_yylsp);

  mu_cfg_yystate = mu_cfg_yyn;
  goto mu_cfg_yynewstate;


/*-------------------------------------.
| mu_cfg_yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
mu_cfg_yyacceptlab:
  mu_cfg_yyresult = 0;
  goto mu_cfg_yyreturn;

/*-----------------------------------.
| mu_cfg_yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
mu_cfg_yyabortlab:
  mu_cfg_yyresult = 1;
  goto mu_cfg_yyreturn;

#ifndef mu_cfg_yyoverflow
/*-------------------------------------------------.
| mu_cfg_yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
mu_cfg_yyexhaustedlab:
  mu_cfg_yyerror (YY_("memory exhausted"));
  mu_cfg_yyresult = 2;
  /* Fall through.  */
#endif

mu_cfg_yyreturn:
  if (mu_cfg_yychar != YYEOF && mu_cfg_yychar != YYEMPTY)
     mu_cfg_yydestruct ("Cleanup: discarding lookahead",
		 mu_cfg_yytoken, &mu_cfg_yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (mu_cfg_yylen);
  YY_STACK_PRINT (mu_cfg_yyss, mu_cfg_yyssp);
  while (mu_cfg_yyssp != mu_cfg_yyss)
    {
      mu_cfg_yydestruct ("Cleanup: popping",
		  mu_cfg_yystos[*mu_cfg_yyssp], mu_cfg_yyvsp);
      YYPOPSTACK (1);
    }
#ifndef mu_cfg_yyoverflow
  if (mu_cfg_yyss != mu_cfg_yyssa)
    YYSTACK_FREE (mu_cfg_yyss);
#endif
#if YYERROR_VERBOSE
  if (mu_cfg_yymsg != mu_cfg_yymsgbuf)
    YYSTACK_FREE (mu_cfg_yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (mu_cfg_yyresult);
}


#line 429 "cfg_parser.y"


static int
_cfg_default_printer (void *unused, mu_log_level_t level, const char *str)
{
  fprintf (stderr, "%s", str);
  return 0;
}

mu_debug_t
mu_cfg_get_debug ()
{
  if (!_mu_cfg_debug)
    {
      mu_debug_create (&_mu_cfg_debug, NULL);
      mu_debug_set_print (_mu_cfg_debug, _cfg_default_printer, NULL);
      mu_debug_set_level (_mu_cfg_debug, mu_global_debug_level ("config"));
    }
  return _mu_cfg_debug;
}

void
mu_cfg_set_debug ()
{
  if (mu_debug_check_level (mu_cfg_get_debug (), MU_DEBUG_TRACE7))
    mu_cfg_yydebug = 1;
}

int
mu_cfg_parse (mu_cfg_tree_t **ptree)
{
  int rc;
  mu_cfg_tree_t *tree;
  mu_opool_t pool;
  
  mu_cfg_set_debug ();
  _mu_cfg_errcnt = 0;

  rc = mu_cfg_yyparse ();
  pool = mu_cfg_lexer_pool ();
  if (rc == 0 && _mu_cfg_errcnt)
    {
      mu_opool_destroy (&pool);
      rc = 1;
    }
  else
    {
      tree = mu_alloc (sizeof (*tree));
      tree->debug = _mu_cfg_debug;
      _mu_cfg_debug = NULL;
      tree->nodes = parse_node_list;
      tree->pool = pool;
      parse_node_list = NULL;
      *ptree = tree;
    }
  return rc;
}

int
mu_cfg_tree_union (mu_cfg_tree_t **pa, mu_cfg_tree_t **pb)
{
  mu_cfg_tree_t *a, *b;
  int rc;
  
  if (!pb)
    return EINVAL;
  if (!*pb)
    return 0;
  b = *pb;
  if (!pa)
    return EINVAL;
  if (!*pa)
    {
      *pa = b;
      *pb = NULL;
      return 0;
    }
  else
    a = *pa;
  
  /* Merge opools */
  rc = mu_opool_union (&b->pool, &a->pool);
  if (rc)
    return rc;
    
  /* Link node lists */
  if (b->nodes)
    {
      mu_list_append_list (a->nodes, b->nodes);
      mu_list_destroy (&b->nodes);
    }
  
  mu_debug_destroy (&b->debug, mu_debug_get_owner (b->debug));
  free (b);
  *pb = NULL;
  return 0;
}

static mu_cfg_tree_t *
do_include (const char *name, int flags, mu_cfg_locus_t *loc)
{
  struct stat sb;
  char *tmpname = NULL;
  mu_cfg_tree_t *tree = NULL;
  
  if (name[0] != '/')
    {
      name = tmpname = mu_make_file_name (SYSCONFDIR, name);
      if (!name)
        {
          mu_error ("%s", mu_strerror (errno));
          return NULL;
        }
    }
  if (stat (name, &sb) == 0)
    {
      int rc = 0;

      if (S_ISDIR (sb.st_mode))
	{
	  if (flags & MU_PARSE_CONFIG_GLOBAL)
	    {
	      char *file = mu_make_file_name (name, mu_program_name);
	      rc = mu_cfg_parse_file (&tree, file, flags);
	      free (file);
	    }
	}
      else
	rc = mu_cfg_parse_file (&tree, name, flags);
	      
      if (rc == 0 && tree)
	mu_cfg_tree_postprocess (tree, flags & ~MU_PARSE_CONFIG_GLOBAL);
    }
  else if (errno == ENOENT)
    mu_cfg_perror (tree->debug, loc,
		   _("include file or directory does not exist"));
  else
    mu_cfg_perror (tree->debug, loc,
		   _("cannot stat include file or directory: %s"),
		   mu_strerror (errno));
  free (tmpname);
  return tree;
}
    
int
mu_cfg_tree_postprocess (mu_cfg_tree_t *tree, int flags)
{
  int rc;
  mu_iterator_t itr;

  if (!tree->nodes)
    return 0;
  rc = mu_list_get_iterator (tree->nodes, &itr);
  if (rc)
    return rc;
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      mu_cfg_node_t *node;

      mu_iterator_current (itr, (void**) &node);

      if (node->type == mu_cfg_node_statement)
	{
	  if ((flags & MU_PARSE_CONFIG_GLOBAL) &&
	      strcmp (node->tag, "program") == 0)
	    {
	      if (node->label->type == MU_CFG_STRING)
		{
		  if (strcmp (node->label->v.string, mu_program_name) == 0)
		    {
		      /* Move all nodes from this block to the topmost
			 level */
		      mu_iterator_ctl (itr, mu_itrctl_insert_list,
				       node->nodes);
		      mu_iterator_ctl (itr, mu_itrctl_delete, NULL);
		      /*FIXME:mu_cfg_free_node (node);*/
		    }
		}
	      else
		{
		  mu_cfg_perror (tree->debug, &node->locus,
				 _("argument to `program' is not a string"));
		  mu_iterator_ctl (itr, mu_itrctl_delete, NULL);
		}
	    }
	}
      else if (node->type == mu_cfg_node_param &&
	       strcmp (node->tag, "include") == 0)
	{
	  if (node->label->type == MU_CFG_STRING)
	    {
	      mu_cfg_tree_t *t = do_include (node->label->v.string, flags,
					     &node->locus);
	      if (t)
		{
		  /* Merge the new tree into the current point and
		     destroy the rest of it */
		  mu_iterator_ctl (itr, mu_itrctl_insert_list, t->nodes);
		  mu_opool_union (&tree->pool, &t->pool);
		  mu_cfg_destroy_tree (&t);
		}		      
	    }
	  else
	    mu_cfg_perror (tree->debug, &node->locus,
			   _("argument to `include' is not a string"));
	  /* Remove node from the list */
	  mu_iterator_ctl (itr, mu_itrctl_delete, NULL);
	}
    }
  return 0;
}

static int
_mu_cfg_preorder_recursive (void *item, void *cbdata)
{
  mu_cfg_node_t *node = item;
  struct mu_cfg_iter_closure *clos = cbdata;

  switch (node->type)
    {
    case mu_cfg_node_undefined:
      abort ();

    case mu_cfg_node_statement:
      switch (clos->beg (node, clos->data))
	{
	case MU_CFG_ITER_OK:
	  if (mu_cfg_preorder (node->nodes, clos))
	    return 1;
	  if (clos->end && clos->end (node, clos->data) == MU_CFG_ITER_STOP)
	    return 1;
	  break;

	case MU_CFG_ITER_SKIP:
	  break;

	case MU_CFG_ITER_STOP:
	  return 1;
	}
      break;

    case mu_cfg_node_param:
      return clos->beg (node, clos->data) == MU_CFG_ITER_STOP;
    }
  return 0;
}

int
mu_cfg_preorder (mu_list_t nodelist, struct mu_cfg_iter_closure *clos)
{
  return mu_list_do (nodelist, _mu_cfg_preorder_recursive, clos);
}



void
mu_cfg_destroy_tree (mu_cfg_tree_t **ptree)
{
  if (ptree && *ptree)
    {
      mu_cfg_tree_t *tree = *ptree;
      mu_list_destroy (&tree->nodes);
      mu_opool_destroy (&tree->pool);
      *ptree = NULL;
    }
}



struct mu_cfg_section_list
{
  struct mu_cfg_section_list *next;
  struct mu_cfg_section *sec;
};

struct scan_tree_data
{
  struct mu_cfg_section_list *list;
  void *target;
  void *call_data;
  mu_cfg_tree_t *tree;
  int error;
};

static struct mu_cfg_cont *
find_container (mu_list_t list, enum mu_cfg_cont_type type,
		const char *ident, size_t len)
{
  mu_iterator_t iter;
  struct mu_cfg_cont *ret = NULL;

  if (len == 0)
    len = strlen (ident);

  mu_list_get_iterator (list, &iter);
  for (mu_iterator_first (iter); !mu_iterator_is_done (iter);
       mu_iterator_next (iter))
    {
      struct mu_cfg_cont *cont;
      mu_iterator_current (iter, (void**) &cont);

      if (cont->type == type
	  && strlen (cont->v.ident) == len
	  && memcmp (cont->v.ident, ident, len) == 0)
	{
	  ret = cont;
	  break;
	}
    }
  mu_iterator_destroy (&iter);
  return ret;
}

static struct mu_cfg_section *
find_subsection (struct mu_cfg_section *sec, const char *ident, size_t len)
{
  if (sec)
    {
      if (sec->children)
	{
	  struct mu_cfg_cont *cont = find_container (sec->children,
						     mu_cfg_cont_section,
						     ident, len);
	  if (cont)
	    return &cont->v.section;
	}
    }
  return NULL;
}

static struct mu_cfg_param *
find_param (struct mu_cfg_section *sec, const char *ident, size_t len)
{
  if (sec)
    {
      if (sec->children)
	{
	  struct mu_cfg_cont *cont = find_container (sec->children,
						     mu_cfg_cont_param,
						     ident, len);
	  if (cont)
	    return &cont->v.param;
	}
    }
  return NULL;
}

static int
push_section (struct scan_tree_data *dat, struct mu_cfg_section *sec)
{
  struct mu_cfg_section_list *p = mu_alloc (sizeof *p);
  if (!p)
    {
      mu_cfg_perror (dat->tree->debug, NULL, _("not enough memory"));
      return 1;
    }
  p->sec = sec;
  p->next = dat->list;
  dat->list = p;
  return 0;
}

static struct mu_cfg_section *
pop_section (struct scan_tree_data *dat)
{
  struct mu_cfg_section_list *p = dat->list;
  struct mu_cfg_section *sec = p->sec;
  dat->list = p->next;
  free (p);
  return sec;
}

#define STRTONUM(s, type, base, res, limit, d, loc)			\
  {									\
    type sum = 0;							\
									\
    for (; *s; s++)							\
      {									\
	type x;								\
									\
	if ('0' <= *s && *s <= '9')					\
	  x = sum * base + *s - '0';					\
	else if (base == 16 && 'a' <= *s && *s <= 'f')			\
	  x = sum * base + *s - 'a';					\
	else if (base == 16 && 'A' <= *s && *s <= 'F')			\
	  x = sum * base + *s - 'A';					\
	else								\
	  break;							\
	if (x <= sum)							\
	  {								\
	    mu_cfg_perror (d, loc, _("numeric overflow"));		\
	    return 1;							\
	  }								\
	else if (limit && x > limit)					\
	  {								\
	    mu_cfg_perror (d, loc,					\
			    _("value out of allowed range"));		\
	    return 1;							\
	  }								\
	sum = x;							\
      }									\
    res = sum;								\
  }

#define STRxTONUM(s, type, res, limit, d, loc)				\
  {									\
    int base;								\
    if (*s == '0')							\
      {									\
	s++;								\
	if (*s == 0)							\
	  base = 10;							\
	else if (*s == 'x' || *s == 'X')				\
	  {								\
	    s++;							\
	    base = 16;							\
	  }								\
	else								\
	  base = 8;							\
      } else								\
      base = 10;							\
    STRTONUM (s, type, base, res, limit, d, loc);			\
  }

#define GETUNUM(str, type, res, d, loc)					\
  {									\
    type tmpres;							\
    const char *s = str;						\
    STRxTONUM (s, type, tmpres, 0, d, loc);				\
    if (*s)								\
      {									\
	mu_cfg_perror (d, loc,						\
		       _("not a number (stopped near `%s')"),		\
		       s);						\
	return 1;							\
      }									\
    res = tmpres;							\
  }

#define GETSNUM(str, type, res, d, loc)					\
  {									\
    unsigned type tmpres;						\
    const char *s = str;						\
    int sign;								\
    unsigned type limit;						\
									\
    if (*s == '-')							\
      {									\
	sign++;								\
	s++;								\
	limit = TYPE_MINIMUM (type);					\
	limit = - limit;						\
      }									\
    else								\
      {									\
	sign = 0;							\
	limit = TYPE_MAXIMUM (type);					\
      }									\
									\
    STRxTONUM (s, unsigned type, tmpres, limit, d, loc);		\
    if (*s)								\
      {									\
	mu_cfg_perror (d, loc,						\
		       _("not a number (stopped near `%s')"),		\
		       s);						\
	return 1;							\
      }									\
    res = sign ? - tmpres : tmpres;					\
  }

static int
parse_ipv4 (struct scan_tree_data *sdata, const mu_cfg_locus_t *locus,
	    const char *str, struct in_addr *res)
{
  struct in_addr addr;
  if (inet_aton (str, &addr) == 0)
    {
      mu_cfg_perror (sdata->tree->debug, locus, _("not an IPv4"));
      return 1;
    }
  addr.s_addr = ntohl (addr.s_addr);
  *res = addr;
  return 0;
}

static int
parse_host (struct scan_tree_data *sdata, const mu_cfg_locus_t *locus,
	    const char *str, struct in_addr *res)
{
  struct in_addr addr;
  struct hostent *hp = gethostbyname (str);
  if (hp)
    {
      addr.s_addr = *(unsigned long *)hp->h_addr;
    }
  else if (inet_aton (str, &addr) == 0)
    {
      mu_cfg_perror (sdata->tree->debug, locus,
		     _("cannot resolve hostname `%s'"),
		     str);
      return 1;
    }
  addr.s_addr = ntohl (addr.s_addr);
  *res = addr;
  return 0;
}

static int
parse_cidr (struct scan_tree_data *sdata, const mu_cfg_locus_t *locus,
	    const char *str, mu_cfg_cidr_t *res)
{
  struct in_addr addr;
  unsigned long mask;
  char astr[16];
  const char *p, *s;

  p = strchr (str, '/');
  if (p)
    {
      int len = p - str;
      if (len > sizeof astr - 1) {
	mu_cfg_perror (sdata->tree->debug, locus,
		       _("not a valid IPv4 address in CIDR"));
	return 1;
      }
      memcpy (astr, str, len);
      astr[len] = 0;
      if (inet_aton (astr, &addr) == 0)
	{
	  mu_cfg_perror (sdata->tree->debug, locus,
			 _("not a valid IPv4 address in CIDR"));
	  return 1;
	}
      addr.s_addr = ntohl (addr.s_addr);

      p++;
      s = p;
      STRxTONUM (s, unsigned long, mask, 0, sdata->tree->debug, locus);
      if (*s == '.')
	{
	  struct in_addr a;
	  if (inet_aton (p, &a) == 0)
	    {
	      mu_cfg_perror (sdata->tree->debug, locus,
			     _("not a valid network in CIDR"));
	      return 1;
	    }
	  a.s_addr = ntohl (a.s_addr);
	  for (mask = 0; (a.s_addr & 1) == 0 && mask < 32; )
	    {
	      mask++;
	      a.s_addr >>= 1;
	    }
	  mask = 32 - mask;
	}
      else if (mask > 32)
	{
	  mu_cfg_perror (sdata->tree->debug, locus,
			 _("not a valid network mask in CIDR"));
	  return 1;
	}
    }
  else
    {
      int i;
      unsigned short x;
      addr.s_addr = 0;

      p = str;
      for (i = 0; i < 3; i++)
	{
	  STRxTONUM (p, unsigned short, x, 255, sdata->tree->debug, locus);
	  if (*p != '.')
	    break;
	  addr.s_addr = (addr.s_addr << 8) + x;
	}

      if (*p)
	{
	  mu_cfg_perror (sdata->tree->debug, locus,
			 _("not a CIDR (stopped near `%s')"),
			 p);
	  return 1;
	}

      mask = i * 8;

      addr.s_addr <<= (4 - i) * 8;
    }

  res->addr = addr;
  res->mask = mask;
  return 0;
}

int
mu_cfg_parse_boolean (const char *str, int *res)
{
  if (strcmp (str, "yes") == 0
      || strcmp (str, "on") == 0
      || strcmp (str, "t") == 0
      || strcmp (str, "true") == 0
      || strcmp (str, "1") == 0)
    *res = 1;
  else if (strcmp (str, "no") == 0
	   || strcmp (str, "off") == 0
	   || strcmp (str, "nil") == 0
	   || strcmp (str, "false") == 0
	   || strcmp (str, "0") == 0)
    *res = 0;
  else
    return 1;
  return 0;
}

static int
parse_bool (struct scan_tree_data *sdata, const mu_cfg_locus_t *locus,
	    const char *str, int *res)
{
  if (mu_cfg_parse_boolean (str, res))
    {
      mu_cfg_perror (sdata->tree->debug, locus, _("not a boolean"));
      return 1;
    }
  return 0;
}

static int
valcvt (struct scan_tree_data *sdata, const mu_cfg_locus_t *locus,
	void *tgt,
	enum mu_cfg_param_data_type type, mu_config_value_t *val)
{
  if (val->type != MU_CFG_STRING)
    {
      mu_cfg_perror (sdata->tree->debug, locus, _("expected string value"));
      return 1;
    }
  switch (type)
    {
    case mu_cfg_string:
      {
	char *s = mu_strdup (val->v.string);
	/* FIXME: free tgt? */
	*(char**)tgt = s;
	break;
      }

    case mu_cfg_short:
      GETSNUM (val->v.string, short, *(short*)tgt, sdata->tree->debug, locus);
      break;

    case mu_cfg_ushort:
      GETUNUM (val->v.string, unsigned short, *(unsigned short*)tgt,
	       sdata->tree->debug, locus);
      break;

    case mu_cfg_int:
      GETSNUM (val->v.string, int, *(int*)tgt, sdata->tree->debug, locus);
      break;

    case mu_cfg_uint:
      GETUNUM (val->v.string, unsigned int, *(unsigned int*)tgt,
	       sdata->tree->debug, locus);
      break;

    case mu_cfg_long:
      GETSNUM (val->v.string, long, *(long*)tgt,
	       sdata->tree->debug, locus);
      break;

    case mu_cfg_ulong:
      GETUNUM (val->v.string, unsigned long, *(unsigned long*)tgt,
	       sdata->tree->debug, locus);
      break;

    case mu_cfg_size:
      GETUNUM (val->v.string, size_t, *(size_t*)tgt,
	       sdata->tree->debug, locus);
      break;

    case mu_cfg_off:
      mu_cfg_perror (sdata->tree->debug, locus, _("not implemented yet"));
      /* GETSNUM(node->tag_label, off_t, *(off_t*)tgt); */
      return 1;

    case mu_cfg_time:
      GETUNUM (val->v.string, time_t, *(time_t*)tgt,
	       sdata->tree->debug, locus);
      break;

    case mu_cfg_bool:
      if (parse_bool (sdata, locus, val->v.string, (int*) tgt))
	return 1;
      break;

    case mu_cfg_ipv4:
      if (parse_ipv4 (sdata, locus, val->v.string, (struct in_addr *)tgt))
	return 1;
      break;

    case mu_cfg_cidr:
      if (parse_cidr (sdata, locus, val->v.string, (mu_cfg_cidr_t *)tgt))
	return 1;
      break;

    case mu_cfg_host:
      if (parse_host (sdata, locus, val->v.string, (struct in_addr *)tgt))
	return 1;
      break;

    default:
      return 1;
    }
  return 0;
}

struct set_closure
{
  mu_list_t list;
  enum mu_cfg_param_data_type type;
  struct scan_tree_data *sdata;
  const mu_cfg_locus_t *locus;
};

static size_t config_type_size[] = {
  sizeof (char*),          /* mu_cfg_string */
  sizeof (short),          /* mu_cfg_short */
  sizeof (unsigned short), /* mu_cfg_ushort */
  sizeof (int),            /* mu_cfg_int */
  sizeof (unsigned),       /* mu_cfg_uint */
  sizeof (long),           /* mu_cfg_long */
  sizeof (unsigned long),  /* mu_cfg_ulong */
  sizeof (size_t),         /* mu_cfg_size */
  sizeof (mu_off_t),       /* mu_cfg_off */
  sizeof (time_t),         /* mu_cfg_time */
  sizeof (int),            /* mu_cfg_bool */
  sizeof (struct in_addr), /* mu_cfg_ipv4 */
  sizeof (mu_cfg_cidr_t),  /* mu_cfg_cidr */
  sizeof (struct in_addr), /* mu_cfg_host */
  0,                       /* mu_cfg_callback */
  0,                       /* mu_cfg_section */
}  ;

static int
_set_fun (void *item, void *data)
{
  mu_config_value_t *val = item;
  struct set_closure *clos = data;
  void *tgt;
  size_t size;

  if (clos->type >= MU_ARRAY_SIZE(config_type_size)
      || (size = config_type_size[clos->type]) == 0)
    {
    mu_cfg_perror (clos->sdata->tree->debug, clos->locus,
		   _("INTERNAL ERROR at %s:%d: unhandled data type %d"),
		   __FILE__, __LINE__, clos->type);
    return 1;
    }

  tgt = mu_alloc (size);
  if (!tgt)
    {
      mu_cfg_perror (clos->sdata->tree->debug, clos->locus,
		     _("not enough memory"));
      return 1;
    }

  if (valcvt (clos->sdata, clos->locus, &tgt, clos->type, val) == 0)
    mu_list_append (clos->list, tgt);
  return 0;
}

static int
parse_param (struct scan_tree_data *sdata, const mu_cfg_node_t *node)
{
  void *tgt;
  struct set_closure clos;
  struct mu_cfg_param *param = find_param (sdata->list->sec, node->tag,
					   0);

  if (!param)
    {
      mu_cfg_perror (sdata->tree->debug, &node->locus,
		     _("unknown keyword `%s'"),
		     node->tag);
      return 1;
    }

  if (param->data)
    tgt = param->data;
  else if (sdata->list->sec->target)
    tgt = (char*)sdata->list->sec->target + param->offset;
  else if (sdata->target)
    tgt = (char*)sdata->target + param->offset;
  else if (param->type == mu_cfg_callback)
    tgt = NULL;
  else
    {
      mu_cfg_perror (sdata->tree->debug, &node->locus,
		     _("INTERNAL ERROR: cannot determine target offset for "
		       "%s"), param->ident);
      abort ();
    }

  memset (&clos, 0, sizeof clos);
  clos.type = MU_CFG_TYPE (param->type);
  if (MU_CFG_IS_LIST (param->type))
    {
      clos.sdata = sdata;
      clos.locus = &node->locus;
      switch (node->label->type)
	{
	case MU_CFG_LIST:
	  break;

	case MU_CFG_STRING:
	  {
	    mu_list_t list;
	    mu_list_create (&list);
	    mu_list_append (list, config_value_dup (node->label));
	    node->label->type = MU_CFG_LIST;
	    node->label->v.list = list;
	  }
	  break;

	case MU_CFG_ARRAY:
	  mu_cfg_perror (sdata->tree->debug, &node->locus,
			 _("expected list, but found array"));
	  return 1;
	}

      mu_list_create (&clos.list);
      mu_list_do (node->label->v.list, _set_fun, &clos);
      *(mu_list_t*)tgt = clos.list;
    }
  else if (clos.type == mu_cfg_callback)
    {
      mu_debug_set_locus (sdata->tree->debug, node->locus.file,
			  node->locus.line);
      if (!param->callback)
	{
	  mu_cfg_perror (sdata->tree->debug, &node->locus,
			 _("INTERNAL ERROR: %s: callback not defined"),
			 node->tag);
	  abort ();
	}
      if (param->callback (sdata->tree->debug, tgt, node->label))
	return 1;

    }
  else
    return valcvt (sdata, &node->locus, tgt, clos.type, node->label);

  return 0;
}


static int
_scan_tree_helper (const mu_cfg_node_t *node, void *data)
{
  struct scan_tree_data *sdata = data;
  struct mu_cfg_section *sec;

  switch (node->type)
    {
    case mu_cfg_node_undefined:
      abort ();

    case mu_cfg_node_statement:
      sec = find_subsection (sdata->list->sec, node->tag, 0);
      if (!sec)
	{
	  if (mu_cfg_parser_verbose)
	    {
	      _mu_cfg_debug_set_locus (sdata->tree->debug, &node->locus);
	      mu_cfg_format_error (sdata->tree->debug, MU_DIAG_WARNING,
				   _("unknown section `%s'"),
				   node->tag);
	    }
	  return MU_CFG_ITER_SKIP;
	}
      if (!sec->children)
	return MU_CFG_ITER_SKIP;
      if (sdata->list->sec->target)
	sec->target = (char*)sdata->list->sec->target + sec->offset;
      else if (sdata->target)
	sec->target = (char*)sdata->target + sec->offset;
      if (sec->parser)
	{
	  mu_debug_set_locus (sdata->tree->debug,
			      node->locus.file ?
				     node->locus.file : _("unknown file"),
			      node->locus.line);
	  if (sec->parser (mu_cfg_section_start, node,
			   sec->label, &sec->target,
			   sdata->call_data, sdata->tree))
	    {
	      sdata->error++;
	      return MU_CFG_ITER_SKIP;
	    }
	}
      push_section(sdata, sec);
      break;

    case mu_cfg_node_param:
      if (parse_param (sdata, node))
	{
	  sdata->error++;
	  return MU_CFG_ITER_SKIP;
	}
      break;
    }
  return MU_CFG_ITER_OK;
}

static int
_scan_tree_end_helper (const mu_cfg_node_t *node, void *data)
{
  struct scan_tree_data *sdata = data;
  struct mu_cfg_section *sec;

  switch (node->type)
    {
    default:
      abort ();

    case mu_cfg_node_statement:
      sec = pop_section (sdata);
      if (sec && sec->parser)
	{
	  if (sec->parser (mu_cfg_section_end, node,
			   sec->label, &sec->target,
			   sdata->call_data, sdata->tree))
	    {
	      sdata->error++;
	      return MU_CFG_ITER_SKIP;
	    }
	}
    }
  return MU_CFG_ITER_OK;
}

int
mu_cfg_scan_tree (mu_cfg_tree_t *tree, struct mu_cfg_section *sections,
		  void *target, void *data)
{
  mu_debug_t debug = NULL;
  struct scan_tree_data dat;
  struct mu_cfg_iter_closure clos;
    
  dat.tree = tree;
  dat.list = NULL;
  dat.error = 0;
  dat.call_data = data;
  dat.target = target;
  
  if (!tree->debug)
    {
      mu_diag_get_debug (&debug);
      tree->debug = debug;
    }
  if (push_section (&dat, sections))
    return 1;
  clos.beg = _scan_tree_helper;
  clos.end = _scan_tree_end_helper;
  clos.data = &dat;
  mu_cfg_preorder (tree->nodes, &clos);
  if (debug)
    {
      mu_debug_set_locus (debug, NULL, 0);
      tree->debug = NULL;
    }
  pop_section (&dat);
  return dat.error;
}

int
mu_cfg_find_section (struct mu_cfg_section *root_sec,
		     const char *path, struct mu_cfg_section **retval)
{
  while (path[0])
    {
      struct mu_cfg_section *sec;
      size_t len;
      const char *p;

      while (*path == MU_CFG_PATH_DELIM)
	path++;

      if (*path == 0)
	return MU_ERR_NOENT;

      p = strchr (path, MU_CFG_PATH_DELIM);
      if (p)
	len = p - path;
      else
	len = strlen (path);

      sec = find_subsection (root_sec, path, len);
      if (!sec)
	return MU_ERR_NOENT;
      root_sec = sec;
      path += len;
    }
  if (retval)
    *retval = root_sec;
  return 0;
}


int
mu_cfg_tree_create (struct mu_cfg_tree **ptree)
{
  struct mu_cfg_tree *tree = calloc (1, sizeof *tree);
  if (!tree)
    return errno;
  mu_opool_create (&tree->pool, 1);
  *ptree = tree;
  return 0;
}

void
mu_cfg_tree_set_debug (struct mu_cfg_tree *tree, mu_debug_t debug)
{
  tree->debug = debug;
}

mu_cfg_node_t *
mu_cfg_tree_create_node (struct mu_cfg_tree *tree,
			 enum mu_cfg_node_type type,
			 const mu_cfg_locus_t *loc,
			 const char *tag, const char *label,
			 mu_list_t nodelist)
{
  char *p;
  mu_cfg_node_t *np;
  size_t size = sizeof *np + strlen (tag) + 1;
  mu_config_value_t val;

  np = mu_alloc (size);
  np->type = type;
  if (loc)
    np->locus = *loc;
  else
    memset (&np->locus, 0, sizeof np->locus);
  p = (char*) (np + 1);
  np->tag = p;
  strcpy (p, tag);
  p += strlen (p) + 1;
  val.type = MU_CFG_STRING;
  if (label)
    {
      mu_opool_clear (tree->pool);
      mu_opool_appendz (tree->pool, label);
      val.v.string = mu_opool_finish (tree->pool, NULL);
      np->label = config_value_dup (&val);
    }
  else
    np->label = NULL;
  np->nodes = nodelist;
  return np;
}

void
mu_cfg_tree_add_node (mu_cfg_tree_t *tree, mu_cfg_node_t *node)
{
  if (!node)
    return;
  if (!tree->nodes)
    /* FIXME: return code? */
    mu_cfg_create_node_list (&tree->nodes);
  mu_list_append (tree->nodes, node);
}

void
mu_cfg_tree_add_nodelist (mu_cfg_tree_t *tree, mu_list_t nodelist)
{
  if (!nodelist)
    return;
  if (!tree->nodes)
    /* FIXME: return code? */
    mu_cfg_create_node_list (&tree->nodes);
  mu_list_append_list (tree->nodes, nodelist);
}


/* Return 1 if configuration value A equals B */
int
mu_cfg_value_eq (mu_config_value_t *a, mu_config_value_t *b)
{
  if (a->type != b->type)
    return 0;
  switch (a->type)
    {
    case MU_CFG_STRING:
      if (a->v.string == NULL)
	return b->v.string == NULL;
      return strcmp (a->v.string, b->v.string) == 0;
		     
    case MU_CFG_LIST:
      {
	int ret = 1;
	size_t cnt;
	size_t i;
	mu_iterator_t aitr, bitr;
	
	mu_list_count (a->v.list, &cnt);
	mu_list_count (b->v.list, &i);
	if (i != cnt)
	  return 1;

	mu_list_get_iterator (a->v.list, &aitr);
	mu_list_get_iterator (b->v.list, &bitr);
	for (i = 0,
	       mu_iterator_first (aitr),
	       mu_iterator_first (bitr);
	     !mu_iterator_is_done (aitr) && !mu_iterator_is_done (bitr);
	     mu_iterator_next (aitr),
	       mu_iterator_next (bitr),
	       i++)
	  {
	    mu_config_value_t *ap, *bp;
	    mu_iterator_current (aitr, (void**)&ap);
	    mu_iterator_current (bitr, (void**)&bp);
	    ret = mu_cfg_value_eq (ap, bp);
	    if (!ret)
	      break;
	  }
	mu_iterator_destroy (&aitr);
	mu_iterator_destroy (&bitr);
	return ret && i == cnt;
      }
	
    case MU_CFG_ARRAY:
      if (a->v.arg.c == b->v.arg.c)
	{
	  size_t i;
	  for (i = 0; i < a->v.arg.c; i++)
	    if (!mu_cfg_value_eq (&a->v.arg.v[i], &b->v.arg.v[i]))
	      return 0;
	  return 1;
	}
    }
  return 0;
}


struct find_data
{
  int argc;
  char **argv;
  int tag;
  mu_config_value_t *label;
  const mu_cfg_node_t *node;
};

static void
free_value_mem (mu_config_value_t *p)
{
  switch (p->type)
    {
    case MU_CFG_STRING:
      free ((char*)p->v.string);
      break;
      
    case MU_CFG_LIST:
      /* FIXME */
      break;
      
    case MU_CFG_ARRAY:
      {
	size_t i;
	for (i = 0; i < p->v.arg.c; i++)
	  free_value_mem (&p->v.arg.v[i]);
      }
    }
}

static void
destroy_value (void *p)
{
  mu_config_value_t *val = p;
  if (val)
    {
      free_value_mem (val);
      free (val);
    }
}

static mu_config_value_t *
parse_label (const char *str)
{
  mu_config_value_t *val = NULL;
  int count, i;
  char **vect;
  size_t len = strlen (str);
  
  if (len > 1 && str[0] == '(' && str[len-1] == ')')
    {
      mu_list_t lst;
      mu_argcv_get_np (str + 1, len - 2,
		       ", \t", NULL,
		       0,
		       &count, &vect, NULL);
      mu_list_create (&lst);
      mu_list_set_destroy_item (lst, destroy_value);
      for (i = 0; i < count; i++)
	{
	  mu_config_value_t *p = mu_alloc (sizeof (*p));
	  p->type = MU_CFG_STRING;
	  p->v.string = vect[i];
	  mu_list_append (lst, p);
	}
      free (vect);
      val = mu_alloc (sizeof (*val));
      val->type = MU_CFG_LIST;
      val->v.list = lst;
    }
  else
    {      
      mu_argcv_get (str, NULL, NULL, &count, &vect);
      val = mu_alloc (sizeof (*val));
      if (count == 1)
	{
	  val->type = MU_CFG_STRING;
	  val->v.string = vect[0];
	  free (vect);
	}
      else
	{
	  val->type = MU_CFG_ARRAY;
	  val->v.arg.c = count;
	  val->v.arg.v = mu_alloc (count * sizeof (val->v.arg.v[0]));
	  for (i = 0; i < count; i++)
	    {
	      val->v.arg.v[i].type = MU_CFG_STRING;
	      val->v.arg.v[i].v.string = vect[i];
	    }
	  free (vect);
	}
    }
  return val;
}

static void
parse_tag (struct find_data *fptr)
{
  char *p = strchr (fptr->argv[fptr->tag], '=');
  if (p)
    {
      *p++ = 0;
      fptr->label = parse_label (p);
    }
  else
    fptr->label = NULL;
}

static int
node_finder (const mu_cfg_node_t *node, void *data)
{
  struct find_data *fdptr = data;
  if (strcmp (fdptr->argv[fdptr->tag], node->tag) == 0
      && (!fdptr->label || mu_cfg_value_eq (fdptr->label, node->label)))
    {
      fdptr->tag++;
      if (fdptr->tag == fdptr->argc)
	{
	  fdptr->node = node;
	  return MU_CFG_ITER_STOP;
	}
      parse_tag (fdptr);
      return MU_CFG_ITER_OK;
    }
  
  return node->type == mu_cfg_node_statement ?
               MU_CFG_ITER_SKIP : MU_CFG_ITER_OK;
}

int	    
mu_cfg_find_node (mu_cfg_tree_t *tree, const char *path, mu_cfg_node_t **pval)
{
  int rc;
  struct find_data data;
  struct mu_cfg_iter_closure clos;

  rc = mu_argcv_get_np (path, strlen (path),
			MU_CFG_PATH_DELIM_STR, NULL,
			0, &data.argc, &data.argv, NULL);
  if (rc)
    return rc;
  data.tag = 0;
  parse_tag (&data);

  clos.beg = node_finder;
  clos.end = NULL;
  clos.data = &data;
  rc = mu_cfg_preorder (tree->nodes, &clos);
  destroy_value (data.label);
  if (rc)
    {
      *pval = (mu_cfg_node_t *) data.node;
      return 0;
    }
  return MU_ERR_NOENT;
}



int
mu_cfg_create_subtree (const char *path, mu_cfg_node_t **pnode)
{
  int rc;
  int argc, i;
  char **argv;
  mu_cfg_locus_t locus;
  enum mu_cfg_node_type type;
  mu_cfg_node_t *node = NULL;
  
  locus.file = "<int>";
  locus.line = 0;
  
  rc = mu_argcv_get_np (path, strlen (path), MU_CFG_PATH_DELIM_STR, NULL, 0,
			&argc, &argv, NULL);
  if (rc)
    return rc;

  for (i = argc - 1; i >= 0; i--)
    {
      mu_list_t nodelist = NULL;
      char *p = strrchr (argv[i], '=');
      mu_config_value_t *label = NULL;

      type = mu_cfg_node_statement;
      if (p)
	{
	  *p++ = 0;
	  label = parse_label (p);
	  if (i == argc - 1)
	    type = mu_cfg_node_param;
	}

      if (node)
	{
	  mu_cfg_create_node_list (&nodelist);
	  mu_list_append (nodelist, node);
	}
      node = mu_cfg_alloc_node (type, &locus, argv[i], label, nodelist);
    }

  mu_argcv_free (argc, argv);
  *pnode = node;
  return 0;
}

  
	  
        
  

