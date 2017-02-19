#include "../glob.h"

static int
glob3(Char *pathbuf, Char *pathend, Char *pathlim, Char *pattern)
{
  int dc;
  int old;
  dc = 0;
  for (;;)
    if (pathend + dc > pathlim) break;
    else {
      pathend[dc] = 1;
      old = dc;
      dc++;
      /* OK */
      if (pathend[old] == EOS) break;
    }

  return 0;
}

int main ()
{
  Char *buf;
  Char *pattern;
  Char *bound;

  Char A [MAXPATHLEN+1];
  Char B [PATTERNLEN];

  buf = A;
  pattern = B;

  bound = A + sizeof(A) - 1;

  glob3 (buf, buf, bound, pattern);

  return 0;
}
