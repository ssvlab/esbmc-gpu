#include "../stubs.h"
#include "../base.h"

#define MAXLINE BASE_SZ

int main (void)
{
  char fbuf[MAXLINE+1];
  int fb;
  int c1;

  fb = 0;

  while ((c1 = nondet_int ()) != EOF)
  {
    /* OK */
    fbuf[fb] = c1;
    fb++;
    if (fb >= MAXLINE)
      fb = 0;
  }

  /* force out partial last line */
  if (fb > 0)
  {
    /* OK */
    fbuf[fb] = EOS;
  }

  return 0;
}
