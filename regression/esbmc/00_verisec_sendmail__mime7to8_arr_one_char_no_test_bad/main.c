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
    /* BAD */
    fbuf[fb] = c1;
    fb++;
  }

  /* force out partial last line */
  if (fb > 0)
  {
    /* BAD */
    fbuf[fb] = EOS;
  }

  return 0;
}
