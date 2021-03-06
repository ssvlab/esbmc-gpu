#include "../stubs.h"
#include "../base.h"

#define MAXLINE BASE_SZ

int main (void)
{
  char fbuf[MAXLINE+1];
  int fb;
  int c1, c2, c3;

  fb = 0;
  while ((c1 = nondet_int ()) != EOF)
  {
    c2 = nondet_int ();
    if (c2 == EOF)
      break;

    c3 = nondet_int ();
    if (c3 == EOF)
      break;

    if (c1 == '=' || c2 == '=')
      continue;

    /* BAD */
    fbuf[fb] = c1;

    /* BAD */
    if (fbuf[fb] == '\n')
    {
      fb--;
      if (fb < 0)
	fb = 0;
      else if (fbuf[fb] != '\r') 
	fb++;

      /* BAD */
      fbuf[fb] = 0;
      fb = 0;
    }
    else
      fb++;

    /* BAD */
    fbuf[fb] = c2;

    /* BAD */
    if (fbuf[fb] == '\n')
    {
      fb--;
      if (fb < 0)
	fb = 0;
      else if (fbuf[fb] != '\r') 
	fb++;

      /* BAD */
      fbuf[fb] = 0;
      fb = 0;
    }
    else
      fb++;

    if (c3 == '=')
      continue;
    /* BAD */
    fbuf[fb] = c3;

    /* BAD */
    if (fbuf[fb] == '\n')
    {
      fb--;
      if (fb < 0)
	fb = 0;
      else if (fbuf[fb] != '\r') 
	fb++;

      /* BAD */
      fbuf[fb] = 0;
      fb = 0;
    }
    else
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
