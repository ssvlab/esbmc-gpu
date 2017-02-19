#include "../constants.h"

int
main (void)
{
  // these were parameters
  char login[LOGIN + 1];
  char gecos[GECOS + 1];

  char buf[BUF + 1];
  char c;
  int i, j;
  int l;

  login[(int) (sizeof login - 1)] = EOS;
  gecos[(int) (sizeof gecos - 1)] = EOS;

  i = 0;
  if (gecos[i] == '*')
    i++;

  /* bail out early if we'd overflow buf[] */
  l = strlen(gecos+i);
  if (l > BUF)
    return 0;

  c = gecos[i];
  j = 0;
  while (c != EOS && c != ',' && c != ';' && c != '%')
  {
    if (c == '&')
    {

    }
    else
    {
      /* OK */
      buf[j] = c;
      j++;
    }	    
    i++;
    c = gecos[i];
  }
  buf[j] = EOS;
  return 0;
}

