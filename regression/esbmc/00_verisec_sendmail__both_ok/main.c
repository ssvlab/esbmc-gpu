#include "../constants.h"

char *r_strncpy (char *dest, const char *src, size_t n)
{
  int _i;

  /* r_strncpy RELEVANT */
  if (n > 0) dest[n-1];

  for (_i = 0; _i < n; _i++) {
    dest[_i] = src[_i]; // DO NOT CHANGE THE POSITION OF THIS LINE
    if (src[_i] == EOS)
      break;
  }
  return dest;
}

int
main (void)
{
  // these were parameters
  char login[LOGIN + 1];
  char gecos[GECOS + 1];

  char buf[BUF + 1];
  char c;
  int i, j;
  int p, l;

  login[(int) (sizeof login - 1)] = EOS;
  gecos[(int) (sizeof gecos - 1)] = EOS;

  i = 0;
  if (gecos[i] == '*')
    i++;

  /* find length of final string */
  l = 0;
  p = i;
  c = gecos[p];
  while (c != EOS && c != ',' && c != ';' && c != '%')
  {
    if (c == '&')
      l += strlen(login);
    else
      l++;
    p++;
    c = gecos[p];
  }

  /* bail out early if we'd overflow buf[] */
  if (l > BUF)
    return 0;

  c = gecos[i];
  j = 0;
  while (c != EOS && c != ',' && c != ';' && c != '%')
  {
    if (c == '&')
    {
      /* OK */
      (void) r_strncpy (buf + j, login, sizeof (buf) - j);
      buf[sizeof(buf)-1] = EOS;
      while (buf[j] != EOS)
	j++;
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
