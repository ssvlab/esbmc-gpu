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

  login[(int) (sizeof login - 1)] = EOS;
  gecos[(int) (sizeof gecos - 1)] = EOS;

  j = 0;
  /* OK */
  (void) r_strncpy (buf + j, login, sizeof (buf) - j);
  buf[sizeof(buf)-1] = EOS;

  return 0;
}

