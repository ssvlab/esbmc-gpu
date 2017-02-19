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

int main ()
{
    struct sockaddr_un serv_adr;
    char               filename [FILENAME_SZ];

    /* server filename */
    filename[FILENAME_SZ-1] = EOS;
    
    /* initialize the server address structure */
    /* OK */
    r_strncpy (serv_adr.sun_path, filename, SUN_PATH_SZ-1);

    return 0;
}
