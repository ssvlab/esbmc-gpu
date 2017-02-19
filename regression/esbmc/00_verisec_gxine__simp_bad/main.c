#include "../constants.h"


char *r_strcpy (char *dest, const char *src)
{
  int i;
  char tmp;
  for (i = 0; ; i++) {
    tmp = src[i];
    /* r_strcpy RELEVANT */
    dest[i] = tmp; // DO NOT CHANGE THE POSITION OF THIS LINE
    if (src[i] == EOS)
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
    /* BAD */
    r_strcpy (serv_adr.sun_path, filename);

    return 0;
}
