#include "../wu-ftpd.h"

/* Allocated size of buffer pathname[] in main () */
#define PATHNAME_SZ MAXPATHLEN+1

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


char *
realpath(const char *pathname, char *result, char* chroot_path)
{
  char curpath[MAXPATHLEN];

  if (result == NULL)
    return(NULL);

  if(pathname == NULL){
    *result = EOS; 
    return(NULL);
  }

  /* OK */
  r_strncpy(curpath, pathname, MAXPATHLEN);

  return result;
}

int main ()
{
  char pathname [PATHNAME_SZ];
  char result [MAXPATHLEN];
  char chroot_path [MAXPATHLEN];

  pathname [PATHNAME_SZ-1] = EOS;
  result [MAXPATHLEN-1] = EOS;
  chroot_path [MAXPATHLEN-1] = EOS;

  realpath(pathname, result, chroot_path);

  return 0;
}
