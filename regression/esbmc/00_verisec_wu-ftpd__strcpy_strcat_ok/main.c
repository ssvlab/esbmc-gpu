#include "../wu-ftpd.h"

char *r_strncat(char *dest, const char *src, size_t n)
{
  int i, j;
  char tmp;
  i = 0; j = 0;
  while (dest[i] != EOS)
    i++;
  do {
    if (j >= n) break;
    tmp = src[j];
    /* replace this line.... */
    dest[i] = tmp;
    i++; j++;
  } while (src[j] != EOS);

  /* strncat man page says that strcat null-terminates dest */
  /* r_strncat RELEVANT */
  dest[i] = EOS;

  return dest;
}

char *
realpath(const char *pathname, char *result, char* chroot_path)
{
  char curpath[MAXPATHLEN],
    workpath[MAXPATHLEN],
    namebuf[MAXPATHLEN];

  if (result == NULL)
    return(NULL);

  if(pathname == NULL){
    *result = EOS; 
    return(NULL);
  }

  workpath[MAXPATHLEN-1] = EOS;
  strcpy(curpath, pathname);
  strcpy(namebuf, workpath);

  /* OK */
  r_strncat(namebuf, curpath, MAXPATHLEN-strlen(namebuf)-1);

  return result;
}

int main ()
{
  char pathname [MAXPATHLEN];
  char result [MAXPATHLEN];
  char chroot_path [MAXPATHLEN];

  /* Don't use too big a pathname; we're not trying to overflow curpath */
  pathname [MAXPATHLEN-1] = EOS;
  result [MAXPATHLEN-1] = EOS;
  chroot_path [MAXPATHLEN-1] = EOS;

  realpath(pathname, result, chroot_path);

  return 0;
}
