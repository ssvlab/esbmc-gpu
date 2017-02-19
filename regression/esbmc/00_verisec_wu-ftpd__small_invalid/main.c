#include "../wu-ftpd.h"

char pathspace[ MAXPATHLEN ];
char old_mapped_path[ MAXPATHLEN ];

char mapped_path[ MAXPATHLEN ] = "/";

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

int mapping_chdir(char *orig_path)
{
  int ret;
  char *sl, *path;

  strcpy( old_mapped_path, mapped_path );
  path = &pathspace[0];
      
  /* BAD */
  r_strcpy( path, orig_path );

  return ret;
}

int main ()
{
  char in [INSZ];
  in [INSZ-1] = EOS;

  mapping_chdir (in);

  return 0;
}
