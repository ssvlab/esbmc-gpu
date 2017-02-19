
#include <string.h>

#if 0
unsigned int strlen (const char * str)
{
  int n;

  n = 0;
  while (*str++)
	n++;
  return(n);
}
#endif

int main()
{
  _Bool C_plus_plus;
  char *program_name = "flex";

  program_name = "Executable";

	if ( program_name[0] != '\0' &&
	     program_name[strlen( program_name ) - 1] == '+' )
		C_plus_plus = 1;

  return 0;
}
