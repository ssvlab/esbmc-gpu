#include <string.h>
#include <stdio.h>
#include <assert.h>

int main(){
   int m, n, k;
   char  s1[100], s2[100],  s3[200];
   __ESBMC_assume( m > 0);
   __ESBMC_assume( m < 100);

   s1[m-1]=0;
   s2[m-1]=0;

   strcpy(s3,s1);
   strcat(s3,s2);

#if 0
   assert( strlen(s3) <= 200 ); 
   assert(strlen(s3) <= 2 * m); 
#endif
   return 1;
}
