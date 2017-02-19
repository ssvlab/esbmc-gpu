#include <stdlib.h>
#include <limits.h>

int * main (){
   int x, y;
   int * a;
   int i;
   
   if ( x< 0 || y < 0 || y > x ) return (int *) 0;

   // Ensure allocation doesn't wrap around into negative range or otherwise
   // overflow.
   __ESBMC_assume(x < (INT_MAX / sizeof(int)));
   a = (int *) malloc( x * sizeof(int));

   if (a == 0 ) exit(1);
   
   for (i=0; i < y ; ++i){
      a[i] = 0;
   }

   return a;  
}


