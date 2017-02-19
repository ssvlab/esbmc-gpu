#include <stdlib.h>
#include <limits.h>

typedef struct foo{
   
   int x;
   char * y;
   int z;

} foo_t;


foo_t * array;


int main(){
   int n;
   int alen;  
   int * x;

   __ESBMC_assume( n > 0 && n < (INT_MAX / sizeof(foo_t)));
   array = (foo_t *) malloc(n * sizeof(foo_t));
   __ESBMC_assume(array);
   memset(array,0, sizeof(foo_t)* n);
   /*-- check
     Length(array) * sizeof(*array) >= sizeof(foo_t) * n
     
     --*/
   
   return 1;
}
