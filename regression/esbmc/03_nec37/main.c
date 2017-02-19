#include <stdlib.h>
#include <limits.h>
typedef struct foo{

   int x;
   int z;

} foo_t;


int main(){
   int n;
   foo_t * a;
   int * b;
   __ESBMC_assume(n>0 && n < (INT_MAX/sizeof(foo_t)));
   //fSfT__assume(n > 0);

   a = (foo_t*) malloc(n * sizeof(foo_t));
   __ESBMC_assume( a != (foo_t*)0);
   //fSfT__assume( a != (foo_t*)0);
   b = (int *) a; /*-- down casting. Length(b) = 2 * Length(a) --*/

   b[2*n -1] = '\0';


   return 1;
}
