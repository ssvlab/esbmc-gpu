#include <stdlib.h>

int main(){

   int * a;
   int k = nondet_int(); //__NONDET__();
   int i;

   __ESBMC_assume(k > 0 && (k < 0x1FFFFFFF));
   
   a= malloc( k * sizeof(int));
   __ESBMC_assume(a);
   
   for (i =0 ; i != k; i++)
      if (a[i]) return 1;

   return 0;

}
