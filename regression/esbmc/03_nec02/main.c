#include <stdlib.h>

int *a;

int test(int * n ){
//   assert(*n==1);
   int i;
   for (i = 0; i < (*n); ++i){
      a[i] = 0;            
   }


   return 1;
}

unsigned int nondet_uint();
int nondet_int();

int main(){

   int n = nondet_int(); //__NONDET__();
   
   if (n <= 0 || n >= 10){
      exit(1);
   } else {
      a = (int *) malloc( n * sizeof(int));
      __ESBMC_assume(a);
   }

   //__ESBMC_assume(a);

   test(&n);

   return 1;
}
