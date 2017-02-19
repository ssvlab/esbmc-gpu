#include<stdlib.h>
typedef struct {
   int a;
   int b;
} f_t;

int bar(f_t * w, int n){
   int i;
   for (i=0; i < n ; ++i){
      w[i].a=-1;
      w[i].b=-2;
   }
   return 1;
}
int foo(int * y, int n){
   int i;
   for (i=0; i < n ; ++i){
      y[i]=-1;
   }
   return 1;
}
int nondet_int();
int main(){

   f_t * x, *z;
   int * y, *w;
   int n;
   n = nondet_int(); //__NONDET__();
   __ESBMC_assume(n>0); //fSfT__assume(n > 0 );
   __ESBMC_assume(n<100); //fSfT__assume( n < 100);
   x = (f_t*) malloc(n * sizeof(f_t));
   y = (int *) x;
   __ESBMC_assume(y != (int *) 0); //fSfT__assume(y != (int *) 0);
   foo(y,2*n);
   w = (int *) malloc( 2*n * sizeof(int));
   z = (f_t*) w;
   __ESBMC_assume(z != (int *) 0); //fSfT__assume( z != (int *) 0);
   bar(z,n);

   return 1;
}

