#include <stdlib.h>
#include <assert.h>
#include <string.h>

int my_malloc( unsigned int size, void * * b){
   *b = malloc(size);
   if (*b) return 1;
   return 0;

}


int my_free( unsigned int size, void * c){
   if (c) {
      free(c);
      return 1;
   }

   return 0;

}




int main(){
   int * x;
   int * a;
   int retval = my_malloc(100, &a);
   if (retval == 1){
      retval = my_malloc(100, &x);
      if (retval != 1){
         my_free(100,x);
         return 0;
      }
   } else {
      my_free(100,a);
      return 0;
   }

   assert(a);
   assert(x);

   memcpy(a,x,100);

   return 1;
}
