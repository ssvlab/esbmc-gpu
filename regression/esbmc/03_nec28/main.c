#include <string.h>


int main(){
   char * x = "a";
   char y[100];
   int i;
   memset(y,0,100);
   for (i=0; i < 100; ++i)
      strcat(y,x);
   return 1;
}


/* 
   Benchmark ex28.c comment 
   (added in version 1.1, January 2011, by Franjo Ivancic, ivancic@nec-labs.com)

   At each iteration of the loop, we add the string "a" and a null-terminating character, 
   per the specification of strcat. This is fine for the first 99 iterations. However, 
   at the 100th iteration, when i==99, we add 'a' into y[99], and then add the 
   null-terminating character into position y[100] causing a buffer overflow. 

   Thanks to Lucas Cordeiro for pointing out the omission of this bug in the bugs file. 
*/
