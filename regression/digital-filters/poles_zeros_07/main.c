#include<assert.h>

int main()
{
  float a[] =  {

  

   1.000000000000000,  -2.000000000000000,   1.968750000000000 , -2.000000000000000 ,  1.968750000000000,  -2.000000000000000,

  

   1.343750000000000 , -0.531250000000000,   0.156250000000000,  -0.031250000000000,                   0 ,                  0,

  

                   0


};
 float b[] =  {

  

                   0,                  0,                   0,   0.031250000000000,   0.062500000000000,   0.093750000000000,

  

   0.093750000000000,   0.093750000000000,   0.062500000000000,   0.031250000000000 ,                  0,                   0,

  

                   0
};
  assert(__ESBMC_check_stability(a, b));
  return 0;
}
