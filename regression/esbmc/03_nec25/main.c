int a[100];

int foo(int x){
   int i;
   if (x == 0) return 1;
   ASSERT(x < 100);
   ASSERT(x > 0);

   for (i = 0; i < x; ++i)
      a[i]=0;
   return 0;
}


int main(){
   int y;
   int z;
   for (y=0; y < 10; ++y)
      foo(y);

   for (z=0; z < 100; ++z)
      foo(z);

   return 1;

}
