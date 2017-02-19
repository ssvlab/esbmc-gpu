
int x,y;

int main(){
   int a[20];
   __ESBMC_assume(x >= 0);
   __ESBMC_assume(y >= 0);
   __ESBMC_assume(x< 9);
   __ESBMC_assume(y < 10);

   if (x * y - x*x >= 50){
      x=x+1;
   }

   a[x]=1;
   return 1;
}
