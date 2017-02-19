/*-- from blast tacas 06 examplex --*/
extern int __NONDET__();


int main(){
  
  int x,y;
  
  x = 0; y = 0;
  
  while(__NONDET__()){
    x++; y++;
  }
  
  while(x > 0){
    x--;
    y--;
  }
  
  if(y != 0){
     assert(1==0);
  }

  return 1;

}
