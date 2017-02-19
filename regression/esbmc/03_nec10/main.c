struct _addr {
  unsigned char len;
  unsigned char dat[16];
};
typedef struct _addr Addr;

struct _buffer {
  int dummy;
};
typedef struct _buffer Buffer;

int f1(Addr *addr, Buffer *buf);
int f2(unsigned char val, Buffer *buf);
int f3(unsigned char data, Buffer *buf);

int main(void)
{
  // Start with free addr and buffer;
  Addr a;
  Buffer b;
   
   if (a.len < 0 || a.len >= 16) {
      return 0;
   }
   return f1(&a, &b);
}


int f1(Addr *addr, Buffer *buf) 
{
  int i;
  int ret1;
  int ret2;
  
  if ((int)addr->len > 16) {
    return 1;
  }

  i = (int)addr->len;
  
  while (i ) {
     ret1 = f2(addr->dat[i - 1], buf);
     if (ret1 != 0) {
        return ret1;
     }
     i = i - 1;
  }
  ret2 = f2(addr->len, buf);
  if (ret2 != 0) {
    return ret2;
  }
  
  return 0;
}

int f2(unsigned char val, Buffer *buf) 
{ 
  f3(val, buf);
  return 0;
}

int f3(unsigned char data, Buffer *buf) 
{ 
  return 0;
}

