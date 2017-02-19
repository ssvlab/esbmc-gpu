/*--
  laplace.c
  Taken from Prof. Goubault page.
  --*/

/* size of the matrix */
#define N 10
#define epsilon 0.005
typedef float NUM;
NUM A[N][N];
NUM b[N];
NUM gi[N];
NUM gsi[N];
NUM hi[N];
NUM hsi[N];
NUM xi[N];
NUM xsi[N];
NUM gamma;
NUM rho;

void evalA(NUM *x, NUM *y) {
  /* computes y=Ax */
  int i,j;
  for (i=0;i<N;i++) {
    y[i] = 0;
    for (j=0;j<N;j++)
      y[i] = y[i]+A[i][j]*x[j];
  }
}

NUM scalarproduct(NUM *x, NUM *y) {
  /* computes (x,y) */
  int i;
  NUM res;
  res = 0;
  for (i=0;i<N;i++)
    res = res+x[i]*y[i];
  return res;
}

void multadd(NUM *x, NUM *y, NUM alpha, NUM beta, NUM *z) {
  /* computes z=alpha*x+beta*y */
  int i;
  for (i=0;i<N;i++) {
    z[i] = alpha*x[i]+beta*y[i];
  }
}

int main() {
  int i,j,k;
  NUM temp[N];
  NUM norm, norm2, beta;

  /* init A - discretisation du laplacien en dimension un */
  for (i=0;i<N;i++) {
      A[i][i] = __BUILTIN_DAED_FBETWEEN(2.0/(N+1)-0.0000001,2.0/(N+1)+0.0000001);
      if (i < N-1) {
         A[i][i+1] = -1.0/(N+1);
         A[i+1][i] = -1.0/(N+1);
      }
  }

  /* init B */
  for (i=0;i<N;i++)
    b[i] = 1;

  /* init x */
  for (i=0;i<N;i++)
    xi[i] = __BUILTIN_DAED_FBETWEEN(0,0.0000001);

  /* init solution */
  evalA(xi,temp);
  multadd(b,temp,1,-1,gi); /* g0=b-Ax */

  for (j=0;j<N;j++)
    hi[j] = gi[j];

  norm = scalarproduct(gi,gi);

  k = 0;
  /* conjugate gradient algorithm */
  while (norm > epsilon) {
/*   if (k>=1)
      norm = __BUILTIN_DAED_FPRINT(norm,norm);*/
    evalA(hi,temp); 
    rho = scalarproduct(hi,temp); 
    norm2 = norm;
    gamma = norm2/rho;
    multadd(xi,hi,1,gamma,xsi);
    multadd(gi,temp,1,-gamma,gsi);
    norm = scalarproduct(gsi,gsi); 
    beta = norm/norm2;
    multadd(gsi,hi,1,beta,hsi);

    for (j=0;j<N;j++) {
      xi[j] = xsi[j];
      gi[j] = gsi[j];
      hi[j] = hsi[j];
    }

    k++;
  }

  for (j=0;j<N;j++)
    xi[j] = __BUILTIN_DAED_FPRINT(xi[j],xi[j]);

  evalA(xi,temp);
  for (j=0;j<N;j++)
     temp[j] = __BUILTIN_DAED_FPRINT(temp[j],temp[j]);
}
