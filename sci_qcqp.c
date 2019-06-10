#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

//[xopt,fopt] = qcqp(x,H,f,m,A,b,p,Aeq,beq,q,Q,c,r,lb,ub)

struct problem
{
   /* data */
   //Objective function : x'Hx+f'x
double ** H;
double * f;
//No of variables
int n;

//Linear inequality constraint Ax <= b
double * A;
double * b;
//No of Linear inequality Constraints
int m;

//Linear equality constraint Aeqx = b
double * Aeq;
double * beq;
//No of Linear equality constraint 
int p;

//Quadratic inequality constraint x'Qx + c'x <= r
double *** Q;
double ** c;
double *r; 
//No of Quadratic Constraint
int q;

//Lower and Upper bounds
double * lb;
double * ub;

};

extern problem;

void c_algencan(void *myevalf, void *myevalg, void *myevalh, void *myevalc,
	void *myevaljac, void *myevalhc, void *myevalfc, void *myevalgjac,
	void *myevalgjacp, void *myevalhl, void *myevalhlp, int jcnnzmax,
       	int hnnzmax,double *epsfeas, double *epsopt, double *efstin,
	double *eostin, double *efacc, double *eoacc, char *outputfnm,
	char *specfnm, int nvparam,char **vparam, int n, double *x,
	double *l, double *u, int m, double *lambda, _Bool *equatn,
	_Bool *linear, _Bool *coded, _Bool checkder, double *f,
	double *cnorm, double *snorm, double *nlpsupn,int *inform);

/* ******************************************************************
    Objective Function. Must be Quadratic.
    f(x) = x'.H.x + f'.x 
   ****************************************************************** */

void myevalf(int n, double *x, double *f_, int *flag) {
   int i = 0;
   int j = 0;

   *flag = 0;
   *f_ = 0;

   for( i = 0 ; i < n ; i++ )
   {
      for( j = 0 ; j < n ; j++ ){
         *f_ += x[i]*problem.H[i][j]*x[j];
      }
   }   

   for( i = 0 ; i < n ; i++ ){
      *f_ += x[i]*problem.f[i];
   }

}

/* ******************************************************************
    Gradient of Objective function.
    f'(x)  
   ****************************************************************** */

void myevalg(int n, double *x, double *g, int *flag) {
   int i = 0;
   int j = 0;
   *flag = 0;
   for( i = 0 ; i < n ; i++ ){
      g[i] = 0;
      for( j = 0 ; j < n ; j++ ){
            g[i] += problem.H[i][j]*x[j];
      }
      g[i] += problem.H[i][i]*x[i] + problem.f[i]; 
   }
}

/* ******************************************************************
    Hessian of Objective function.
    f''(x) 
   ****************************************************************** */

void myevalh(int n, double *x, int *hrow, int *hcol, double *hval, int *hnnz,
	     int lim, _Bool *lmem, int *flag) {

   *flag = 0;
   *lmem = 0;

   int i = 0;
   int j = 0;

   *hnnz = (n*(n+1))/2;
   if( *hnnz > lim ) {
     *lmem = 1;
     return;
   }

   for( int i =  0 ; i < n ; i++ ){
      hrow[i] = i;
      for( int j = 0 ; j <= i ; j++ ){
         hcol[j] = j;
         hval[i+j] = 2*problem.H[i][j];
      }
   }
}

/* ******************************************************************
    Equality and Inequality Constraints
   ****************************************************************** */

void myevalc(int n, double *x, int ind, double *c, int *flag) {

   *flag = 0;
   int i = 0;
   int j = 0;
   *c = 0;
   if(ind < problem.m){
      for( i = 0 ; i < n ; i++ ){
         *c += problem.A[ind][i] * x[i];
      }
   }else if(ind < problem.p){
      for( i = 0 ; i < n ; i++ ){
         *c += problem.Aeq[ind-problem.m][i] * x[i];
      }
   }else{
      for( i = 0 ; i < n ; i++ )
      {
         for( j = 0 ; j < n ; j++ ){
            *c += x[i]*problem.Q[ind - problem.m - problem.p][i][j]*x[j];
         }
      }   

      for( i = 0 ; i < n ; i++ ){
         *c += x[i]*problem.f[ind - problem.m - problem.p];
      }
   }
}

/* ******************************************************************
    Gradient of Constraints
   ****************************************************************** */

void myevaljac(int n, double *x, int ind, int *jcvar, double *jcval,
	       int *jcnnz, int lim, _Bool *lmem, int *flag) {
  *flag = 0;
  *lmem = 0;
   int i = 0;
   int j = 0;
   
   *jcnnz = n;
   if( *jcnnz > lim ) {
     *lmem = 1;
     return;
   }

   if(ind < problem.m){
      for( i = 0 ; i < n ; i++ ){
         jcvar[i]=i;
         jcval[i] = problem.A[ind][i];
      }
   }else if(ind < problem.p){
      for( i = 0 ; i < n ; i++ ){
         jcvar[i]=i;
         jcval[i] = problem.Aeq[ind-problem.m][i];
      }
   }else{
      for( i = 0 ; i < n ; i++ ){
         jcval[i] = 0;
         jcvar[i]=i;
         for( j = 0 ; j < n ; j++ ){
               jcval[i] += problem.Q[i][j]*x[j];
         }
         jcval[i] += problem.Q[i][i]*x[i] + problem.c[i]; 
      }
   }
}

/* ******************************************************************
    Hessian of Constraints
   ****************************************************************** */

void myevalhc(int n, double *x, int ind, int *hcrow, int *hccol, double *hcval,
	      int *hcnnz, int lim, _Bool *lmem, int *flag) {
  *flag = 0;
  *lmem = 0;
   int i = 0;
   int j = 0;
   
   
   
   if(ind < problem.m + problem.p){
      *hcnnz = 0;
   }else{
      *hcnnz = (n*(n+1))/2;
      if( *hcnnz > lim ) {
        *lmem = 1;
         return;
      }
      for( int i =  0 ; i < n ; i++ ){
         hcrow[i] = i;
         for( int j = 0 ; j <= i ; j++ ){
            hccol[j] = j;
            hcval[i+j] = 2*problem.Q[i][j];
         }
      }
      
   }  
}

/* *****************************************************************
   ***************************************************************** */

void myevalfc(int n, double *x, double *f, int m, double *c, int *flag) {

   *flag = -1;
}

/* *****************************************************************
   ***************************************************************** */

void myevalgjac(int n, double *x, double *g, int m, int *jcfun, int *jcvar,
		double *jcval, int *jcnnz, int lim, _Bool *lmem, int *flag) {

   *flag = -1;
}

/* *****************************************************************
   ***************************************************************** */

void myevalgjacp(int n, double *x, double *g, int m, double *p, double *q,
		 char work, _Bool *gotj, int *flag) {

   *flag = -1;
}

/* *****************************************************************
   ***************************************************************** */

void myevalhl(int n, double *x, int m, double *lambda, double scalef,
	      double *scalec, int *hlrow, int *hlcol, double *hlval,
	      int *hlnnz, int lim, _Bool *lmem, int *flag) {

   *flag = -1;
}

/* *****************************************************************
   ***************************************************************** */

void myevalhlp(int n, double *x, int m, double *lambda, double scalef,
	       double *scalec, double *p, double *hp, _Bool *goth, 
	       int *flag) {

   *flag = -1;
}

/* ******************************************************************
   ****************************************************************** */

int main() {
  _Bool  checkder;
  int    hnnzmax,hnnzmax1,hnnzmax2,hnnzmax3,i,jcnnzmax,inform,m,n,nvparam,ncomp;
  double cnorm,efacc,efstin,eoacc,eostin,epsfeas,epsopt,f,nlpsupn,snorm;
  
  char   *specfnm, *outputfnm, **vparam;
  _Bool  coded[11],*equatn,*linear;
  double *l,*lambda,*u,*x;
  
  n = 2;
  m = 2;
  
  /* Memory allocation */
  x      = (double *) malloc(n * sizeof(double));
  l      = (double *) malloc(n * sizeof(double));
  u      = (double *) malloc(n * sizeof(double));
  lambda = (double *) malloc(m * sizeof(double));
  equatn = (_Bool  *) malloc(m * sizeof(_Bool ));
  linear = (_Bool  *) malloc(m * sizeof(_Bool ));
  
  if (     x == NULL ||      l == NULL ||      u == NULL ||
      lambda == NULL || equatn == NULL || linear == NULL ) {
    
    printf( "\nC ERROR IN MAIN PROGRAM: It was not possible to allocate memory.\n" );
    exit( 0 );
    
  }

  /* Initial point */
  for(i = 0; i < n; i++) x[i] = 0.0;
  
  /* Lower and upper bounds */
  l[0] = - 10.0;
  u[0] =   10.0;
  l[1] = - 1.0e20;  
  u[1] =   1.0e20;
  
  /* For each constraint i, set equatn[i] = 1. if it is an equality
     constraint of the form c_i(x) = 0, and set equatn[i] = 0 if it is
     an inequality constraint of the form c_i(x) <= 0. */
  equatn[0] = 0;
  equatn[1] = 0;

  /* For each constraint i, set linear[i] = 1 if it is a linear
     constraint, otherwise set linear[i] = 0 */
  linear[0] = 0;
  linear[1] = 1;
  
  /* Lagrange multipliers approximation. */
  for( i = 0; i < m; i++ ) lambda[i] = 0.0;
  
  /* In this C interface evalf, evalg, evalh, evalc, evaljac and
     evalhc are present. evalfc, evalgjac, evalhl and evalhlp are
     not. */
  
  coded[0]  = 1; /* fsub     */
  coded[1]  = 1; /* gsub     */
  coded[2]  = 1; /* hsub     */
  coded[3]  = 1; /* csub     */
  coded[4]  = 1; /* jacsub   */
  coded[5]  = 1; /* hcsub    */
  coded[6]  = 0; /* fcsub    */
  coded[7]  = 0; /* gjacsub  */
  coded[8]  = 0; /* gjacpsub */
  coded[9]  = 0; /* hlsub    */
  coded[10] = 0; /* hlpsub   */
 
  /* Upper bounds on the number of sparse-matrices non-null
     elements */
  jcnnzmax = 4;
  hnnzmax1 = 0;
  hnnzmax2 = 1;
  hnnzmax3 = 6;
  hnnzmax  = hnnzmax1 + hnnzmax2 + hnnzmax3;

  /* Check derivatives? */
  checkder = 0;

  /* Parameters setting */
  epsfeas = 1.0e-08;
  epsopt  = 1.0e-08;
  efstin  = sqrt( epsfeas );
  eostin  = pow( epsopt, 1.5 );
  efacc   = sqrt( epsfeas );
  eoacc   = sqrt( epsopt );

  outputfnm = "algencan.out";
  specfnm   = "";

  nvparam = 1;
  
  /* Allocates VPARAM array */
  vparam = ( char ** ) malloc( nvparam * sizeof( char * ) );

  /* Set algencan parameters */
  vparam[0] = "ITERATIONS-OUTPUT-DETAIL 10";

  /* Optimize */
  c_algencan(&myevalf,&myevalg,&myevalh,&myevalc,&myevaljac,&myevalhc,&myevalfc,
	     &myevalgjac,&myevalgjacp,&myevalhl,&myevalhlp,jcnnzmax,hnnzmax,
	     &epsfeas,&epsopt,&efstin,&eostin,&efacc,&eoacc,outputfnm,specfnm,
	     nvparam,vparam,n,x,l,u,m,lambda,equatn,linear,coded,checkder,
	     &f,&cnorm,&snorm,&nlpsupn,&inform);

  /* Memory deallocation */
  free(x     );
  free(l     );
  free(u     );
  free(lambda);
  free(equatn);
  free(linear);
}
