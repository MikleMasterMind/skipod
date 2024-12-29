#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

//#define  N   1026
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
int N;
double eps;
double*** A,  ***B;

void relax();
void resid();
void init();
void verify(); 

int main(int an, char **as)
{
    N = strtol(as[1], 0, 10);

    A = (double***)malloc(N*sizeof(double**));
    B = (double***)malloc(N*sizeof(double**));
    for(i = 0; i < N; ++i)
    {
        A[i] = (double**)malloc(N*sizeof(double*));
        B[i] = (double**)malloc(N*sizeof(double*));
        for(j = 0; j < N; ++j) 
        {
            A[i][j] = (double*)malloc(N*sizeof(double));
            B[i][j] = (double*)malloc(N*sizeof(double));
        }
    }

	double start = omp_get_wtime();
	int it;
	init();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		//printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
	verify();
	printf("%f\n", omp_get_wtime()-start);

    for(i = 0; i < N; ++i)
    {
        for(j = 0; j < N; ++j) 
        {
            free(A[i][j]);
            free(B[i][j]);
        }
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);
    
	return 0;
}

void init()
{ 
	for(k=0; k<=N-1; k++)
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) A[i][j][k]= 0.;
		else A[i][j][k]= ( 4. + i + j + k) ;
	}
} 

void relax()
{
	for(k=1; k<=N-2; k++)
	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	{
		B[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1])/6.;
	}
}

void resid()
{ 
	for(k=1; k<=N-2; k++)
	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	{
		double e;
		e = fabs(A[i][j][k] - B[i][j][k]);         
		A[i][j][k] = B[i][j][k]; 
		eps = Max(eps,e);
	}
}

void verify()
{
	double s;
	s=0.;
	for(k=0; k<=N-1; k++)
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
	//printf("  S = %f\n",s);
}
