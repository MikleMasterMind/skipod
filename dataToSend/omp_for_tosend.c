// compile with -fopenmp
// solve var32.c using omp for

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

// #define NUM_THREADS 4
// #define N 66
double   maxeps = 0.1e-7;
int itmax = 100;
int tred;
int i,j,k;
double eps;
double*** A,  ***B;
int N, it;

void init();
void relax();
void resid();
void verify(); 

int main(int an, char **as)
{
	tred = strtol(as[1], 0, 10);
    N = strtol(as[2], 0, 10);

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
    init();
    for (it=1; (it<=itmax); it++)
    {
        eps = 0.;
        relax();
        resid();
        // printf( "it=%4i   eps=%f\n", it, eps);
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
        if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) A[k][j][i]= 0.;
        else A[k][j][i] = (4. + i + j + k);
    }
} 

void relax()
{

    #pragma omp parallel for private(i, j, k) shared(A, B) num_threads(tred)
	for(k=1; k<=N-2; k++)
	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	{
		if (it % 2 == 1)
			B[k][j][i]=(A[k-1][j][i]+A[k+1][j][i]+A[k][j-1][i]+A[k][j+1][i]+A[k][j][i-1]+A[k][j][i+1])/6.;
		else
			A[k][j][i]=(B[k-1][j][i]+B[k+1][j][i]+B[k][j-1][i]+B[k][j+1][i]+B[k][j][i-1]+B[k][j][i+1])/6.;
	}
}

void resid()
{
    #pragma omp parallel for private(i, j, k) shared(A, B) num_threads(tred) reduction(max: eps)
	for(k=1; k<=N-2; k++)
	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	{
		double e;
		e = fabs(A[k][j][i] - B[k][j][i]);         
		if (it % 2 == 1)        
			A[k][j][i] = B[k][j][i];
		else
			B[k][j][i] = A[k][j][i];
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
		s=s+A[k][j][i]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
	// printf("  S = %f\n",s);
}
