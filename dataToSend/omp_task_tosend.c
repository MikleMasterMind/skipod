// compile with -fopenmp
// solve var32.c using omp task

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

// #define N 66
double   maxeps = 0.1e-7;
int itmax = 100;
int tred;
int i,j,k, it;
double eps;
double* local_eps;
double*** A,  ***B;
int N;

void relax();
void resid();
void init();
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
    local_eps = (double*)malloc(tred*sizeof(double));

    double start = omp_get_wtime();
    
    #pragma omp parallel num_threads(tred)
    {
        #pragma omp master
        {
            init();
            for(it=1; it<=itmax; it++)
            {
                eps = 0.;
                #pragma omp taskwait
                relax();
                #pragma omp taskwait
                resid();
                #pragma omp taskwait
                //printf( "it=%4i   eps=%f\n", it,eps);
                if (eps < maxeps) break;
            }
        }
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
    free(local_eps);
    
	return 0;
}

void init()
{
    int i, j, k;
    for(k=0; k<=N-1; k++)
    for(j=0; j<=N-1; j++)
    for(i=0; i<=N-1; i++)
    {
        if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) A[k][j][i]= 0.;
        else A[k][j][i]= ( 4. + i + j + k) ;
    }
} 

void relax()
{
    int i, j, k;
    for(k=1; k<=N-2; k++)
    #pragma omp task firstprivate(i, j, k)
    {
        for(j=1; j<=N-2; j++)
        for(i=1; i<=N-2; i++)
        {
            if (it % 2 == 1)
                B[k][j][i]=(A[k-1][j][i]+A[k+1][j][i]+A[k][j-1][i]+A[k][j+1][i]+A[k][j][i-1]+A[k][j][i+1])/6.;
            else
                A[k][j][i]=(B[k-1][j][i]+B[k+1][j][i]+B[k][j-1][i]+B[k][j+1][i]+B[k][j][i-1]+B[k][j][i+1])/6.;
        }
    }
}

void resid()
{ 
    int i, j, k;
	for(k=1; k<=N-2; k++)
    {
        #pragma omp task firstprivate(i, j, k, local_eps) 
        {
            int local_index = omp_get_thread_num();
            local_eps[local_index] = 0.;
            for(j=1; j<=N-2; j++)
            for(i=1; i<=N-2; i++)
            {
                double e;
                e = fabs(A[k][j][i] - B[k][j][i]);         
                if (it % 2 == 1)        
                    A[k][j][i] = B[k][j][i];
                else
                    B[k][j][i] = A[k][j][i];
                local_eps[local_index] = Max(local_eps[local_index], e);
            }
        }
    }
    #pragma omp taskwait
    
    for (i = 0; i < tred; ++i)
        eps = Max(eps, local_eps[i]);
}

void verify()
{
    int i, j, k;
	double s;
	s=0.;
	for(k=0; k<=N-1; k++)
    for(j=0; j<=N-1; j++)
    for(i=0; i<=N-1; i++)
    {
        s=s+A[k][j][i]*(i+1)*(j+1)*(k+1)/(N*N*N);
    }
	printf("  S = %f\n",s);
}
