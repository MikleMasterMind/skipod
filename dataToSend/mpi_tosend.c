// compile -> mpicc mpi.c -o mpi
// run -> mpirun -np 4 ./mpi

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define Min(a,b) ((a)<(b)?(a):(b))
#define INDEX(i, j, k) (((i + 1 - start) * N + j) * N + k)
double maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;
double *A, *B;
int rank, size, segmentsize;
int start, end;
int N;

void init();
void relax();
void resid();
void verify();
void synchronize_data();

int main(int argc, char **argv)
{
    N = strtol(argv[1], 0, 10);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    struct timeval starttime, stoptime;
    if (rank == 0)
    {
        gettimeofday(&starttime, NULL);
    }

    segmentsize = N % size > rank ? N / size + 1 : N / size;
    A = (double*)malloc(N * N * (segmentsize + 2) * sizeof(double));
    B = (double*)malloc(N * N * (segmentsize + 2) * sizeof(double));

    start = rank * (N / size) + Min(rank, N % size);
    end = start + segmentsize - 1;

    init();

    int it;
    for(it=1; it<=itmax; it++)
    {
        synchronize_data();
        eps = 0.;
        relax();
        resid();

        if (rank == 0)
        {
            //printf( "it=%d   eps=%f\n", it,eps);
        }

        if (eps < maxeps) 
        {
            break;
        }
    }
    
    verify();

    if (rank == 0)
    {
        gettimeofday(&stoptime, NULL);
        long sec = stoptime.tv_sec - starttime.tv_sec;
        long msec  = stoptime.tv_usec - starttime.tv_usec;
        printf("%f\n", sec + msec * 1e-6);
    }

    free(A);
    free(B);

    MPI_Finalize();
	return 0;
}

void init()
{
	for(i = start; i <= end; i++)
    for(j = 0; j <= N - 1; j++)
    for(k = 0; k <= N - 1; k++)
	{
        if(i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
        {
            A[INDEX(i, j, k)] = 0.;
        }
		else 
        {
            A[INDEX(i, j, k)] = (4. + i + j + k);
        }
	}
} 

void relax()
{
    int shift_start = rank == 0 ? 1 : 0;
    int shift_end = rank == size - 1 ? -1 : 0;
    for(i = start + shift_start; i <= end + shift_end; i++)
    for(j = 1; j <= N - 2; j++)
    for(k = 1; k <= N - 2; k++)
	{
        B[INDEX(i, j, k)] = 
           (A[INDEX(i - 1, j, k)] + A[INDEX(i + 1, j, k)] + 
            A[INDEX(i, j - 1, k)] + A[INDEX(i, j + 1, k)] +
            A[INDEX(i, j, k - 1)] + A[INDEX(i, j, k + 1)]) / 6.;
	}
}

void resid()
{
    double tmp = 0.;
    double e;
	for(i = start; i <= end; i++)
    for(j = 1; j <= N - 2; j++)
    for(k = 1; k <= N - 2; k++)
	{
		e = fabs(A[INDEX(i, j, k)] - B[INDEX(i, j, k)]);         
		A[INDEX(i, j, k)] = B[INDEX(i, j, k)]; 
		tmp = Max(tmp, e);
	}
    MPI_Allreduce(&tmp, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void verify()
{
	double global_s;
    double s = 0.;
	for(i = start; i <= end; i++)
    for(j = 0; j <= N - 1; j++)
    for(k = 0; k <= N - 1; k++)
	{
		s += A[INDEX(i, j, k)] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
	}

    MPI_Reduce(&s, &global_s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
	    // printf("  S = %f\n",global_s);
    }
}

void synchronize_data(){
    MPI_Request request;
    MPI_Status status;
    if (rank != 0)
    {
        MPI_Isend(A + N * N, N * N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request);
    }
    if (rank != size - 1)
    {
        MPI_Isend(A + segmentsize * N * N, N * N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request);
    }
    if (rank != 0)
    {
        MPI_Recv(A , N * N, MPI_DOUBLE, rank - 1 , 0 , MPI_COMM_WORLD , &status);
    }
    if (rank != size - 1)
    {
        MPI_Recv(A + (segmentsize + 1) * N * N, N * N, MPI_DOUBLE, rank + 1 , 0 , MPI_COMM_WORLD , &status);
    }
}
