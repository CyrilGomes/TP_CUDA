/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

static long num_steps = 100000000;
static long thread_per_block = 1;
static long steps_per_thread = 64;
double step;

void write_perf_csv(int thread_per_block, int step_per_thread,int num_steps, double runtime){
  ofstream myfile;
  myfile.open ("stats_part1.csv", ios_base::app);
  myfile.precision(8);
  myfile <<"1_1 basic"<<"," << thread_per_block << ","<< step_per_thread <<","<< num_steps << "," << runtime << "\n";

  myfile.close();
}

__global__ void compute_pi(float* pi, long num_steps){
      int i;
      double x, sum = 0.0;
      int threadi = threadIdx.x + blockIdx.x * blockDim.x;
      int stride = blockDim.x * gridDim.x;
      double step = 1.0/(double) num_steps;

      for (i=threadi;i< num_steps; i+= stride){
        x = (i-0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
      }

      atomicAdd(pi, sum);


}

int main (int argc, char** argv)
{
    
      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        }
        if ( ( strcmp( argv[ i ], "-tpb" ) == 0 )) {
            thread_per_block = atol( argv[ ++i ] );
            printf( "  User thread per block is %ld\n", thread_per_block );
        } 
        if ( ( strcmp( argv[ i ], "-spt" ) == 0 )) {
            steps_per_thread = atol( argv[ ++i ] );
            printf( "  User steps per thread is %ld\n", steps_per_thread );
        } 
        else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      
    float pi_h = 0;

    float* pi_d;
    cudaMalloc((void **) &pi_d, sizeof(float) );
    cudaMemcpy(pi_d, &pi_h, sizeof(float), cudaMemcpyHostToDevice);

	  
    step = 1.0/(double) num_steps;

    int num_blocks = num_steps/(thread_per_block*steps_per_thread) + 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    printf("%d", num_blocks);
    compute_pi<<<num_blocks, thread_per_block>>>(pi_d, num_steps);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;

    cudaMemcpy(&pi_h, pi_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsedTime, start, stop);

	  pi_h = step * pi_h;

    
    printf("\n pi with %ld steps is %lf in %lf s\n",num_steps,pi_h,elapsedTime/1000.0);
    write_perf_csv(thread_per_block,steps_per_thread,num_steps, elapsedTime/1000.0);
    cudaFree(pi_d);
}
