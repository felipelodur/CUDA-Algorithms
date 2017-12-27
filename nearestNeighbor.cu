#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>
#define BLOCK_SIZE 1024

__global__ void distance(int * gax, int * gay, int * gac, float  * gdistances, int * gcategoria, int len, int x, int y) {

   unsigned int t = threadIdx.x;
   unsigned int b = blockIdx.x*blockDim.x;

          __shared__ float partialDis[BLOCK_SIZE];
          __shared__ float partialAC[BLOCK_SIZE];

        partialDis[threadIdx.x] = sqrtf(powf(x-gax[t + b],2) + powf(y-gay[t + b],2));
        partialAC[threadIdx.x] = gac[threadIdx.x + b];

        __syncthreads();
        int metade = 512;
        if((b + threadIdx.x) < len){
        for(unsigned int i = 0; i<metade;metade = metade/2 ){
                if(threadIdx.x < metade){
                        if(partialDis[threadIdx.x] > partialDis[threadIdx.x+metade]){
                                partialDis[threadIdx.x] = partialDis[threadIdx.x+metade];
                                partialAC[threadIdx.x] = partialAC[threadIdx.x+metade];


                        }
                        __syncthreads();
                }
        }
        }
        __syncthreads();
        if(t == 0){
                gdistances[blockIdx.x] = partialDis[0];
                gcategoria[blockIdx.x] = partialAC[0];
        }

}


int main(int argc, char ** argv) {

   FILE * fp;
  int x,y,t;
  unsigned int i;
  int *ax, *ay, *ac;
  int ox, oy, oc;
  float *distances;
  float sDistance;

  int *gax, *gay, *gac;
  float *gdistances;
  int * gcategoria;
  /* Reading inputs */

   fp = fopen ("input.txt", "r");
   fscanf(fp,"%d %d",&x,&y);
   fscanf(fp,"%d",&t);

   ax = (int*) malloc(t*sizeof(int));
   ay = (int*) malloc(t*sizeof(int));
   ac = (int*) malloc(t*sizeof(int));
   int  * test  = (int*) malloc(t * sizeof(int));
   float * test2 = (float*) malloc( (ceil(t/BLOCK_SIZE) * sizeof(float)));

   cudaMalloc((void**) &gax, t * sizeof(int));
   cudaMalloc((void**) &gay, t * sizeof(int));
   cudaMalloc((void**) &gac, t * sizeof(int));
   cudaMalloc((void**) &gdistances, t *  sizeof(float));
   cudaMalloc((void**) &gcategoria, ceil(t/BLOCK_SIZE) * sizeof(int));

  for(i=0; i < t; i++)
    fscanf(fp,"%d %d %d",&ax[i],&ay[i],&ac[i]);

   fclose(fp);
   cudaMemcpy(gax, ax, t, cudaMemcpyHostToDevice);
   cudaMemcpy(gay, ay, t, cudaMemcpyHostToDevice);
   cudaMemcpy(gac, ac, t, cudaMemcpyHostToDevice);

   //@@ Define grid and block sizes  
   dim3 DimGrid(ceil(t/BLOCK_SIZE), 1, 1);
   dim3 DimBlock(BLOCK_SIZE, 1, 1);
   distance<<<DimGrid, DimBlock>>>(gax, gay, gac, gdistances, gcategoria, t, x, y);

   cudaMemcpy(ax, gax, t, cudaMemcpyDeviceToHost);
   cudaMemcpy(ay, gay, t, cudaMemcpyDeviceToHost);
   cudaMemcpy(ac, gac, t, cudaMemcpyDeviceToHost);
   cudaMemcpy(test, gdistances, ceil(t/BLOCK_SIZE), cudaMemcpyDeviceToHost);
   cudaMemcpy(test2, gcategoria, ceil(t/BLOCK_SIZE), cudaMemcpyDeviceToHost);
   cudaFree(gax);
   cudaFree(gay);
   cudaFree(gac);
   cudaFree(gdistances);
   cudaFree(gcategoria);

  double start = omp_get_wtime();

  /* Calculating nearest neighbor */
  sDistance = FLT_MAX;
  distances = (float*) malloc(ceil(t/BLOCK_SIZE)*sizeof(float));

  for(i=0; i < ceil(t/BLOCK_SIZE); i++) {
    if (sDistance > test[i]) {
         sDistance = distances[i];
         oc = test2[i];
    }
  }

  double end = omp_get_wtime();
  printf("\nTime = %f",end-start);

  printf("\nCategory = %d\n",oc);



        return 0;
}
~                                                                                               
