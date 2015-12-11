//question:
//does contention for accessing global memory affect read performance?
//for instance, n threads access each of the n rows in a column at the same time.
//would it be faster for thread i to access row i % n first and proceed to i + j % n?

#define TILE_WIDTH 32
#define DIVIDE_ROUND_UP(a, b)((a + b - 1) / b)
#define GET_INDEX(row, column, numcols)(row * numcols + column)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


//define matrix type
typedef struct{
  int row_count;
  int column_count;
  int* elements;
} Matrix;

typedef struct{
  int bin_count;
  int bin_width;
  int* counts;
} Histogram;


Matrix ones(int row_count, int column_count);
Matrix random(int row_count, int column_count);
Histogram make_histogram(Matrix image);
void print_hist(Histogram hist);

void print_matrix(Matrix mat);

int main(){
  //make the matrices you want to multiply
  srand(time(NULL));
  Matrix image = random(512, 512);
  Histogram result = make_histogram(image);
  print_hist(result);

}

//global memory
__global__ void global_atomic_histogram(const Matrix image, Histogram hist){
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;
  int index = row_index * image.column_count + column_index;
  int value = image.elements[index];
  int bin = value / hist.bin_width;
  atomicAdd(&(image.elements[index]), hist.counts[bin]);
  __syncthreads();

}

//shared memory
__global__ void local_atomic_histogram(const Matrix image, Histogram hist){
//todo:
}

Histogram make_histogram(Matrix image){
  cudaError_t error;
  //step 1: allocate memory on the kernel for matrix
  Matrix image_d;
  image_d.row_count = image.row_count;
  image_d.column_count = image.column_count;
  size_t image_size = image.row_count * image.column_count * sizeof(int);
  error = cudaMalloc((void**) &image_d.elements, image_size);
  if(error != cudaSuccess){
    printf("error allocating image matrix\n");
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  //step 2: allocate memory on host and device for histogram
  Histogram hist, hist_d;
  hist_d.bin_count = hist.bin_count = 200;
  hist_d.bin_width = hist.bin_width = 1;
  size_t hist_size = hist_d.bin_count * sizeof(int);
  error = cudaMalloc((void**) &hist_d.counts, hist_size);
  if(error != cudaSuccess){
    printf("error allocating histogram\n");
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  hist.counts = (int*) malloc(hist_size);

  //step 3: initialize histogram counts
  for(int i = 0; i < hist.bin_count; i++){
    hist.counts[i] = 0;
  }

  //step 4: copy image to device
  error = cudaMemcpy(image_d.elements, image.elements, image_size, cudaMemcpyHostToDevice);
  if(error != cudaSuccess){ printf("error copying matrix\n"); }
  //step 5: copy histogram zeros do device
  error = cudaMemcpy(hist_d.counts, hist.counts, hist_size, cudaMemcpyHostToDevice);
  if(error != cudaSuccess){ printf("error copying histogram\n"); }

  //step 4: launch kernel

  dim3 block_dims(TILE_WIDTH, TILE_WIDTH);
  dim3 grid_dims(DIVIDE_ROUND_UP(image_d.column_count, block_dims.x), DIVIDE_ROUND_UP(image_d.row_count, block_dims.y));
  global_atomic_histogram <<<grid_dims, block_dims>>> (image_d, hist_d);

  //step 5: copy results back to host
  error = cudaMemcpy(hist.counts, hist_d.counts, hist_size, cudaMemcpyDeviceToHost);
  if(error != cudaSuccess){
  	printf("error copying result histogram\n");
  	printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  return hist;
}

Matrix ones (int row_count, int column_count){
  Matrix result;
  result.row_count = row_count;
  result.column_count = column_count;
  result.elements = (int*) malloc(row_count * column_count * sizeof(int));
  for(int i = 0; i < row_count * column_count; i++){
    result.elements[i] = 1;
  }
  return result;
}

Matrix random (int row_count, int column_count){
  Matrix result;
  result.row_count = row_count;
  result.column_count = column_count;
  result.elements = (int*) malloc(row_count * column_count * sizeof(int));
  for(int i = 0; i < row_count * column_count; i++){
    result.elements[i] = rand() % 200;
  }
  return result;
}

void print_matrix(Matrix mat){
  int num_elements = mat.row_count * mat.column_count;
  for(int i = 0; i < num_elements; i++){
    printf(" %d", mat.elements[i]);
    if(!((i + 1) % mat.column_count)){ printf("\n"); }
  }
}

void print_hist(Histogram hist){
  for(int i = 0; i < hist.bin_count; i++){
    printf(" %d", hist.counts[i]);
  }
}
