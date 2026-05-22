/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>

// Define some error checking macros.
#define cudaErrCheck(stat)                                                     \
  {                                                                            \
    cudaErrCheck_((stat), __FILE__, __LINE__);                                 \
  }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file,
            line);
  }
}

#define cublasErrCheck(stat)                                                   \
  {                                                                            \
    cublasErrCheck_((stat), __FILE__, __LINE__);                               \
  }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
  }
}

#define curandErrCheck(stat)                                                   \
  {                                                                            \
    curandErrCheck_((stat), __FILE__, __LINE__);                               \
  }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define SIZE 1024
#define MATRIX_M SIZE
#define MATRIX_N SIZE
#define MATRIX_K SIZE

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Naive matrix-multiply in a 2D configuration
__global__ void matrixMultKernel(half *a, half *b, float *c, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
      half fa = a[col * n + k];
      half fb = b[k * n + row];
      half fc = __hmul(fa, fb);
      float ffc = __half2float(fc);
      sum += ffc;
    }
    c[col * n + row] = sum;
  }
}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration
// purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K,
                             float alpha, float beta) {
  // Leading dimensions. Packed with no transpositions.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < K; i += WMMA_K) {

    int aRow = warpM * WMMA_M;
    int aCol = i;

    int bRow = i;
    int bCol = warpN * WMMA_N;

    // Bounds checking
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;

  if (cRow < M && cCol < N) {

    // Commented this part to make it equivalent to the HAT output for a fair
    // comparison. Note that CUBLAS must be launched with beta=0.0f

    // wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc,
    // wmma::mem_col_major); #pragma unroll for(int i=0; i <
    // c_frag.num_elements; i++) {
    //    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    // }

    // Store the output
    wmma::store_matrix_sync(c + cRow + cCol * ldc, acc_frag, ldc, wmma::mem_col_major);
  }
}

// Tensor Computation using the WMMA API with row_major
__global__ void wmma_example2(half *a, half *b, float *c, int M, int N, int K,
                              float alpha, float beta) {
  // Leading dimensions. Packed with no transpositions.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;

    int bRow = i;
    int bCol = warpN * WMMA_N;

    // Bounds checking
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;
  if (cRow < M && cCol < N) {
    wmma::store_matrix_sync(c + cCol + cRow * ldc, acc_frag, ldc, wmma::mem_row_major);
  }
}

__global__ void convertFp32ToFp16(half *out, float *in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

int main(int argc, char *argv[]) {
  float *a_fp32;
  float *b_fp32;
  half *a_fp16;
  half *b_fp16;

  float *c;
  float *c_cublas;
  float *c_wmma;
  float *c_wmma2;
  float *c_cuda;

  float *c_host_cublas;
  float *c_host_wmma;
  float *c_host_wmma2;
  float *c_host_cuda;

  curandGenerator_t gen;
  cublasHandle_t cublasHandle;

  cudaEvent_t startCUDA;
  cudaEvent_t stopCUDA;

  cudaEvent_t startWMMA;
  cudaEvent_t stopWMMA;

  cudaEvent_t startWMMA2;
  cudaEvent_t stopWMMA2;

  cudaEvent_t startcublas;
  cudaEvent_t stopcublas;

  cudaErrCheck(cudaEventCreate(&startCUDA));
  cudaErrCheck(cudaEventCreate(&stopCUDA));

  cudaErrCheck(cudaEventCreate(&startWMMA));
  cudaErrCheck(cudaEventCreate(&stopWMMA));

  cudaErrCheck(cudaEventCreate(&startWMMA2));
  cudaErrCheck(cudaEventCreate(&stopWMMA2));

  cudaErrCheck(cudaEventCreate(&startcublas));
  cudaErrCheck(cudaEventCreate(&stopcublas));

  cublasErrCheck(cublasCreate(&cublasHandle));

  // Use tensor cores
  cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

  cudaErrCheck(
      cudaMalloc((void **)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
  cudaErrCheck(
      cudaMalloc((void **)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
  cudaErrCheck(
      cudaMalloc((void **)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(
      cudaMalloc((void **)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

  cudaErrCheck(cudaMalloc((void **)&c, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(
      cudaMalloc((void **)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(
      cudaMalloc((void **)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(
      cudaMalloc((void **)&c_wmma2, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(
      cudaMalloc((void **)&c_cuda, MATRIX_M * MATRIX_N * sizeof(float)));

  c_host_cublas = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_host_wmma = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_host_wmma2 = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_host_cuda = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));

  curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
  curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(
      a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  convertFp32ToFp16<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(
      b_fp16, b_fp32, MATRIX_K * MATRIX_N);

  curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));

  curandErrCheck(curandDestroyGenerator(gen));

  cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToDevice));
  cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToDevice));
  cudaErrCheck(cudaMemcpy(c_cuda, c, MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToDevice));

  float alpha = 1.0f;
  float beta = 0.0f;

  printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M,
         MATRIX_N, MATRIX_K, alpha, beta);

  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x =
      (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
  gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  // Define grid and block dimensions
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((MATRIX_N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (MATRIX_M + threadsPerBlock.y - 1) / threadsPerBlock.y);
  // reset the c_cublas buffer
  cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToDevice));

  printf("Running with CUDA No Tensor WMMA...\n");
  for (int i = 0; i < 10; i++) {
    cudaErrCheck(cudaEventRecord(startCUDA));
    matrixMultKernel<<<numBlocks, threadsPerBlock>>>(a_fp16, b_fp16, c_cuda,
                                                     MATRIX_M);
    cudaErrCheck(cudaEventRecord(stopCUDA));
    cudaErrCheck(cudaEventSynchronize(stopCUDA));
  }

  // reset the c_cublas buffer
  cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToDevice));

  printf("Running with Tensor WMMA col_major ...\n");
  printf("\tgrid: %i, %i, %i\n", gridDim.x, gridDim.y, gridDim.z);
  printf("\tblock: %i, %i, %i\n", blockDim.x, blockDim.y, blockDim.z);
  for (int i = 0; i < 10; i++) {
    cudaErrCheck(cudaEventRecord(startWMMA));
    wmma_example<<<gridDim, blockDim>>>(a_fp16, b_fp16, c_wmma, MATRIX_M,
                                        MATRIX_N, MATRIX_K, alpha, beta);
    cudaErrCheck(cudaEventRecord(stopWMMA));
    cudaErrCheck(cudaEventSynchronize(stopWMMA));
  }

  // reset the c_cublas buffer
  cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToDevice));

  printf("Running with Tensor WMMA row_major ...\n");
  printf("\tgrid: %i, %i, %i\n", gridDim.x, gridDim.y, gridDim.z);
  printf("\tblock: %i, %i, %i\n", blockDim.x, blockDim.y, blockDim.z);
  for (int i = 0; i < 10; i++) {
    cudaErrCheck(cudaEventRecord(startWMMA2));
    wmma_example2<<<gridDim, blockDim>>>(a_fp16, b_fp16, c_wmma2, MATRIX_M,
                                         MATRIX_N, MATRIX_K, alpha, beta);
    cudaErrCheck(cudaEventRecord(stopWMMA2));
    cudaErrCheck(cudaEventSynchronize(stopWMMA2));
  }

  // Now using cuBLAS
  printf("Running with cuBLAS...\n");
  // Warm up cuBLAS run starts
  cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_M,
                              MATRIX_N, MATRIX_K, &alpha, a_fp16, CUDA_R_16F,
                              MATRIX_M, b_fp16, CUDA_R_16F, MATRIX_K, &beta,
                              c_cublas, CUDA_R_32F, MATRIX_M, CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // Warm up cuBLAS run ends

  // reset the c_cublas buffer
  cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToDevice));

  for (int i = 0; i < 10; i++) {
    cudaErrCheck(cudaEventRecord(startcublas));
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                MATRIX_M, MATRIX_N, MATRIX_K, &alpha, a_fp16,
                                CUDA_R_16F, MATRIX_M, b_fp16, CUDA_R_16F,
                                MATRIX_K, &beta, c_cublas, CUDA_R_32F, MATRIX_M,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
  }

  // Error checking
  printf("\nChecking results...\n");
  cudaErrCheck(cudaMemcpy(c_host_cuda, c_cuda,
                          MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma,
                          MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_wmma2, c_wmma2,
                          MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas,
                          MATRIX_M * MATRIX_N * sizeof(float),
                          cudaMemcpyDeviceToHost));

  // 0.01% relative tolerance. 1e-5 absolute tolerance.
  int errors = 0;
  for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
    float v1 = c_host_wmma[i];
    float v2 = c_host_cublas[i];
    float diff = fabs(v1 - v2);
    float relative_err = diff / v2;
    float eps = 1e-4;
    if ((relative_err >= eps)) {
      errors++;
      if (errors < 10)
        printf("%f %f\n", v1, v2);
    }
  }

  if (errors > 0) {
    printf("\u274c WMMA does not agree with cuBLAS! %d errors!\n", errors);
  } else {
    printf("\u2705 Results verified: cublas and WMMA agree.\n\n");
    float wmmaTime;
    float cublasTime;
    cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    printf("wmma took %fms\n", wmmaTime);
    printf("cublas took %fms\n", cublasTime);
    printf("\nFor a faster code using wmma you should check out the "
           "cudaTensorCoreGemm sample in the CUDA Toolkit.\nThis code was "
           "written as a demo only!\n\n");
  }

  errors = 0;
  for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
    float v1 = c_host_wmma2[i];
    float v2 = c_host_cuda[i];
    float diff = fabs(v1 - v2);
    float relative_err = diff / v2;
    float eps = 1e-4;
    if ((relative_err >= eps)) {
      errors++;
      if (errors < 10)
        printf("%f %f\n", v1, v2);
    }
  }

  if (errors > 0) {
    printf("\u274c WMMA2 (Row-Major) does not agree with naive impl.! %d "
           "errors!\n",
           errors);
  } else {
    printf(
        "\u2705 Results verified: WMMA2 Kernel checked with Naive Impl.\n\n");
    float cudaTime;
    cudaErrCheck(cudaEventElapsedTime(&cudaTime, startCUDA, stopCUDA));
    printf("cuda took %fms\n", cudaTime);
  }

  cudaErrCheck(cudaEventDestroy(startWMMA));
  cudaErrCheck(cudaEventDestroy(stopWMMA));

  cudaErrCheck(cudaEventDestroy(startcublas));
  cudaErrCheck(cudaEventDestroy(stopcublas));

  cudaErrCheck(cudaFree(a_fp32));
  cudaErrCheck(cudaFree(b_fp32));
  cudaErrCheck(cudaFree(a_fp16));
  cudaErrCheck(cudaFree(b_fp16));

  cudaErrCheck(cudaFree(c));
  cudaErrCheck(cudaFree(c_cublas));
  cudaErrCheck(cudaFree(c_wmma));
  cudaErrCheck(cudaFree(c_wmma2));

  free(c_host_cublas);
  free(c_host_wmma);

  cudaErrCheck(cudaDeviceReset());
  return 0;
}
