#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 64

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// buffers
int* dev_in;
int* dev_out;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Helper functions from testing_helpers.hpp
        */
        void printArray(int n, const int* a, bool abridged = false) {
          printf("    [ ");
          for (int i = 0; i < n; i++) {
            if (abridged && i + 2 == 15 && n > 16) {
              i = n - 2;
              printf("... ");
            }
            printf("%3d ", a[i]);
          }
          printf("]\n");
        }

        void zeroArray(int n, int* a) {
          for (int i = 0; i < n; i++) {
            a[i] = 0;
          }
        }

        // shift right kernel
        __global__ void shift_right_kernel(int n, int* output, int* input) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (idx >= n) {
            return;
          }

          if (idx != 0) {
            output[idx] = input[idx - 1];
          }
        }

        // TODO: __global__ kernel
        __global__ void scan_kernel(int n, int layer, int* output, int* input) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) {
              return;
            }

            if (idx >= pow(2, layer-1)) {
                output[idx] = input[idx - (int)pow(2, layer - 1)] + input[idx];
            }
            else {
                output[idx] = input[idx];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            // setup block structure
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // allocate memory on GPU
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMalloc((void**)&dev_out, n * sizeof(int));

            // copy input to device
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            // run kernel ilog2ceil(n) times
            bool swap = true;
            for (int d = 1; d <= ilog2ceil(n); d++) {
              if (swap) {
                scan_kernel << <fullBlocksPerGrid, blockSize >> > (n, d, dev_out, dev_in);
              }
              else {
                scan_kernel << <fullBlocksPerGrid, blockSize >> > (n, d, dev_in, dev_out);
              }
              swap = !swap;
            }

            // shift right to make exclusive and copy output to host
            if (swap) {
              shift_right_kernel << <fullBlocksPerGrid, blockSize >> > (n, dev_out, dev_in);
              cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            }
            else {
              shift_right_kernel << <fullBlocksPerGrid, blockSize >> > (n, dev_in, dev_out);
              cudaMemcpy(odata, dev_in, sizeof(int) * n, cudaMemcpyDeviceToHost);
            }

            odata[0] = 0;

            // free data from GPU
            cudaFree(dev_in);
            cudaFree(dev_out);

            // TODO
            timer().endGpuTimer();
        }
    }
}
