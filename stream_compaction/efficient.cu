#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

/*****************
* Configuration *
*****************/

#define blockSize 64

// buffers
int* dev_buf;

namespace StreamCompaction {
    namespace Efficient {
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

        // kernel for upsweep 
        __global__ void upsweep_kernel(int d, int* buffer) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

          int div = pow(2, d + 1);
          if (idx % div == 0) {
            buffer[idx + (int)pow(2, d+1) - 1] += buffer[idx + (int)pow(2, d) - 1];
          }
        }

        // kernel for downsweep 
        __global__ void downsweep_kernel(int d, int* buffer) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

          int div = pow(2, d + 1);
          if (idx % div == 0) {
            // save left child
            int left = buffer[idx + (int)pow(2, d) - 1];
            
            // set left child to this node's value
            buffer[idx + (int)pow(2, d) - 1] = buffer[idx + (int)pow(2, d + 1) - 1];

            // set right child to old left value + this node's value
            buffer[idx + (int)pow(2, d + 1) - 1] += left;
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            // expand N to next power of 2
            int n2 = pow(2, ilog2ceil(n));

            // setup block structure
            dim3 fullBlocksPerGrid((n2 + blockSize - 1) / blockSize);

            // allocate memory on GPU
            cudaMalloc((void**)&dev_buf, n2 * sizeof(int));

            // copy only first n of input to device
            cudaMemcpy(dev_buf, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            // upsweep
            for (int d = 0; d <= ilog2ceil(n2) - 1; d++) {
                upsweep_kernel << <fullBlocksPerGrid, blockSize >> > (d, dev_buf);
            }

            // copy back to host to change root to 0, then back to device
            cudaMemcpy(odata, dev_buf, sizeof(int) * n2, cudaMemcpyDeviceToHost);
            odata[n2 - 1] = 0;
            cudaMemcpy(dev_buf, odata, sizeof(int) * n2, cudaMemcpyHostToDevice);

            // downsweep
            for (int d = ilog2ceil(n2) - 1; d >= 0; d--) {
              downsweep_kernel << <fullBlocksPerGrid, blockSize >> > (d, dev_buf);
            }

            // copy output to host
            cudaMemcpy(odata, dev_buf, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // free data from GPU
            cudaFree(dev_buf);

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
