#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

/*****************
* Configuration *
*****************/

#define blockSize 256

// buffers
int* dev_scan;

int* dev_idata;
int* dev_bools;
int* dev_indices;
int* dev_scatter;

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
        __global__ void upsweep_kernel(int n, int d, int* buffer) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (idx >= n) {
            return;
          }

          int div = pow(2, d + 1);
          if (idx % div == 0) {
            buffer[idx + (int)pow(2, d+1) - 1] += buffer[idx + (int)pow(2, d) - 1];
          }
        }

        // kernel for downsweep 
        __global__ void downsweep_kernel(int n, int d, int* buffer) {
          int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (idx >= n) {
            return;
          }

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
            cudaMalloc((void**)&dev_scan, n2 * sizeof(int));

            // copy only first n of input to device
            cudaMemcpy(dev_scan, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            // upsweep
            for (int d = 0; d <= ilog2ceil(n2) - 1; d++) {
                upsweep_kernel << <fullBlocksPerGrid, blockSize >> > (n2, d, dev_scan);
            }

            // copy back to host to change root to 0, then back to device
            cudaMemcpy(odata, dev_scan, sizeof(int) * n2, cudaMemcpyDeviceToHost);
            odata[n2 - 1] = 0;
            cudaMemcpy(dev_scan, odata, sizeof(int) * n2, cudaMemcpyHostToDevice);

            // downsweep
            for (int d = ilog2ceil(n2) - 1; d >= 0; d--) {
              downsweep_kernel << <fullBlocksPerGrid, blockSize >> > (n2, d, dev_scan);
            }

            // copy output to host
            cudaMemcpy(odata, dev_scan, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // free data from GPU
            cudaFree(dev_scan);

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
            //timer().startGpuTimer();
            
            // setup block structure
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // initialize host arrays
            int* host_bools = new int[n];
            int* host_indices = new int[n];

            // allocate memory on GPU
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            cudaMalloc((void**)&dev_scatter, n * sizeof(int));

            // copy input to device
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            // compute temp array containing booleans
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);

            // move device bools to host
            cudaMemcpy(host_bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // run exclusive scan on bool array
            scan(n, host_indices, host_bools);

            // move host indices to device
            cudaMemcpy(dev_indices, host_indices, sizeof(int) * n, cudaMemcpyHostToDevice);

            // run scatter to compute final array
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_scatter, dev_idata, dev_bools, dev_indices);

            // copy output to host
            cudaMemcpy(odata, dev_scatter, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // free data from GPU
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_scatter);

            // compute remaining elements
            if (idata[n - 1] == 0)
                return n - host_indices[n - 1];
            else
                return n - host_indices[n - 1] - 1;

            //timer().endGpuTimer();
        }
    }
}
