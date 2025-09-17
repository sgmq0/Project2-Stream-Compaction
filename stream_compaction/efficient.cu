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
         * Helper function for doing scan
         */
        void up_down_sweep(dim3 fullBlocksPerGrid, int n2, int* dev_array) {
            for (int d = 0; d <= ilog2ceil(n2) - 1; d++) {
              upsweep_kernel << <fullBlocksPerGrid, blockSize >> > (n2, d, dev_array);
            }

            // set root to 0
            cudaMemset(dev_array + n2 - 1, 0, sizeof(int));

            // downsweep
            for (int d = ilog2ceil(n2) - 1; d >= 0; d--) {
              downsweep_kernel << <fullBlocksPerGrid, blockSize >> > (n2, d, dev_array);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // expand N to next power of 2
            int n2 = pow(2, ilog2ceil(n));

            // setup block structure
            dim3 fullBlocksPerGrid((n2 + blockSize - 1) / blockSize);

            // allocate memory on GPU
            cudaMalloc((void**)&dev_scan, n2 * sizeof(int));

            // copy only first n of input to device
            cudaMemcpy(dev_scan, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // upsweep, downsweep
            up_down_sweep(fullBlocksPerGrid, n2, dev_scan);

            timer().endGpuTimer();

            // copy output to host
            cudaMemcpy(odata, dev_scan, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // free data from GPU
            cudaFree(dev_scan);
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

            // expand N to next power of 2
            int n2 = pow(2, ilog2ceil(n));
            
            // setup block structure
            dim3 fullBlocksPerGrid((n2 + blockSize - 1) / blockSize);

            // initialize host arrays
            int* host_bools = new int[n2];
            int* host_indices = new int[n2];

            // allocate memory on GPU
            cudaMalloc((void**)&dev_idata, n2 * sizeof(int));
            cudaMalloc((void**)&dev_bools, n2 * sizeof(int));
            cudaMalloc((void**)&dev_indices, n2 * sizeof(int));
            cudaMalloc((void**)&dev_scatter, n2 * sizeof(int));

            // copy input to device
            cudaMemcpy(dev_idata, idata, sizeof(int) * n2, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // compute temp array containing booleans
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n2, dev_bools, dev_idata);

            // copy dev_bools to dev_indices
            cudaMemcpy(dev_indices, dev_bools, sizeof(int) * n2, cudaMemcpyDeviceToDevice);

            // run exclusive scan on bool array
            up_down_sweep(fullBlocksPerGrid, n2, dev_indices);

            // run scatter to compute final array
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n2, dev_scatter, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            // copy output to host
            cudaMemcpy(odata, dev_scatter, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_indices, dev_indices, sizeof(int) * n2, cudaMemcpyDeviceToHost);

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
        }
    }
}
