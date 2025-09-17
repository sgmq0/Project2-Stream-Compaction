#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = idata[i - 1] + odata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int idx = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[idx] = idata[i];
                    idx++;
                }
            }
            timer().endCpuTimer();
            return n-idx;
        }

        int scatter(int n, int* odata, const int* compact, const int* scan, const int* idata) {
          int count = 0;
          for (int i = 0; i < n; i++) {
            if (compact[i] == 1) {
              odata[scan[i]] = idata[i];
              count++;
            }
          }
          return n - count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* compact = new int[n];
            int compact_count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    compact[i] = 1;
                    compact_count++;
                }
                else {
                    compact[i] = 0;
                } 
            }

            int* scan_result = new int[n];
            scan_result[0] = 0;
            for (int i = 1; i < n; i++) {
              scan_result[i] = compact[i - 1] + scan_result[i - 1];
            }

            int items_left = scatter(n, odata, compact, scan_result, idata);

            timer().endCpuTimer();
            return items_left;
        }
    }
}
