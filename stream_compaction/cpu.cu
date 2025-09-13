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
         * Helper functions from testing_helpes.hpp
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
            //timer().startCpuTimer();
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
            zeroArray(n, scan_result);
            scan(n, scan_result, compact);

            int items_left = scatter(n, odata, compact, scan_result, idata);

            //timer().endCpuTimer();
            return items_left;
        }
    }
}
