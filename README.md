CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Raymond Feng
  * [LinkedIn](https://www.linkedin.com/in/raymond-ma-feng/), [personal website](https://www.rfeng.dev/)
* Tested on: Windows 11, i9-9900KF @ 3.60GHz 32GB, NVIDIA GeForce RTX 2070 SUPER (Personal Computer)

This project implements the all-prefix sums scan algorithm on the CPU, a naive version on the GPU, a work-efficient version on the GPU, and an implementation using the thrust library. It also implements work-efficient parallel stream compaction.

## Project Description

### CPU Scan

### Naive GPU Scan

### Work-Efficient GPU Scan and Stream Compaction

### Thrust Scan

## Performance Analysis

### Block Size

### Naive GPU Scan

### Work-Efficient Scan

### Work-Efficient Stream Compaction

### Thrust

## Output
A sample of my output. N=2^8.

```
****************
** SCAN TESTS **
****************
    [   8  28  26  35   8  29   6  13   3  36  31  18  45 ...  33   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0   8  36  62  97 105 134 140 153 156 192 223 241 ... 6044 6077 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0   8  36  62  97 105 134 140 153 156 192 223 241 ... 5968 5977 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.385344ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.07872ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.464288ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.151552ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.862272ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.03824ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   2   0   3   0   3   2   3   1   0   3   0   3 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   2   2   3   3   2   3   1   3   3   3   3   2   1 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   2   2   3   3   2   3   1   3   3   3   3   2   1 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0018ms    (std::chrono Measured)
    [   2   2   3   3   2   3   1   3   3   3   3   2   1 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.464704ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.15792ms    (CUDA Measured)
    passed```