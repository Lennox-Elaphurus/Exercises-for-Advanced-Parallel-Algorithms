/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <assert.h>

namespace cg = cooperative_groups;

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads
    - only works for power-of-2 arrays

    This version adds multiple elements per thread sequentially.  This reduces
   the overall cost of the algorithm while keeping the work complexity O(n) and
   the step complexity O(log n). (Brent's Theorem optimization)

    See the CUDA SDK "reduction" sample for more information.
*/

template <unsigned int blockSize>
__device__ void reduceBlock(volatile float *sdata, float mySum,
                            const unsigned int tid, cg::thread_block cta)
{
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  sdata[tid] = mySum;
  cg::sync(tile32);

  const int VEC = 32;
  const int vid = tid & (VEC - 1);

  float beta = mySum;
  float temp;

  for (int i = VEC / 2; i > 0; i >>= 1)
  {
    if (vid < i)
    {
      temp = sdata[tid + i];
      beta += temp;
      sdata[tid] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (cta.thread_rank() == 0)
  {
    beta = 0;
    for (int i = 0; i < blockDim.x; i += VEC)
    {
      beta += sdata[i];
    }
    sdata[0] = beta;
  }
  cg::sync(cta);
}

/* don't need to change reduceBlock
__device__ void reduceBlockGridSync(volatile float *sdata, float mySum,
                            const unsigned int tid, cg::grid_group grid)
{
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  sdata[tid] = mySum;
  cg::sync(tile32);

  const int VEC = 32;
  const int vid = tid & (VEC - 1);

  float beta = mySum;
  float temp;

  for (int i = VEC / 2; i > 0; i >>= 1)
  {
    if (vid < i)
    {
      temp = sdata[tid + i];
      beta += temp;
      sdata[tid] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (cta.thread_rank() == 0)
  {
    beta = 0;
    for (int i = 0; i < blockDim.x; i += VEC)
    {
      beta += sdata[i];
    }
    sdata[0] = beta;
  }
  cg::sync(cta);
}
*/

template <unsigned int blockSize, bool nIsPow2>
__device__ void reduceBlocks(const float *g_idata, float *g_odata,
                             unsigned int n, cg::thread_block cta)
{
  extern __shared__ float sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  float mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  // do reduction in shared mem
  reduceBlock<blockSize>(sdata, mySum, tid, cta);

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__device__ void reduceBlocksGridSync(const float *g_idata, float *g_odata,
                                     unsigned int n, cg::thread_block cta, float *grid_array)
{
  extern __shared__ float sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  float mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  // do reduction in shared mem
  reduceBlock<blockSize>(sdata, mySum, tid, cta);

  // write result for this block to grid variable instead of global mem
  if (tid == 0)
    // g_odata[blockIdx.x] = sdata[0];
    grid_array[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceMultiPass(const float *g_idata, float *g_odata,
                                unsigned int n)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);
}

// Global variable used by reduceSinglePass to count how many blocks have
// finished
__device__ unsigned int retirementCount = 0;

cudaError_t setRetirementCount(int retCnt)
{
  return cudaMemcpyToSymbol(retirementCount, &retCnt, sizeof(unsigned int), 0,
                            cudaMemcpyHostToDevice);
}

// This reduction kernel reduces an arbitrary size array in a single kernel
// invocation It does so by keeping track of how many blocks have finished.
// After each thread block completes the reduction of its own block of data, it
// "takes a ticket" by atomically incrementing a global counter.  If the ticket
// value is equal to the number of thread blocks, then the block holding the
// ticket knows that it is the last block to finish.  This last block is
// responsible for summing the results of all the other blocks.
//
// In order for this to work, we must be sure that before a block takes a
// ticket, all of its memory transactions have completed.  This is what
// __threadfence() does -- it blocks until the results of all outstanding memory
// transactions within the calling thread are visible to all other threads.
//
// For more details on the reduction algorithm (notably the multi-pass
// approach), see the "reduction" sample in the CUDA SDK.
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const float *g_idata, float *g_odata,
                                 unsigned int n)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  //
  // PHASE 1: Process all inputs assigned to this block
  //

  reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);

  //
  // PHASE 2: Last block finished will process all partial sums
  //

  if (gridDim.x > 1)
  {
    const unsigned int tid = threadIdx.x;
    __shared__ bool amLast;
    extern float __shared__ smem[];

    // wait until all outstanding memory instructions in this thread are
    // finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0)
    {
      unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
      // If the ticket ID is equal to the number of blocks, we are the last
      // block!
      amLast = (ticket == gridDim.x - 1);
    }

    cg::sync(cta);

    // The last block sums the results of all other blocks
    if (amLast)
    {
      int i = tid;
      float mySum = 0;

      while (i < gridDim.x)
      {
        mySum += g_odata[i];
        i += blockSize;
      }

      reduceBlock<blockSize>(smem, mySum, tid, cta);

      if (tid == 0)
      {
        g_odata[0] = smem[0];

        // reset retirement count so that next run succeeds
        retirementCount = 0;
      }
    }
  }
}

// struct Arg
// {
//   const float *g_idata;
//   float *g_odata;
//   unsigned int n;
//   cg::grid_group grid;
// }

// my work, use grid.sync();
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePassGridSync(const float *g_idata, float *g_odata,
                                         unsigned int n)
{
  assert(blockSize < 0 && "In reduceSinglePassGridSync");
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();

  //
  // PHASE 1: Process all inputs assigned to this block
  //

  float grid_array[blockSize];
  // reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);
  reduceBlocksGridSync<blockSize, nIsPow2>(g_idata, g_odata, n, cta, (float *)&grid_array);
  cg::sync(grid);

  //
  // PHASE 2: Last block finished will process all partial sums
  //

  if (gridDim.x > 1)
  {
    const unsigned int tid = threadIdx.x;
    __shared__ bool amLast;
    extern float __shared__ smem[];

    // wait until all outstanding memory instructions in this thread are
    // finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0)
    {
      unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
      // If the ticket ID is equal to the number of blocks, we are the last
      // block!
      amLast = (ticket == gridDim.x - 1);
    }

    cg::sync(grid); // or  cg::sync(cta);?

    // The last block sums the results of all other blocks
    if (amLast)
    {
      int i = tid;
      float mySum = 0;

      while (i < gridDim.x)
      {
        mySum += g_odata[i];
        i += blockSize;
      }

      reduceBlock<blockSize>(smem, mySum, tid, cta);

      if (tid == 0)
      {
        g_odata[0] = smem[0];

        // reset retirement count so that next run succeeds
        retirementCount = 0;
      }
    }
  }
}

bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
extern "C" void reduce(int size, int threads, int blocks, float *d_idata,
                       float *d_odata)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size))
  {
    switch (threads)
    {
    case 512:
      reduceMultiPass<512, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 256:
      reduceMultiPass<256, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 128:
      reduceMultiPass<128, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 64:
      reduceMultiPass<64, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 32:
      reduceMultiPass<32, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 16:
      reduceMultiPass<16, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 8:
      reduceMultiPass<8, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 4:
      reduceMultiPass<4, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 2:
      reduceMultiPass<2, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 1:
      reduceMultiPass<1, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    }
  }
  else
  {
    switch (threads)
    {
    case 512:
      reduceMultiPass<512, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 256:
      reduceMultiPass<256, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 128:
      reduceMultiPass<128, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 64:
      reduceMultiPass<64, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 32:
      reduceMultiPass<32, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 16:
      reduceMultiPass<16, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 8:
      reduceMultiPass<8, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 4:
      reduceMultiPass<4, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 2:
      reduceMultiPass<2, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 1:
      reduceMultiPass<1, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    }
  }
}

extern "C" void reduceSinglePass(int size, int threads, int blocks,
                                 float *d_idata, float *d_odata)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  int smemSize = threads * sizeof(float);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size))
  {
    switch (threads)
    {
    case 512:
      reduceSinglePass<512, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 256:
      reduceSinglePass<256, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 128:
      reduceSinglePass<128, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 64:
      reduceSinglePass<64, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 32:
      reduceSinglePass<32, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 16:
      reduceSinglePass<16, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 8:
      reduceSinglePass<8, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 4:
      reduceSinglePass<4, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 2:
      reduceSinglePass<2, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 1:
      reduceSinglePass<1, true>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    }
  }
  else
  {
    switch (threads)
    {
    case 512:
      reduceSinglePass<512, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 256:
      reduceSinglePass<256, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 128:
      reduceSinglePass<128, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 64:
      reduceSinglePass<64, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 32:
      reduceSinglePass<32, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 16:
      reduceSinglePass<16, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 8:
      reduceSinglePass<8, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 4:
      reduceSinglePass<4, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 2:
      reduceSinglePass<2, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 1:
      reduceSinglePass<1, false>
          <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    }
  }
}

extern "C" void reduceSinglePassGridSync(int size, int threads, int blocks,
                                         float *d_idata, float *d_odata)
{
  std::cout<<"In reduceSinglePassGridSync() extern 'C'"<<std::endl;
  // dim3 dimBlock(threads, 1, 1);
  // dim3 dimGrid(blocks, 1, 1);
  int smemSize = 0;
  
  // source: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#grid-synchronization-cg

  // It is good practice to first ensure the device supports cooperative launches by querying the device attribute cudaDevAttrCooperativeLaunch:
  int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

  /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
  int numBlocksPerSm = 0;
  // Number of threads my_kernel will be launched with
  int numThreads = 128;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<threads,isPow2(size)>, numThreads, 0);
  // error: argument types are: (int *, <unknown-type>, int, int), the compiler can't solve the function template

  // launch
  void *args[] = {d_idata, d_odata, &size};
  dim3 dimBlock(numThreads, 1, 1);
  // dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);
  dim3 dimGrid(1, 1, 1);

  // cudaLaunchCooperativeKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )
  // can't use template function as func or will get error
  // error: cannot determine which instance of overloaded function "reduceSinglePassGridSync" is intended

  std::cout<<"In reduceSinglePassGridSync() extern 'C' before switch"<<std::endl;
  // choose which of the optimized versions of reduction to launch
  if (isPow2(size))
  {
    switch (threads)
    {
    case 512:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<512, true>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<512, true>,
                                  dimGrid, dimBlock, args, smemSize);
      break;

    case 256:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<256, true>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);
      
      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<256, true>, dimGrid, dimBlock, args, smemSize);
      break;

    case 128:
      
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<128, true>, numThreads, 0);
      std::cout<<"after cudaOccupancyMaxActiveBlocksPerMultiprocessor"<<std::endl;
      dimGrid.x =deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);
      std::cout<<"before cudaLaunchCooperativeKernel"<<std::endl;

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<128, true>, dimGrid, dimBlock, args, smemSize);
      
      std::cout<<"after cudaLaunchCooperativeKernel"<<std::endl;
      break;

    case 64:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<64, true>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<64, true>, dimGrid, dimBlock, args, smemSize);
      break;

    case 32:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<32, true>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<32, true>, dimGrid, dimBlock, args, smemSize);
      break;

    case 16:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<16, true>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<16, true>, dimGrid, dimBlock, args, smemSize);
      break;

    case 8:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<8, true>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<8, true>, dimGrid, dimBlock, args, smemSize);
      break;

    case 4:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<4, true>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<4, true>, dimGrid, dimBlock, args, smemSize);
      break;

    case 2:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<2, true>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<2, true>, dimGrid, dimBlock, args, smemSize);
      break;

    case 1:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<1, true>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<1, true>, dimGrid, dimBlock, args, smemSize);
      break;
    }
  }
  else
  {
    switch (threads)
    {
    case 512:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<512, false>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<512, false>, dimGrid, dimBlock, args, smemSize);
      break;

    case 256:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<256, false>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<256, false>, dimGrid, dimBlock, args, smemSize);
      break;

    case 128:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<128, false>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<128, false>, dimGrid, dimBlock, args, smemSize);
      break;

    case 64:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<64, false>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<64, false>, dimGrid, dimBlock, args, smemSize);
      break;

    case 32:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<32, false>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<32, false>, dimGrid, dimBlock, args, smemSize);
      break;

    case 16:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<16, false>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<16, false>, dimGrid, dimBlock, args, smemSize);
      break;

    case 8:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<8, false>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<8, false>, dimGrid, dimBlock, args, smemSize);
      break;

    case 4:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<4, false>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<4, false>, dimGrid, dimBlock, args, smemSize);
      break;

    case 2:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<2, false>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<2, false>, dimGrid, dimBlock, args, smemSize);
      break;

    case 1:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassGridSync<1, false>, numThreads, 0);
      dimGrid.x = deviceProp.multiProcessorCount * numBlocksPerSm;
      smemSize = (dimGrid.x <= 32) ? 2 * dimGrid.x * sizeof(float) : dimGrid.x * sizeof(float);

      cudaLaunchCooperativeKernel((void *)reduceSinglePassGridSync<1, false>, dimGrid, dimBlock, args, smemSize);
      break;
    }
  }

  std::cout<<"In reduceSinglePassGridSync() extern 'C' after calling cudaLaunchCooperativeKernel"<<std::endl;
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch cudaLaunchCooperativeKernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#endif // #ifndef _REDUCE_KERNEL_H_
