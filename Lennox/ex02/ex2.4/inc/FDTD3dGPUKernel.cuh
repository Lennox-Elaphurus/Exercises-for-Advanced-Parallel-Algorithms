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

#include "FDTD3dGPU.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Note: If you change the RADIUS, you should also change the unrolling below
#define RADIUS 4
// __constant__ float stencil[RADIUS + 1];//original linear stencil
__constant__ float stencil[2 * RADIUS + 1][2 * RADIUS + 1][2 * RADIUS + 1]; // cube kernel
__constant__ float stencil2[2 * RADIUS + 1];                                // line kernel

__global__ void FiniteDifferencesKernel(float *output, const float *input,
                                        const int dimx, const int dimy,
                                        const int dimz)
{
  bool validr = true;
  bool validw = true;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int ltidx = threadIdx.x;
  const int ltidy = threadIdx.y;
  const int workx = blockDim.x;
  const int worky = blockDim.y;
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];

  const int stride_y = dimx + 2 * RADIUS;
  const int stride_z = stride_y * (dimy + 2 * RADIUS);

  int inputIndex = 0;
  int outputIndex = 0;

  // Advance inputIndex to start of inner volume
  inputIndex += RADIUS * stride_y + RADIUS;

  // Advance inputIndex to target element
  inputIndex += gtidy * stride_y + gtidx;

  float infront[RADIUS];
  float behind[RADIUS];
  float current;

  const int tx = ltidx + RADIUS;
  const int ty = ltidy + RADIUS;

  // Check in bounds
  if ((gtidx >= dimx + RADIUS) || (gtidy >= dimy + RADIUS))
    validr = false;

  if ((gtidx >= dimx) || (gtidy >= dimy))
    validw = false;

  // Preload the "infront" and "behind" data
  for (int i = RADIUS - 2; i >= 0; i--)
  {
    if (validr)
      behind[i] = input[inputIndex];

    inputIndex += stride_z;
  }

  if (validr)
    current = input[inputIndex];

  outputIndex = inputIndex;
  inputIndex += stride_z;

  for (int i = 0; i < RADIUS; i++)
  {
    if (validr)
      infront[i] = input[inputIndex];

    inputIndex += stride_z;
  }

// Step through the xy-planes
#pragma unroll 9

  for (int global_z = 0; global_z < dimz; global_z++)
  {
    // Advance the slice (move the thread-front)
    for (int i = RADIUS - 1; i > 0; i--)
      behind[i] = behind[i - 1];

    behind[0] = current;
    current = infront[0];
#pragma unroll 4

    for (int i = 0; i < RADIUS - 1; i++)
      infront[i] = infront[i + 1];

    if (validr)
      infront[RADIUS - 1] = input[inputIndex];

    inputIndex += stride_z;
    outputIndex += stride_z;
    cg::sync(cta);

    // Note that for the work items on the boundary of the problem, the
    // supplied index when reading the halo (below) may wrap to the
    // previous/next row or even the previous/next xy-plane. This is
    // acceptable since a) we disable the output write for these work
    // items and b) there is at least one xy-plane before/after the
    // current plane, so the access will be within bounds.

    // Update the data slice in the local tile
    // Halo above & below
    if (ltidy < RADIUS)
    {
      tile[ltidy][tx] = input[outputIndex - RADIUS * stride_y];
      tile[ltidy + worky + RADIUS][tx] = input[outputIndex + worky * stride_y];
    }

    // Halo left & right
    if (ltidx < RADIUS)
    {
      tile[ty][ltidx] = input[outputIndex - RADIUS];
      tile[ty][ltidx + workx + RADIUS] = input[outputIndex + workx];
    }

    tile[ty][tx] = current;
    cg::sync(cta);

    // Compute the output value
    float value = stencil2[0] * current;
#pragma unroll 4

    for (int i = 1; i <= RADIUS; i++)
    {
      value +=
          stencil2[i] * (infront[i - 1] + behind[i - 1] + tile[ty - i][tx] +
                         tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
    }

    // Store the output value
    if (validw)
      output[outputIndex] = value;
  }
}

__device__ void copyPlane(float (*dst)[k_blockDimX + 2 * RADIUS], const float (*src)[k_blockDimX + 2 * RADIUS])
{
  // dst and src are all in shared memory
  for (int idy = 0; idy < k_blockDimMaxY + 2 * RADIUS; ++idy)
  {
    for (int idx = 0; idx < k_blockDimX + 2 * RADIUS; ++idx)
    {
      dst[idy][idx] = src[idy][idx];
    }
  }

  /*
   if (validr)
   {
     // above & below
     if (ltidy < RADIUS)
     {
       dst[ltidy][tx] = src[centerIndex - RADIUS * stride_y];
       dst[ltidy + worky + RADIUS][tx] = src[centerIndex + worky * stride_y];
     }

     // left & right
     if (ltidx < RADIUS)
     {
       dst[ty][ltidx] = src[centerIndex - RADIUS];
       dst[ty][ltidx + workx + RADIUS] = src[centerIndex + workx];
     }
   }
   */
}

__global__ void FiniteDifferencesKernelCube(float *output, const float *input,
                                            const int dimx, const int dimy,
                                            const int dimz)
{
  bool validr = true;
  bool validw = true;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int ltidx = threadIdx.x;
  const int ltidy = threadIdx.y;
  const int workx = blockDim.x;
  const int worky = blockDim.y;
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  const int stride_y = dimx + 2 * RADIUS;
  const int stride_z = stride_y * (dimy + 2 * RADIUS);

  int inputIndex = 0;
  int outputIndex = 0;

  // Advance inputIndex to start of inner volume
  inputIndex += RADIUS * stride_y + RADIUS;
  const int startIndex = inputIndex; // the start of block of this thread

  // Advance inputIndex to target element
  inputIndex += gtidy * stride_y + gtidx;
  const int centerIndex = inputIndex; // the index of center element of thins thread

  __shared__ float cube_data[2 * RADIUS + 1][k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS]; // changed to full plane sglobal_ze
  // float current;

  const int tx = ltidx + RADIUS; // tile x for x-y plane
  const int ty = ltidy + RADIUS;

  // Check in bounds
  if ((gtidx >= dimx + RADIUS) || (gtidy >= dimy + RADIUS))
    validr = false; // in the whole block with halo

  if ((gtidx >= dimx) || (gtidy >= dimy))
    validw = false; // in the whole block

  // Preload the small cube data
  // "input" start with infront, we iterate from infront to behind (z)
  // Each thread do the convolution with the center of [ty][tx]

  // Loading the first cube data
  for (int global_z = 0; global_z <= 2 * RADIUS; ++global_z)
  {
    cube_data[global_z][ltidy][ltidx] = input[gtidx + gtidy * stride_y + global_z * stride_z];
  }
  cg::sync(cta);

  // Step through the xy-planes

  std::assert((validr == true));

#pragma unroll 9
  // we iterate from infront to behind
  for (int global_z = 0; global_z < dimz; global_z++)
  {
    // Advance the slice (move the thread-front)
    for (int idz = 1; idz <= 2 * RADIUS; ++idz)
    {
      data_cube[idz][ltidy][ltidx] = data_cube[idz - 1][ltidy][ltidz];
    }
    cg::sync(cta);


    // Note that for the work items on the boundary of the problem, the
    // supplied index when reading the halo (below) may wrap to the
    // previous/next row or even the previous/next xy-plane. This is
    // acceptable since a) we disable the output write for these work
    // items and b) there is at least one xy-plane before/after the
    // current plane, so the access will be within bounds.

    /* No use any more because tile now loaded from infront
    // Update the data slice in the local tile
    // Halo above & below
    if (ltidy < RADIUS)
    {
      tile[ltidy][tx] = input[outputIndex - RADIUS * stride_y];
      tile[ltidy + worky + RADIUS][tx] = input[outputIndex + worky * stride_y];
    }

    // Halo left & right
    if (ltidx < RADIUS)
    {
      tile[ty][ltidx] = input[outputIndex - RADIUS];
      tile[ty][ltidx + workx + RADIUS] = input[outputIndex + workx];
    }


    tile[ty][tx] = current;

    cg::sync(cta);
    */

    float value = 0;
#pragma unroll 4
    /*
    for (int i = 1; i <= RADIUS; i++)
    {
      value +=
          stencil[i] * (infront[i - 1] + behind[i - 1] + tile[ty - i][tx] +
                        tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
      //center of tile is tile[ty][tx]
    }
    */

    // Compute the output value of behind
    // Indexes of "behind" are from infront to behind
    // Indexes of "stencil" are from behind to infront
    // the center in x-y plane is behind[ty][tx]

    for (int idz = 0; idz < RADIUS; ++idz)
    {
      for (int idy = 0; idy <= 2 * RADIUS; ++idy)
      {
        for (int idx = 0; idx <= 2 * RADIUS; ++idx)
        {
          value += stencil[idz][idy][idx] * behind[RADIUS - 1 - idz][ltidy + idy][ltidx + idx]; // I should use the whole cube instead of using infront , behind and tile.
        }
      }
    }

    // Compute the output value of tile
    // the center is tile[ty][tx]
    for (int idy = 0; idy <= 2 * RADIUS; ++idy)
    {
      for (int idx = 0; idx <= 2 * RADIUS; ++idx)
      {
        value += stencil[RADIUS][idy][idx] * tile[ltidy + idy][ltidx + idx];
      }
    } // Finish computing the output value of tile

    // Compute the output value of infront
    // Indexes of "infront" are from behind to infront
    // Indexes of "stencil" are from behind to infront
    for (int idz = RADIUS + 1; idz <= 2 * RADIUS; ++idz)
    {
      for (int idy = 0; idy <= 2 * RADIUS; ++idy)
      {
        for (int idx = 0; idx <= 2 * RADIUS; ++idx)
        {
          value += stencil[idz][idy][idx] * infront[idz - RADIUS - 1][ltidy + idy][ltidx + idx]; // I should use the whole cube instead of using infront , behind and tile.
        }
      }
    }

    // Store the output value
    if (validw) // valid white block
      output[outputIndex] = value;
  }
}