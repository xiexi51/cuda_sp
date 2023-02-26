#include "spmm_opt.h"
#include <stdio.h>
const int BLOCK_X = 16;
const int BLOCK_Y = 32;
const int WARPSIZE = 32;
const int NUM_THREADS = BLOCK_X * BLOCK_Y;

inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

__device__ inline void atomicAdd_F(float *address, float value)
{
    float old = value;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f)
        ;
}

__global__ void spmm_kernel_opt(int *coo_row, int *ptr, int *idx, float *val, float *vin, float *vout, int num_e, int feat_in)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warpid = tid / WARPSIZE;
    if (warpid >= num_e)
        return;
    int lane_id = tid & (WARPSIZE - 1);

    int begin = __ldg(ptr + warpid), end = __ldg(ptr + warpid + 1);

    float results[32] = {0};

    for (int i = begin; i < end; i++){
        float left = __ldg(val + i);
        int row = __ldg(coo_row + i), col = __ldg(idx + i);
        int right_from = col * feat_in + lane_id;
        int right_to = row * feat_in + lane_id;    
        results[lane_id] += left * vin[right_from];

    }

    



    

    // __syncwarp();

    // float result = left * vin[right_from];

//     int right_from_temp[32];
//     int right_to_temp[32];
//     float result_temp[32];

// #pragma unroll
//     for (int j = 0; j < 32; ++j)
//     {
//         right_from_temp[j] = __shfl_sync(0xFFFFFFFF, right_from, j);
//         right_to_temp[j] = __shfl_sync(0xFFFFFFFF, right_to, j);
//         result_temp[j] = left * __ldg(vin + right_from_temp[j]);
//         vout[right_to_temp[j]] += result_temp[j];
//     }
    // atomicAdd(vout + right_to, result);
    // atomicAdd_F(vout + right_to, result);
    // vout[right_to] += result;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    int BLOCK_SIZE = 512;
    grid.x = (num_e * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block.x = BLOCK_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_kernel_opt<<<grid, block>>>(coo_row, d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}