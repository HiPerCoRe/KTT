extern "C" __global__ void reduce(const float* input, float* output, const int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int gridStride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread accumulates multiple elements via grid-stride loop
    float sum = 0.0f;

    for (int i = idx; i < n; i += gridStride)
    {
        sum += input[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
#if BLOCK_SIZE >= 512
    if (tid < 256) sdata[tid] += sdata[tid + 256];
    __syncthreads();
#endif
#if BLOCK_SIZE >= 256
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
#endif
#if BLOCK_SIZE >= 128
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
#endif
#if BLOCK_SIZE >= 64
    if (tid < 32) sdata[tid] += sdata[tid + 32];
    __syncthreads();
#endif
    if (tid < 16) sdata[tid] += sdata[tid + 16];
    __syncthreads();
    if (tid < 8) sdata[tid] += sdata[tid + 8];
    __syncthreads();
    if (tid < 4) sdata[tid] += sdata[tid + 4];
    __syncthreads();
    if (tid < 2) sdata[tid] += sdata[tid + 2];
    __syncthreads();

    if (tid == 0)
    {
        atomicAdd(output, sdata[0] + sdata[1]);
    }
}
