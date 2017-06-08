extern "C" __global__ void simpleKernel(float* a, float* b, float* result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    result[index] = a[index] + b[index];
}
