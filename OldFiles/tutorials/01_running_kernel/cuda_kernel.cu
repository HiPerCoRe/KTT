extern "C" __global__ void vectorAddition(const float* a, const float* b, float* result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    result[index] = a[index] + b[index];
}
