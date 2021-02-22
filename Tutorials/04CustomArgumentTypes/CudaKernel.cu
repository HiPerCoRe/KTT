typedef struct KernelData
{
    float a;
    float b;
    float result;
} KernelData;

extern "C" __global__ void vectorAddition(KernelData* data)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    data[index].result = data[index].a + data[index].b;
}
