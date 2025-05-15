extern "C" __global__ void stressMem(const int* __restrict__ input, int* __restrict__ output, const int dataSize, const int repeats) //XXX size has to be power of two
{
    int tid = blockIdx.x*BLOCK + threadIdx.x;

    int id = tid&(dataSize-1);

    for (int i = 0; i < repeats; i++) {
        if ((blockIdx.x == 100) && (threadIdx.x == 0)) printf("%i ", id);
        output[id] = input[id];
        id = (id + gridDim.x*BLOCK)&(dataSize-1);
    }
}

/*extern "C" __global__ void stressMem(const int* __restrict__ input, int* __restrict__ output, const int dataSize, const int repeats) //XXX size has to be power of two
{
    int tid = blockIdx.x*BLOCK + threadIdx.x;

    int id = (int)(virtId&(dataSize-1));

    int tmp = 0;

    while (virtId < copySize) {
        tmp += input[id];
        virtId += gridDim.x*BLOCK;
        id = (int)(virtId&(dataSize-1));
    }

    output[id%dataSize] = tmp;
}*/
