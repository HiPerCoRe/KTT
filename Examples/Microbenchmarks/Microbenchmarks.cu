extern "C" __global__ void stressMem(const int* __restrict__ input, int* __restrict__ output, const int dataSize, const int repeats) //XXX size has to be power of two
{
    int tid = blockIdx.x*BLOCK + threadIdx.x;

    int id = tid&(dataSize-1);
    int res = 0;

    for (int i = 0; i < repeats; i++) {
        res += input[id];
        id = (id + gridDim.x*BLOCK + 11*256)&(dataSize-1);
        //id = (id+BLOCK*11)&(dataSize-1);
    }

    output[tid] = res;
}

