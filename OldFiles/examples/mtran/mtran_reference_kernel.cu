extern "C" __global__ void mtranReference(
    float *output,
    float *input, 
    const int width,
    const int height)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	output[y*width + x] = input[x*height + y];
}
