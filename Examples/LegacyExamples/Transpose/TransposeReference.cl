__kernel void mtranReference(
    __global float *output,
    __global float *input, 
    const int width,
    const int height)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	output[y*width + x] = input[x*height + y];
}
