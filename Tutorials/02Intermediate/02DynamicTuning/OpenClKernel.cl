__kernel void vectorAddition(__global float* a, __global float* b, __global float* result, __global const float* scalar)
{
    int index = get_global_id(0);
    for (int i = 0; i < REPETITIONS; ++i) // repeats the same addition so that each kernel run takes longer
    {
        result[index] = a[index] + b[index] + scalar[0];
    }
}
