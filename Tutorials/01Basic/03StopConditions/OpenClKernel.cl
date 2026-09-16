__kernel void vectorAddition(__global float* a, __global float* b, __global float* result, const float scalar)
{
    int index = get_global_id(0);
    for (int i = 0; i < REPETITIONS; ++i) // so the threads take a little longer 
    {
        result[index] = a[index] + b[index] + scalar;
    }
}
