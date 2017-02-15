__kernel void referenceKernel(float number, __global float* a, __global float* b, __global float* result)
{
    int index = get_global_id(0);

    result[index] = a[index] + b[index] + number;
}
