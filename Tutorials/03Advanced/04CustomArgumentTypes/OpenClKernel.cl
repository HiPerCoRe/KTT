typedef struct KernelData
{
    float a;
    float b;
    float result;
} KernelData;

__kernel void vectorAddition(__global KernelData* data)
{
    int index = get_global_id(0);
    data[index].result = data[index].a + data[index].b;
}
