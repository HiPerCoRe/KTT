__kernel void blurHorizontally(__global const float* input, __global float* output,
    int imageWidth, int imageHeight, int blurRadius)
{
    const int index = get_global_id(0);
    const int x = index % imageWidth;
    const int y = index / imageWidth;

    float sum = 0.0f;
    int blurSize = 0;

    for (int offset = -blurRadius; offset <= blurRadius; ++offset) {
        const int neighborX = x + offset;

        if (neighborX >= 0 && neighborX < imageWidth) {
            sum += input[(y * imageWidth) + neighborX];
            ++blurSize;
        }
    }

#if TRANSPOSE_INTERMEDIATE
    output[(x * imageHeight) + y] = sum / blurSize;
#else
    output[(y * imageWidth) + x] = sum / blurSize;
#endif
}

__kernel void blurVertically(__global const float* input, __global float* output,
    int imageWidth, int imageHeight, int blurRadius)
{
    const int index = get_global_id(0);
    const int x = index % imageWidth;
    const int y = index / imageWidth;

    float sum = 0.0f;
    int blurSize = 0;

    for (int offset = -blurRadius; offset <= blurRadius; ++offset) {
        const int neighborY = y + offset;

        if (neighborY >= 0 && neighborY < imageHeight) {
#if TRANSPOSE_INTERMEDIATE
            sum += input[(x * imageHeight) + neighborY];
#else
            sum += input[(neighborY * imageWidth) + x];
#endif
            ++blurSize;
        }
    }

    output[(y * imageWidth) + x] = sum / blurSize;
}
