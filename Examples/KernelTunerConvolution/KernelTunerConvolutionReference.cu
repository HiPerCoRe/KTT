__global__ void ConvolutionReference(float *output, float *input, float *filter, int image_width, int image_height, int filter_width, int filter_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i, j;
    float sum = 0.0;

    const int input_width = image_width + filter_width-1;

    if (y < image_height && x < image_width) {
        for (j = 0; j < filter_height; j++) {
            for (i = 0; i < filter_width; i++) {
                sum += input[(y + j) * input_width + (x + i)] * filter[j * filter_width + i];
            }
        }
        output[y * image_width + x] = sum;
    }
}
