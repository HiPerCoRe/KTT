#define HFS 1               // Half filter size
#define FS (HFS + HFS + 1)  // Filter size

// =================================================================================================

// Reference implementation of the 2D convolution example
__kernel void conv_reference(const int width, const int height, const __global float* src,
                             __constant float* coeff, __global float* dest) {
  // Thread identifiers
  const int tid_x = get_global_id(0);  // From 0 to width-1
  const int tid_y = get_global_id(1);  // From 0 to height-1
  const int tid_z = get_global_id(2);  // From 0 to depth-1

  // Initializes the accumulation register
  float acc = 0.0f;

  // Loops over the neighbourhood
  for (int fx = -HFS; fx <= HFS; ++fx) {
    const int index_x = tid_x + HFS + fx;
    for (int fy = -HFS; fy <= HFS; ++fy) {
      const int index_y = tid_y + HFS + fy;
      for (int fz = -HFS; fz <= HFS; ++fz) {
        const int index_z = tid_z + HFS + fz;

        // Performs the accumulation
        float coefficient = coeff[(fz + HFS) * FS * FS + (fy + HFS) * FS + (fx + HFS)];
        acc += coefficient * src[index_z * (height + 2 * HFS) * (width + 2 * HFS) +
                                 index_y * (width + 2 * HFS) + index_x];
      }
    }
  }

  // Stores the result
  dest[tid_z * height * width + tid_y * width + tid_x] = acc;
}

// =================================================================================================
