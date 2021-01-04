#define HFS 1               // Half filter size
#define FS (HFS + HFS + 1)  // Filter size

// Vector data-types
#if VECTOR == 1
typedef float floatvec;
#elif VECTOR == 2
typedef float2 floatvec;
#elif VECTOR == 4
typedef float4 floatvec;
#elif VECTOR == 8
typedef float8 floatvec;
#endif

// Coefficients array type
#if CONSTANT_COEFF == 1
#  define COEFFTYPE __constant float*
#else
#  define COEFFTYPE const __global float*
#endif

// =================================================================================================

// Initialize the accumulation registers
inline void InitAccRegisters(float acc[WPTZ][WPTY][WPTX]) {
#pragma unroll
  for (int wz = 0; wz < WPTZ; ++wz) {
#pragma unroll
    for (int wy = 0; wy < WPTY; ++wy) {
#pragma unroll
      for (int wx = 0; wx < WPTX; ++wx) {
        acc[wz][wy][wx] = 0.0f;
      }
    }
  }
}

// =================================================================================================

// Loads data into local memory
#if LOCAL == 2
inline void LoadLocalFull(__local float* lmem, const int lmem_width, const int lmem_height,
                          const __global floatvec* src, const int src_width, const int src_height,
                          const int gid_x, const int gid_y, const int gid_z, const int lid_x,
                          const int lid_y, const int lid_z) {
// Loop over the amount of work per thread
#  pragma unroll
  for (int wx = 0; wx < WPTX / VECTOR; wx++) {
    const int lx = lid_x * WPTX / VECTOR + wx;
#  if WPTX > 0
    if (lx < TBX * WPTX / VECTOR + (2 * HFS) / VECTOR)
#  endif
    {
      const int gx = gid_x * WPTX / VECTOR + wx;
#  pragma unroll
      for (int wy = 0; wy < WPTY; wy++) {
        const int ly = lid_y * WPTY + wy;
#  if WPTY > 0
        if (ly < TBY * WPTY + 2 * HFS)
#  endif
        {
          const int gy = gid_y * WPTY + wy;
#  pragma unroll
          for (int wz = 0; wz < WPTZ; wz++) {
            const int lz = lid_z * WPTZ + wz;
#  if WPTZ > 0
            if (lz < TBZ * WPTZ + 2 * HFS)
#  endif
            {
              const int gz = gid_z * WPTZ + wz;

              // Load the data into local memory (WPTX elements per thread)
              floatvec temp =
                  src[gz * src_height * src_width / VECTOR + gy * src_width / VECTOR + gx];
#  if VECTOR == 1
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR)] = temp;
#  elif VECTOR == 2
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR)] = temp.x;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 1)] = temp.y;
#  elif VECTOR == 4
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR)] = temp.x;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 1)] = temp.y;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 2)] = temp.z;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 3)] = temp.w;
#  elif VECTOR == 8
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR)] = temp.s0;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 1)] = temp.s1;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 2)] = temp.s2;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 3)] = temp.s3;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 4)] = temp.s4;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 5)] = temp.s5;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 6)] = temp.s6;
              lmem[lz * lmem_height * lmem_width + (ly)*lmem_width + (lx * VECTOR + 7)] = temp.s7;
#  endif
            }
          }
        }
      }
    }
  }
}
#endif

// Loads data (plus the halos) into local memory
#if LOCAL == 1
inline void LoadLocalPlusHalo(__local float* lmem, const int lmem_width, const int lmem_height,
                              const __global float* src, const int src_width, const int src_height,
                              const int gid_x, const int gid_y, const int gid_z, const int lid_x,
                              const int lid_y, const int lid_z) {
// Loop over the amount of work per thread
#  pragma unroll
  for (int wx = 0; wx < WPTX; wx++) {
    const int lx = lid_x * WPTX + wx;
    const int gx = gid_x * WPTX + wx;
#  pragma unroll
    for (int wy = 0; wy < WPTY; wy++) {
      const int ly = lid_y * WPTY + wy;
      const int gy = gid_y * WPTY + wy;
#  pragma unroll
      for (int wz = 0; wz < WPTZ; wz++) {
        const int lz = lid_z * WPTZ + wz;
        const int gz = gid_z * WPTZ + wz;

        // Computes the conditionals
        const bool low_x = lx < HFS;
        const bool high_x = lx >= TBX * WPTX - HFS;
        const bool low_y = ly < HFS;
        const bool high_y = ly >= TBY * WPTY - HFS;
        const bool low_z = lz < HFS;
        const bool high_z = lz >= TBZ * WPTZ - HFS;

        // center
        lmem[(lz + HFS) * lmem_height * lmem_width + (ly + HFS) * lmem_width + lx + HFS] =
            src[(gz + HFS) * src_height * src_width + (gy + HFS) * src_width + gx + HFS];
        // some threads should load the half-filter outside the thread block
        if (low_z) {
          const int lmem_z = lz * lmem_height * lmem_width;
          const int src_z = gz * src_height * src_width;
          lmem[lmem_z + (ly + HFS) * lmem_width + lx + HFS] =
              src[src_z + (gy + HFS) * src_width + gx + HFS];
          if (low_y) {
            const int lmem_y = ly * lmem_width;
            const int src_y = gy * src_width;
            lmem[lmem_z + lmem_y + lx + HFS] = src[src_z + src_y + gx + HFS];
            if (low_x) lmem[lmem_z + lmem_y + lx] = src[src_z + src_y + gx];
            if (high_x) lmem[lmem_z + lmem_y + lx + 2 * HFS] = src[src_z + src_y + gx + 2 * HFS];
          }
          if (high_y) {
            const int lmem_y = (ly + 2 * HFS) * lmem_width;
            const int src_y = (gy + 2 * HFS) * src_width;
            lmem[lmem_z + lmem_y + lx + HFS] = src[src_z + src_y + gx + HFS];
            if (low_x) lmem[lmem_z + lmem_y + lx] = src[src_z + src_y + gx];
            if (high_x) lmem[lmem_z + lmem_y + lx + 2 * HFS] = src[src_z + src_y + gx + 2 * HFS];
          }
          const int lmem_y = (ly + HFS) * lmem_width;
          const int src_y = (gy + HFS) * src_width;
          if (low_x) lmem[lmem_z + lmem_y + lx] = src[src_z + src_y + gx];
          if (high_x) lmem[lmem_z + lmem_y + lx + 2 * HFS] = src[src_z + src_y + gx + 2 * HFS];
        }
        if (high_z) {
          const int lmem_z = (lz + 2 * HFS) * lmem_height * lmem_width;
          const int src_z = (gz + 2 * HFS) * src_height * src_width;
          lmem[lmem_z + (ly + HFS) * lmem_width + lx + HFS] =
              src[src_z + (gy + HFS) * src_width + gx + HFS];
          if (low_y) {
            const int lmem_y = ly * lmem_width;
            const int src_y = gy * src_width;
            lmem[lmem_z + lmem_y + lx + HFS] = src[src_z + src_y + gx + HFS];
            if (low_x) lmem[lmem_z + lmem_y + lx] = src[src_z + src_y + gx];
            if (high_x) lmem[lmem_z + lmem_y + lx + 2 * HFS] = src[src_z + src_y + gx + 2 * HFS];
          }
          if (high_y) {
            const int lmem_y = (ly + 2 * HFS) * lmem_width;
            const int src_y = (gy + 2 * HFS) * src_width;
            lmem[lmem_z + lmem_y + lx + HFS] = src[src_z + src_y + gx + HFS];
            if (low_x) lmem[lmem_z + lmem_y + lx] = src[src_z + src_y + gx];
            if (high_x) lmem[lmem_z + lmem_y + lx + 2 * HFS] = src[src_z + src_y + gx + 2 * HFS];
          }
          const int lmem_y = (ly + HFS) * lmem_width;
          const int src_y = (gy + HFS) * src_width;
          if (low_x) lmem[lmem_z + lmem_y + lx] = src[src_z + src_y + gx];
          if (high_x) lmem[lmem_z + lmem_y + lx + 2 * HFS] = src[src_z + src_y + gx + 2 * HFS];
        }
        const int lmem_z = (lz + HFS) * lmem_height * lmem_width;
        const int src_z = (gz + HFS) * src_height * src_width;
        if (low_y) {
          const int lmem_y = ly * lmem_width;
          const int src_y = gy * src_width;
          lmem[lmem_z + lmem_y + lx + HFS] = src[src_z + src_y + gx + HFS];
          if (low_x) lmem[lmem_z + lmem_y + lx] = src[src_z + src_y + gx];
          if (high_x) lmem[lmem_z + lmem_y + lx + 2 * HFS] = src[src_z + src_y + gx + 2 * HFS];
        }
        if (high_y) {
          const int lmem_y = (ly + 2 * HFS) * lmem_width;
          const int src_y = (gy + 2 * HFS) * src_width;
          lmem[lmem_z + lmem_y + lx + HFS] = src[src_z + src_y + gx + HFS];
          if (low_x) lmem[lmem_z + lmem_y + lx] = src[src_z + src_y + gx];
          if (high_x) lmem[lmem_z + lmem_y + lx + 2 * HFS] = src[src_z + src_y + gx + 2 * HFS];
        }
        const int lmem_y = (ly + HFS) * lmem_width;
        const int src_y = (gy + HFS) * src_width;
        if (low_x) lmem[lmem_z + lmem_y + lx] = src[src_z + src_y + gx];
        if (high_x) lmem[lmem_z + lmem_y + lx + 2 * HFS] = src[src_z + src_y + gx + 2 * HFS];
      }
    }
  }
}
#endif

// =================================================================================================

// Accumulates in the global (LOCAL==0) or local (LOCAL==1|2) memory
inline void Accumulate(
#if LOCAL == 0  // src is global memory
    const __global float* src,
#else  // src is local memory
    __local float* src,
#endif
    const int src_width, const int src_height,  // global/local memory sizes
    COEFFTYPE coeff, float acc[WPTZ][WPTY][WPTX], const int tid_x, const int tid_y,
    const int tid_z) {
#if CACHE_WORK_TO_REGS == 1
  // Caches data from global memory into registers
  float rmem[WPTZ + 2 * HFS][WPTY + 2 * HFS][WPTX + 2 * HFS];
#  if REVERSE_LOOP_ORDER3 == 0
#    pragma unroll
  for (int x = 0; x < WPTX + 2 * HFS; x++) {
    const int tx = tid_x * WPTX + x;
#    pragma unroll
    for (int y = 0; y < WPTY + 2 * HFS; y++) {
      const int ty = tid_y * WPTY + y;
#    pragma unroll
      for (int z = 0; z < WPTZ + 2 * HFS; z++) {
        const int tz = tid_z * WPTZ + z;
#  else
#    pragma unroll
  for (int z = 0; z < WPTZ + 2 * HFS; z++) {
    const int tz = tid_z * WPTZ + z;
#    pragma unroll
    for (int y = 0; y < WPTY + 2 * HFS; y++) {
      const int ty = tid_y * WPTY + y;
#    pragma unroll
      for (int x = 0; x < WPTX + 2 * HFS; x++) {
        const int tx = tid_x * WPTX + x;
#  endif
        rmem[z][y][x] = src[tz * src_width * src_height + ty * src_width + tx];
      }
    }
  }
#endif  // CACHE_WORK_TO_REGS == 1

// Loops over the neighbourhood
#if REVERSE_LOOP_ORDER2 == 0
#  pragma unroll UNROLL_FACTOR
  for (int fx = 0; fx < FS; fx++) {
#  pragma unroll UNROLL_FACTOR
    for (int fy = 0; fy < FS; fy++) {
#  pragma unroll UNROLL_FACTOR
      for (int fz = 0; fz < FS; fz++) {
#else
#  pragma unroll UNROLL_FACTOR
  for (int fz = 0; fz < FS; fz++) {
#  pragma unroll UNROLL_FACTOR
    for (int fy = 0; fy < FS; fy++) {
#  pragma unroll UNROLL_FACTOR
      for (int fx = 0; fx < FS; fx++) {
#endif
        const float coefficient = coeff[fz * FS * FS + fy * FS + fx];

// Performs the accumulation
#if REVERSE_LOOP_ORDER == 0
#  pragma unroll
        for (int wx = 0; wx < WPTX; wx++) {
#  pragma unroll
          for (int wy = 0; wy < WPTY; wy++) {
#  pragma unroll
            for (int wz = 0; wz < WPTZ; wz++) {
#else
#  pragma unroll
        for (int wz = 0; wz < WPTZ; wz++) {
#  pragma unroll
          for (int wy = 0; wy < WPTY; wy++) {
#  pragma unroll
            for (int wx = 0; wx < WPTX; wx++) {
#endif
              acc[wz][wy][wx] += coefficient *
#if CACHE_WORK_TO_REGS == 1
                                 rmem[wz + fz][wy + fy][wx + fx];
#else
                                 src[(wz + fz + tid_z * WPTZ) * src_width * src_height +
                                     (wy + fy + tid_y * WPTY) * src_width +
                                     (wx + fx + tid_x * WPTX)];
#endif
            }
          }
        }
      }
    }
  }
}

// =================================================================================================

// Stores the result into global memory
inline void StoreResult(__global floatvec* dest, const int width, const int height,
                        float acc[WPTZ][WPTY][WPTX], const int gid_x, const int gid_y,
                        const int gid_z) {
#pragma unroll
  for (int wx = 0; wx < WPTX / VECTOR; wx++) {
    const int gx = gid_x * WPTX / VECTOR + wx;
#pragma unroll
    for (int wy = 0; wy < WPTY; wy++) {
      const int gy = gid_y * WPTY + wy;
#pragma unroll
      for (int wz = 0; wz < WPTZ; wz++) {
        const int gz = gid_z * WPTZ + wz;
        floatvec temp;
#if VECTOR == 1
        temp = acc[wz][wy][wx * VECTOR];
#elif VECTOR == 2
        temp.x = acc[wz][wy][wx * VECTOR];
        temp.y = acc[wz][wy][wx * VECTOR + 1];
#elif VECTOR == 4
        temp.x = acc[wz][wy][wx * VECTOR];
        temp.y = acc[wz][wy][wx * VECTOR + 1];
        temp.z = acc[wz][wy][wx * VECTOR + 2];
        temp.w = acc[wz][wy][wx * VECTOR + 3];
#elif VECTOR == 8
        temp.s0 = acc[wz][wy][wx * VECTOR];
        temp.s1 = acc[wz][wy][wx * VECTOR + 1];
        temp.s2 = acc[wz][wy][wx * VECTOR + 2];
        temp.s3 = acc[wz][wy][wx * VECTOR + 3];
        temp.s4 = acc[wz][wy][wx * VECTOR + 4];
        temp.s5 = acc[wz][wy][wx * VECTOR + 5];
        temp.s6 = acc[wz][wy][wx * VECTOR + 6];
        temp.s7 = acc[wz][wy][wx * VECTOR + 7];
#endif
        dest[gz * height * width / VECTOR + gy * width / VECTOR + gx] = temp;
      }
    }
  }
}

#if ALGORITHM == 1
#  if LOCAL == 0
__kernel void conv(const int width, const int height, const __global float* src, COEFFTYPE coeff,
                   __global floatvec* dest) {
  // Thread identifiers
  const int gid_x = get_global_id(0);  // From 0 to width/WPTX-1
  const int gid_y = get_global_id(1);  // From 0 to height/WPTY-1
  const int gid_z = get_global_id(2);  // From 0 to depth/WPTZ-1

  // Initializes the accumulation registers
  float acc[WPTZ][WPTY][WPTX];
  InitAccRegisters(acc);

  // Accumulates in global memory
  Accumulate(src, width + 2 * HFS, height + 2 * HFS, coeff, acc, gid_x, gid_y, gid_z);

  // Computes and stores the result
  StoreResult(dest, width, height, acc, gid_x, gid_y, gid_z);
}
#  endif

#  if LOCAL == 1
__kernel void conv(const int width, const int height, const __global float* src, COEFFTYPE coeff,
                   __global floatvec* dest) {
  // Thread identifiers
  const int gid_x = get_global_id(0);  // From 0 to width/WPTX-1
  const int gid_y = get_global_id(1);  // From 0 to height/WPTY-1
  const int gid_z = get_global_id(2);  // From 0 to depth/WPTZ-1

  // Local memory
  const int lid_x = get_local_id(0);  // From 0 to TBX
  const int lid_y = get_local_id(1);  // From 0 to TBY
  const int lid_z = get_local_id(2);  // From 0 to TBZ

  const int lmem_width = TBX * WPTX + 2 * HFS + PADDING;
  const int lmem_height = TBY * WPTY + 2 * HFS;
  __local float
      lmem[(TBZ * WPTZ + 2 * HFS) * (TBY * WPTY + 2 * HFS) * (TBX * WPTX + 2 * HFS + PADDING)];

  // Caches data into local memory
  LoadLocalPlusHalo(lmem, lmem_width, lmem_height, src, width + 2 * HFS, height + 2 * HFS, gid_x,
                    gid_y, gid_z, lid_x, lid_y, lid_z);

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Initializes the accumulation registers
  float acc[WPTZ][WPTY][WPTX];
  InitAccRegisters(acc);

  // Accumulates in local memory
  Accumulate(lmem, lmem_width, lmem_height, coeff, acc, lid_x, lid_y, lid_z);

  // Computes and stores the result
  StoreResult(dest, width, height, acc, gid_x, gid_y, gid_z);
}
#  endif  // LOCAL == 1

#  if LOCAL == 2
__kernel void conv(const int width, const int height, const __global float* src, COEFFTYPE coeff,
                   __global floatvec* dest) {
  // Thread identifiers
  const int gid_x = get_local_id(0) + TBX * get_group_id(0);
  const int gid_y = get_local_id(1) + TBY * get_group_id(1);
  const int gid_z = get_local_id(2) + TBZ * get_group_id(2);

  // Local memory
  const int lid_x = get_local_id(0);  // From 0 to TBX_XL
  const int lid_y = get_local_id(1);  // From 0 to TBY_XL
  const int lid_z = get_local_id(2);  // From 0 to TBZ_XL

  const int lmem_width = TBX * WPTX + 2 * HFS + PADDING;
  const int lmem_height = TBY * WPTY + 2 * HFS;
  __local float
      lmem[(TBZ * WPTZ + 2 * HFS) * (TBY * WPTY + 2 * HFS) * (TBX * WPTX + 2 * HFS + PADDING)];

  // Caches data into local memory
  LoadLocalFull(lmem, lmem_width, lmem_height, src, width + 2 * HFS, height + 2 * HFS, gid_x, gid_y,
                gid_z, lid_x, lid_y, lid_z);

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Cancels some threads (those that were only used for loading halo data)
  if ((lid_x >= TBX) || (lid_y >= TBY) || (lid_z >= TBZ)) {
    return;
  }

  // Initializes the accumulation registers
  float acc[WPTZ][WPTY][WPTX];
  InitAccRegisters(acc);

  // Accumulates in local memory
  Accumulate(lmem, lmem_width, lmem_height, coeff, acc, lid_x, lid_y, lid_z);

  // Computes and stores the result
  StoreResult(dest, width, height, acc, gid_x, gid_y, gid_z);
}
#  endif  // LOCAL == 2
#endif    // ALGORITHM == 1

#if ALGORITHM == 2
#  if LOCAL == 1
inline void ShiftAndLoadNextValue(__local float* lmem, const int lmem_width, const int lmem_height,
                                  const __global float* src, const int src_width,
                                  const int src_height, const int gx, const int gy, const int gz,
                                  const int lx, const int ly) {
  lmem[(ly + HFS) * lmem_width + lx + HFS] =
      lmem[HFS * lmem_height * lmem_width + (ly + HFS) * lmem_width + lx + HFS];
  lmem[HFS * lmem_height * lmem_width + (ly + HFS) * lmem_width + lx + HFS] =
      lmem[2 * HFS * lmem_height * lmem_width + (ly + HFS) * lmem_width + lx + HFS];
  lmem[2 * HFS * lmem_height * lmem_width + (ly + HFS) * lmem_width + lx + HFS] =
      src[(gz + 2 * HFS) * src_height * src_width + (gy + HFS) * src_width + gx + HFS];
}

// Loads data (plus the halos) into local memory
inline void ShiftLocalAndLoadNextHalo(__local float* lmem, const int lmem_width,
                                      const int lmem_height, const __global float* src,
                                      const int src_width, const int src_height, const int gid_x,
                                      const int gid_y, const int gz, const int lid_x,
                                      const int lid_y, const int lid_z) {
// Loop over the amount of work per thread
#    pragma unroll
  for (int wy = 0; wy < WPTY; wy++) {
    const int ly = lid_y * WPTY + wy;
    const int gy = gid_y * WPTY + wy;
#    pragma unroll
    for (int wx = 0; wx < WPTX; wx++) {
      const int lx = lid_x * WPTX + wx;
      const int gx = gid_x * WPTX + wx;

      ShiftAndLoadNextValue(lmem, lmem_width, lmem_height, src, src_width, src_height, gx, gy, gz,
                            lx, ly);

      // Computes the conditionals
      const bool low_x = lx < HFS;
      const bool high_x = lx >= TBX * WPTX - HFS;
      const bool low_y = ly < HFS;
      const bool high_y = ly >= TBY * WPTY - HFS;

      if (low_y) {
        ShiftAndLoadNextValue(lmem, lmem_width, lmem_height, src, src_width, src_height, gx,
                              gy - HFS, gz, lx, ly - HFS);
        if (low_x)
          ShiftAndLoadNextValue(lmem, lmem_width, lmem_height, src, src_width, src_height, gx - HFS,
                                gy - HFS, gz, lx - HFS, ly - HFS);
        if (high_x)
          ShiftAndLoadNextValue(lmem, lmem_width, lmem_height, src, src_width, src_height, gx + HFS,
                                gy - HFS, gz, lx + HFS, ly - HFS);
      }
      if (high_y) {
        ShiftAndLoadNextValue(lmem, lmem_width, lmem_height, src, src_width, src_height, gx,
                              gy + HFS, gz, lx, ly + HFS);
        if (low_x)
          ShiftAndLoadNextValue(lmem, lmem_width, lmem_height, src, src_width, src_height, gx - HFS,
                                gy + HFS, gz, lx - HFS, ly + HFS);
        if (high_x)
          ShiftAndLoadNextValue(lmem, lmem_width, lmem_height, src, src_width, src_height, gx + HFS,
                                gy + HFS, gz, lx + HFS, ly + HFS);
      }
      if (low_x)
        ShiftAndLoadNextValue(lmem, lmem_width, lmem_height, src, src_width, src_height, gx - HFS,
                              gy, gz, lx - HFS, ly);
      if (high_x)
        ShiftAndLoadNextValue(lmem, lmem_width, lmem_height, src, src_width, src_height, gx + HFS,
                              gy, gz, lx + HFS, ly);
    }
  }
}

// requirements: TBZ == 1, WPTZ == 1, HFS == 1
__kernel void conv(const int width, const int height, const int depth, const __global float* src,
                   COEFFTYPE coeff, __global floatvec* dest) {
  // Thread identifiers
  const int gid_x = get_global_id(0);  // From 0 to width/WPTX-1
  const int gid_y = get_global_id(1);  // From 0 to height/WPTY-1
  const int gid_z = get_global_id(2) * Z_ITERATIONS;

  // Local memory
  const int lid_x = get_local_id(0);  // From 0 to TBX-1
  const int lid_y = get_local_id(1);  // From 0 to TBY-1
  const int lid_z = get_local_id(2);  // From 0 to TBZ-1

  const int lmem_width = TBX * WPTX + 2 * HFS + PADDING;
  const int lmem_height = TBY * WPTY + 2 * HFS;
  __local float
      lmem[(TBZ * WPTZ + 2 * HFS) * (TBY * WPTY + 2 * HFS) * (TBX * WPTX + 2 * HFS + PADDING)];

  // Accumulation registers
  float acc[WPTZ][WPTY][WPTX];

  // Caches data into local memory
  LoadLocalPlusHalo(lmem, lmem_width, lmem_height, src, width + 2 * HFS, height + 2 * HFS, gid_x,
                    gid_y, gid_z, lid_x, lid_y, lid_z);

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Resets accumulation registers to 0.0f
  InitAccRegisters(acc);

  // Accumulates in global memory
  Accumulate(lmem, lmem_width, lmem_height, coeff, acc, lid_x, lid_y, lid_z);

  // Computes and stores the result
  StoreResult(dest, width, height, acc, gid_x, gid_y, gid_z);

#    pragma unroll 1
  for (int z = 1; z < Z_ITERATIONS; z++) {
    barrier(CLK_LOCAL_MEM_FENCE);

    ShiftLocalAndLoadNextHalo(lmem, lmem_width, lmem_height, src, width + 2 * HFS, height + 2 * HFS,
                              gid_x, gid_y, gid_z + z, lid_x, lid_y, lid_z);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Resets accumulation registers to 0.0f
    InitAccRegisters(acc);

    // Accumulates in global memory
    Accumulate(lmem, lmem_width, lmem_height, coeff, acc, lid_x, lid_y, lid_z);

    // Computes and stores the result
    StoreResult(dest, width, height, acc, gid_x, gid_y, gid_z + z);
  }
}
#  endif  // LOCAL == 1

#  if LOCAL == 2
// Loads data (plus the halos) into local memory
inline void ShiftLocalAndLoadNextFull(__local float* lmem, const int lmem_width,
                                      const int lmem_height, const __global float* src,
                                      const int src_width, const int src_height, const int gid_x,
                                      const int gid_y, const int gz, const int lid_x,
                                      const int lid_y, const int lid_z) {
  float temp[WPTY][WPTX];
// Load new local memory to registers before overwriting it
#    pragma unroll
  for (int wy = 0; wy < WPTY; wy++) {
    const int ly = lid_y * WPTY + wy;
    if (ly < TBY * WPTY + 2 * HFS) {
      const int gy = gid_y * WPTY + wy;
#    pragma unroll
      for (int wx = 0; wx < WPTX; wx++) {
        const int lx = lid_x * WPTX + wx;
        if (lx < TBX * WPTX + 2 * HFS) {
          const int gx = gid_x * WPTX + wx;

          if (lid_z < HFS + 1)
            temp[wy][wx] = lmem[(lid_z + HFS) * lmem_height * lmem_width + ly * lmem_width + lx];
          else
            temp[wy][wx] = src[gz * src_height * src_width + gy * src_width + gx];
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

// Write registers to local memory
#    pragma unroll
  for (int wy = 0; wy < WPTY; wy++) {
    const int ly = lid_y * WPTY + wy;
    if (ly < TBY * WPTY + 2 * HFS)
#    pragma unroll
      for (int wx = 0; wx < WPTX; wx++) {
        const int lx = lid_x * WPTX + wx;
        if (lx < TBX * WPTX + 2 * HFS)
          lmem[lid_z * lmem_height * lmem_width + ly * lmem_width + lx] = temp[wy][wx];
      }
  }
}

__kernel void conv(const int width, const int height, const int depth, const __global float* src,
                   COEFFTYPE coeff, __global floatvec* dest) {
  // Thread identifiers
  const int gid_x = get_local_id(0) + TBX * get_group_id(0);
  const int gid_y = get_local_id(1) + TBY * get_group_id(1);
  const int gid_z = get_local_id(2) + TBZ * get_group_id(2) * Z_ITERATIONS;

  // Local memory
  const int lid_x = get_local_id(0);  // From 0 to TBX_XL
  const int lid_y = get_local_id(1);  // From 0 to TBY_XL
  const int lid_z = get_local_id(2);  // From 0 to TBZ_XL

  const int lmem_width = TBX * WPTX + 2 * HFS + PADDING;
  const int lmem_height = TBY * WPTY + 2 * HFS;
  __local float
      lmem[(TBZ * WPTZ + 2 * HFS) * (TBY * WPTY + 2 * HFS) * (TBX * WPTX + 2 * HFS + PADDING)];

  // Accumulation registers
  float acc[WPTZ][WPTY][WPTX];

  // Caches data into local memory
  LoadLocalFull(lmem, lmem_width, lmem_height, src, width + 2 * HFS, height + 2 * HFS, gid_x, gid_y,
                gid_z, lid_x, lid_y, lid_z);

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Cancels some threads (those that were only used for loading halo data)
  if (lid_x < TBX && lid_y < TBY && lid_z < TBZ) {
    // Resets accumulation registers to 0.0f
    InitAccRegisters(acc);

    // Accumulates in local memory
    Accumulate(lmem, lmem_width, lmem_height, coeff, acc, lid_x, lid_y, lid_z);

    // Computes and stores the result
    StoreResult(dest, width, height, acc, gid_x, gid_y, gid_z);
  }
#    pragma unroll 1
  for (int z = 1; z < Z_ITERATIONS; z++) {
    barrier(CLK_LOCAL_MEM_FENCE);

    ShiftLocalAndLoadNextFull(lmem, lmem_width, lmem_height, src, width + 2 * HFS, height + 2 * HFS,
                              gid_x, gid_y, gid_z + z, lid_x, lid_y, lid_z);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid_x < TBX && lid_y < TBY && lid_z < TBZ) {
      // Resets accumulation registers to 0.0f
      InitAccRegisters(acc);

      // Accumulates in local memory
      Accumulate(lmem, lmem_width, lmem_height, coeff, acc, lid_x, lid_y, lid_z);

      // Computes and stores the result
      StoreResult(dest, width, height, acc, gid_x, gid_y, gid_z + z);
    }
  }
}
#  endif  // LOCAL == 2
#endif    // ALGORITHM == 2
