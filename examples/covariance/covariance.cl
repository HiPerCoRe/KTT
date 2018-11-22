#if defined(cl_khr_fp64)  // Khronos extension available?
#  pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#  pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

// other types than `float` not tested
typedef float DATA_TYPE;

__kernel void mean_kernel(__global DATA_TYPE *mean, const __global DATA_TYPE *data,
    DATA_TYPE float_n, const int m, const int n) {
  const int j = get_global_id(0);

  for (int i = 0; i < n; i++) {
    mean[j] += data[i * m + j];
  }
  mean[j] /= (DATA_TYPE)float_n;
}

__kernel void reduce_kernel(
    const __global DATA_TYPE *mean, __global DATA_TYPE *data, const int m, const int n) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);

  data[i * m + j] -= mean[j];
}

__kernel void covar_kernel(
    __global DATA_TYPE *symmat, const __global DATA_TYPE *data, const int m, const int n) {
  int j1 = get_global_id(0);

  for (int j2 = j1; j2 < m; j2++) {
    for (int i = 0; i < n; i++) {
      symmat[j1 * m + j2] += data[i * m + j1] * data[i * m + j2];
    }
    symmat[j2 * m + j1] = symmat[j1 * m + j2];
  }
}

__kernel void triangular_to_symmetric(__global DATA_TYPE *symmat, const int m) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  if (i > j) return;

  symmat[j * m + i] = symmat[i * m + j];
}
