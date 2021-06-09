 // process BICG_BATCH elements in thread
#define BICG_BATCH 8
#define BICG_STEP 32/BICG_BATCH

typedef float DATA_TYPE;

extern "C" __global__ void bicgKernel1( DATA_TYPE *A,  DATA_TYPE *p,  DATA_TYPE *q, int m, int n)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i < n)
	{
		q[i] = 0.0;

		int j;
		for (j = 0; j < m; j++)
		{
			q[i] += A[i * m + j] * p[j];
		}
	}

}

extern "C" __global__ void bicgKernel2( DATA_TYPE *A,  DATA_TYPE *r,  DATA_TYPE *s, int m, int n)
{
	int j = blockDim.x*blockIdx.x + threadIdx.x;

	if (j < m)
	{
		s[j] = 0.0;

		int i;
		for (i = 0; i < n; i++)
		{
			s[j] += A[i * m + j] * r[i];
		}
	}

}

extern "C" __global__ void bicgFusedRef( float *A,  float *x1,  float *y1,  float *x2,  float *y2, int m, int n)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	__shared__ float s_A[32][33];
	__shared__ float s_x1[32];
	__shared__ float s_x2[32];

	float l_sum = 0.0f;

	// load x2
	if (ty == 0)
		s_x2[tx] = x2[bx * 32 + tx];
	for (int i = m*by; i < m*(by + 1); i += 32) {
		// load x1
		if (ty == 1)
			s_x1[tx] = x1[i + tx];
		__syncthreads();

		for (int j = 0; j < 32; j += BICG_STEP) {
			s_A[ty + j][tx] = A[(i + ty + j)*n + bx * 32 + tx];
			l_sum += s_A[ty + j][tx] * s_x1[ty + j];
		}
		__syncthreads();
		float tmp = 0.0f;

		for (int j = 0; j < 32; j += BICG_STEP)
			tmp += s_A[tx][ty + j] * s_x2[ty + j];
		s_A[tx][ty] = tmp;
		__syncthreads();

		if (ty < 2)
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + 2];
		__syncthreads();

		if (ty == 0) {
			atomicAdd(y2 + i + tx, tmp + s_A[tx][1]);
		}
	}

	// compute total sum
	__syncthreads();
	s_A[ty][tx] = l_sum;
	__syncthreads();
	if (ty < 2) {
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + 2][tx];
	}
	__syncthreads();
	if (ty == 0) {
		atomicAdd(y1 + bx * 32 + tx, l_sum + s_A[1][tx]);
	}
}
