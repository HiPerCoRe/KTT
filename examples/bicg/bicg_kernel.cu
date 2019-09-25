#define BICG_STEP TILE/BICG_BATCH

extern "C" __global__ void bicgKernel1( float *A,  float *p,  float *q, int nx, int ny)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i < nx)
	{
		q[i] = 0.0;

		int j;
		for (j = 0; j < ny; j++)
		{
			q[i] += A[i * ny + j] * p[j];
		}
	}
}

extern "C" __global__ void bicgKernel2( float *A,  float *r,  float *s, int nx, int ny)
{
	int j = blockDim.x*blockIdx.x + threadIdx.x;

	if (j < ny)
	{
		s[j] = 0.0;

		int i;
		for (i = 0; i < nx; i++)
		{
			s[j] += A[i * ny + j] * r[i];
		}
	}
}

extern "C" __global__ void bicgReduction1(int m, int n,  float *y1) {
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	if (id < n) {
		float sum = 0.0f;
		for (int i = 0; i < m / TILE; i++)
			sum += y1[i*n + id];
		y1[id] = sum;
	}
}

extern "C" __global__ void bicgReduction2(int m, int n,  float *y2) {
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	if (id < m) {
		float sum = 0.0f;
		for (int i = 0; i < n / ROWS_PROCESSED; i++)
			sum += y2[i*m + id];
		y2[id] = sum;
	}
}

extern "C" __global__ void bicgFused( float *A,  float *x1,  float *y1,  float *x2,  float *y2, int m, int n)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

//#if USE_SHARED_MATRIX == 1
	__shared__ float s_A[TILE][TILE + 1];
//#endif
#if USE_SHARED_VECTOR_1 == 1
	__shared__ float s_x1[TILE];
#endif // USE_SHARED_VECTOR_1
#if USE_SHARED_VECTOR_2 == 1
	__shared__ float s_x2[TILE];
#endif // USE_SHARED_VECTOR_2

	float l_sum = 0.0f;

	// load x1
#if USE_SHARED_VECTOR_1 == 1
	if (ty == 0)
		s_x1[tx] = x1[bx * TILE + tx];
#endif // USE_SHARED_VECTOR_1

#pragma unroll 1
	for (int i = ROWS_PROCESSED*by; i < ROWS_PROCESSED*(by + 1); i += TILE) {

		// load x2
#if USE_SHARED_VECTOR_2 == 1
		if (ty == 1) {
			s_x2[tx] = x2[i + tx];
		}
#endif // USE_SHARED_VECTOR_2
#if USE_SHARED_MATRIX == 1 || USE_SHARED_VECTOR_2 == 1
		__syncthreads();
#endif
// multiply x2
#if UNROLL_BICG_STEP == 1
#pragma unroll
#endif
		for (int j = 0; j < TILE; j += BICG_STEP) {
#if USE_SHARED_MATRIX == 1
			s_A[ty + j][tx] = A[(i + ty + j)*m + bx * TILE + tx];
	#if USE_SHARED_VECTOR_2 == 1
			l_sum += s_A[ty + j][tx] * s_x2[ty + j];
	#else
			l_sum += s_A[ty + j][tx] * x2[i + ty + j];
	#endif // USE_SHARED_VECTOR_2
#else
	#if USE_SHARED_VECTOR_2 == 1
			l_sum += A[(i + ty + j)*m + bx * TILE + tx] * s_x2[ty + j];
	#else
			l_sum += A[(i + ty + j)*m + bx * TILE + tx] * x2[i + ty + j];
	#endif // USE_SHARED_VECTOR_2
#endif // USE_SHARED_MATRIX
		}

		__syncthreads();
		float tmp = 0.0f;

// multiply x1
#if UNROLL_BICG_STEP == 1
#pragma unroll
#endif
		for (int j = 0; j < TILE; j += BICG_STEP)
#if USE_SHARED_MATRIX == 1
	#if USE_SHARED_VECTOR_1 == 1
			tmp += s_A[tx][ty + j] * s_x1[ty + j];
	#else
			tmp += s_A[tx][ty + j] * x1[bx*TILE + ty + j];
	#endif // USE_SHARED_VECTOR_1
#else
	#if USE_SHARED_VECTOR_1 == 1
			tmp += A[(i + tx)*m + bx * TILE + ty + j] * s_x1[ty + j];
	#else
			tmp += A[(i + tx)*m + bx * TILE + ty + j] * x1[bx*TILE + ty + j];
	#endif // USE_SHARED_VECTOR_1
#endif // USE_SHARED_MATRIX

#if USE_SHARED_REDUCTION_1 == 1
		s_A[tx][ty] = tmp;
		__syncthreads();
#else
		A[(i + tx)*m + bx * TILE + ty] = tmp;
		__syncthreads();
#endif

#if BICG_BATCH <= 1
		if (ty < TILE / 2)
	#if USE_SHARED_REDUCTION_1 == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + TILE / 2];
		__syncthreads();
	#else
			A[(i + tx)*m + bx * TILE + ty] = tmp = tmp + A[(i + tx)*m + bx * TILE + ty + TILE / 2];
		__syncthreads();
	#endif // USE_SHARED_REDUCTION_1
#endif // BICG_BATCH

#if BICG_BATCH <= 2
		if (ty < TILE / 4)
	#if USE_SHARED_REDUCTION_1 == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + TILE / 4];
		__syncthreads();
	#else
			A[(i + tx)*m + bx * TILE + ty] = tmp = tmp + A[(i + tx)*m + bx * TILE + ty + TILE / 4];
		__syncthreads();
	#endif // USE_SHARED_REDUCTION_1
#endif // BICG_BATCH

#if BICG_BATCH <= 4
		if (ty < TILE / 8)
	#if USE_SHARED_REDUCTION_1 == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + TILE / 8];
		__syncthreads();
	#else
			A[(i + tx)*m + bx * TILE + ty] = tmp = tmp + A[(i + tx)*m + bx * TILE + ty + TILE / 8];
		__syncthreads();
	#endif // USE_SHARED_REDUCTION_1
#endif // BICG_BATCH

#if BICG_BATCH <= 8 && TILE >= 32
		if (ty < TILE / 16)
	#if USE_SHARED_REDUCTION_1 == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + TILE / 16];
		__syncthreads();
	#else
			A[(i + tx)*m + bx * TILE + ty] = tmp = tmp + A[(i + tx)*m + bx * TILE + ty + TILE / 16];
		__syncthreads();
	#endif // USE_SHARED_REDUCTION_1
#endif // BICG_BATCH

#if TILE >= 64
		if (ty < TILE / 32)
	#if USE_SHARED_REDUCTION_1 == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + TILE / 32];
		__syncthreads();
	#else
			A[(i + tx)*m + bx * TILE + ty] = tmp = tmp + A[(i + tx)*m + bx * TILE + ty + TILE / 32];
		__syncthreads();
	#endif // USE_SHARED_REDUCTION_1
#endif // BICG_BATCH

		if (ty == 0) {
#if ATOMICS == 1
	#if USE_SHARED_REDUCTION_1 == 1
			atomicAdd(y1 + i + tx, tmp + s_A[tx][1]);
	#else
			atomicAdd(y1 + i + tx, tmp + A[(i + tx)*m + bx * TILE + 1]);
	#endif // USE_SHARED_REDUCTION_1
#else // reduced later in bicgReduction1
	#if USE_SHARED_REDUCTION_1 == 1
			y1[i + tx + bx*n] = tmp + s_A[tx][1];
	#else
			y1[i + tx + bx*n] = tmp + A[(i + tx)*m + bx * TILE + 1];
	#endif // USE_SHARED_REDUCTION_1
#endif // ATOMICS
		}
	}

	// compute total sum
	__syncthreads();
#if USE_SHARED_REDUCTION_2 == 1
	s_A[ty][tx] = l_sum;
#else
	A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum;
#endif // USE_SHARED_REDUCTION_2

#if BICG_BATCH <= 1
	__syncthreads();
	if (ty < TILE / 2)
	#if USE_SHARED_REDUCTION_2 == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + TILE / 2][tx];
	#else
		A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty + TILE / 2)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // BICG_BATCH

#if BICG_BATCH <= 2
	__syncthreads();
	if (ty < TILE / 4)
	#if USE_SHARED_REDUCTION_2 == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + TILE / 4][tx];
	#else
		A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty + TILE / 4)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // BICG_BATCH

#if BICG_BATCH <= 4
	__syncthreads();
	if (ty < TILE / 8)
	#if USE_SHARED_REDUCTION_2 == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + TILE / 8][tx];
	#else
		A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty + TILE / 8)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // BICG_BATCH

#if BICG_BATCH <= 8 && TILE >= 32
	__syncthreads();
	if (ty < TILE / 16)
	#if USE_SHARED_REDUCTION_2 == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + TILE / 16][tx];
	#else
		A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty + TILE / 16)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // BICG_BATCH

#if TILE >= 64
	__syncthreads();
	if (ty < TILE / 32)
	#if USE_SHARED_REDUCTION_2 == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + TILE / 32][tx];
	#else
		A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty + TILE / 32)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // BICG_BATCH

	__syncthreads();
	if (ty == 0)
#if ATOMICS == 1
	#if USE_SHARED_REDUCTION_2 == 1
		atomicAdd(y2 + bx * TILE + tx, l_sum + s_A[1][tx]);
	#else
		atomicAdd(y2 + bx * TILE + tx, l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + 1)*m + bx * TILE + tx]);
	#endif // USE_SHARED_REDUCTION_2
#else // reduced later in bicgReduction2
	#if USE_SHARED_REDUCTION_2 == 1
		y2[bx * TILE + tx + by*m] = l_sum + s_A[1][tx];
	#else
		y2[bx * TILE + tx + by*m] = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + 1)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // ATOMICS
}

