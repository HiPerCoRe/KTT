/**
 * bicg.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

// process BICG_BATCH elements in thread
#define BICG_STEP TILE/BICG_BATCH

__kernel void bicgKernel1(__global float *A, __global float *p, __global float *q, int nx, int ny)
{
	int i = get_global_id(0);

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

__kernel void bicgKernel2(__global float *A, __global float *r, __global float *s, int nx, int ny)
{
	int j = get_global_id(0);

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

inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
	union {
		unsigned int u32;
		float f32;
	} next, expected, current;
	current.f32 = *addr;
	do {
		expected.f32 = current.f32;
		next.f32 = expected.f32 + val;
		current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr,
			expected.u32, next.u32);
	} while (current.u32 != expected.u32);
}

inline void barrier_sh_red() {
#if USE_SHARED_REDUCTION_2 == 1
	barrier(CLK_LOCAL_MEM_FENCE);
#else
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
}

__kernel void bicgReduction1(int m, int n, __global float *y1) {
	int id = get_global_id(0);
	if (id < n) {
		float sum = 0.0f;
		for (int i = 0; i < m / TILE; i++)
			sum += y1[i*n + id];
		y1[id] = sum;
	}
}

__kernel void bicgReduction2(int m, int n, __global float *y2) {
	int id = get_global_id(0);
	if (id < m) {
		float sum = 0.0f;
		for (int i = 0; i < n / ROWS_PROCESSED; i++)
			sum += y2[i*m + id];
		y2[id] = sum;
	}
}

__kernel void bicgFused(__global float *A, __global float *x1, __global float *y1, __global float *x2, __global float *y2, int m, int n)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int bx = get_group_id(0);
	int by = get_group_id(1);

//#if USE_SHARED_MATRIX == 1
	__local float s_A[TILE][TILE + 1];
//#endif
#if USE_SHARED_VECTOR_1 == 1
	__local float s_x1[TILE];
#endif // USE_SHARED_VECTOR_1
#if USE_SHARED_VECTOR_2 == 1
	__local float s_x2[TILE];
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
		barrier(CLK_LOCAL_MEM_FENCE);
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

		barrier(CLK_LOCAL_MEM_FENCE);
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
		barrier(CLK_LOCAL_MEM_FENCE);
#else
		A[(i + tx)*m + bx * TILE + ty] = tmp;
		barrier(CLK_GLOBAL_MEM_FENCE);
#endif

#if BICG_BATCH <= 1
		if (ty < TILE / 2)
	#if USE_SHARED_REDUCTION_1 == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + TILE / 2];
		barrier(CLK_LOCAL_MEM_FENCE);
	#else
			A[(i + tx)*m + bx * TILE + ty] = tmp = tmp + A[(i + tx)*m + bx * TILE + ty + TILE / 2];
		barrier(CLK_GLOBAL_MEM_FENCE);
	#endif // USE_SHARED_REDUCTION_1
#endif // BICG_BATCH

#if BICG_BATCH <= 2
		if (ty < TILE / 4)
	#if USE_SHARED_REDUCTION_1 == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + TILE / 4];
		barrier(CLK_LOCAL_MEM_FENCE);
	#else
			A[(i + tx)*m + bx * TILE + ty] = tmp = tmp + A[(i + tx)*m + bx * TILE + ty + TILE / 4];
		barrier(CLK_GLOBAL_MEM_FENCE);
	#endif // USE_SHARED_REDUCTION_1
#endif // BICG_BATCH

#if BICG_BATCH <= 4
		if (ty < TILE / 8)
	#if USE_SHARED_REDUCTION_1 == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + TILE / 8];
		barrier(CLK_LOCAL_MEM_FENCE);
	#else
			A[(i + tx)*m + bx * TILE + ty] = tmp = tmp + A[(i + tx)*m + bx * TILE + ty + TILE / 8];
		barrier(CLK_GLOBAL_MEM_FENCE);
	#endif // USE_SHARED_REDUCTION_1
#endif // BICG_BATCH

#if BICG_BATCH <= 8 && TILE >= 32
		if (ty < TILE / 16)
	#if USE_SHARED_REDUCTION_1 == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + TILE / 16];
		barrier(CLK_LOCAL_MEM_FENCE);
	#else
			A[(i + tx)*m + bx * TILE + ty] = tmp = tmp + A[(i + tx)*m + bx * TILE + ty + TILE / 16];
		barrier(CLK_GLOBAL_MEM_FENCE);
	#endif // USE_SHARED_REDUCTION_1
#endif // BICG_BATCH

#if TILE >= 64
		if (ty < TILE / 32)
	#if USE_SHARED_REDUCTION_1 == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + TILE / 32];
		barrier(CLK_LOCAL_MEM_FENCE);
	#else
			A[(i + tx)*m + bx * TILE + ty] = tmp = tmp + A[(i + tx)*m + bx * TILE + ty + TILE / 32];
		barrier(CLK_GLOBAL_MEM_FENCE);
	#endif // USE_SHARED_REDUCTION_1
#endif // BICG_BATCH

		if (ty == 0) {
#if ATOMICS == 1
	#if USE_SHARED_REDUCTION_1 == 1
			atomicAdd_g_f(y1 + i + tx, tmp + s_A[tx][1]);
	#else
			atomicAdd_g_f(y1 + i + tx, tmp + A[(i + tx)*m + bx * TILE + 1]);
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
	barrier(CLK_LOCAL_MEM_FENCE);
#if USE_SHARED_REDUCTION_2 == 1
	s_A[ty][tx] = l_sum;
#else
	A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum;
#endif // USE_SHARED_REDUCTION_2

#if BICG_BATCH <= 1
	barrier_sh_red();
	if (ty < TILE / 2)
	#if USE_SHARED_REDUCTION_2 == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + TILE / 2][tx];
	#else
		A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty + TILE / 2)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // BICG_BATCH

#if BICG_BATCH <= 2
	barrier_sh_red();
	if (ty < TILE / 4)
	#if USE_SHARED_REDUCTION_2 == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + TILE / 4][tx];
	#else
		A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty + TILE / 4)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // BICG_BATCH

#if BICG_BATCH <= 4
	barrier_sh_red();
	if (ty < TILE / 8)
	#if USE_SHARED_REDUCTION_2 == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + TILE / 8][tx];
	#else
		A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty + TILE / 8)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // BICG_BATCH

#if BICG_BATCH <= 8 && TILE >= 32
	barrier_sh_red();
	if (ty < TILE / 16)
	#if USE_SHARED_REDUCTION_2 == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + TILE / 16][tx];
	#else
		A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty + TILE / 16)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // BICG_BATCH

#if TILE >= 64
	barrier_sh_red();
	if (ty < TILE / 32)
	#if USE_SHARED_REDUCTION_2 == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + TILE / 32][tx];
	#else
		A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty)*m + bx * TILE + tx] = l_sum = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + ty + TILE / 32)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // BICG_BATCH

	barrier_sh_red();
	if (ty == 0)
#if ATOMICS == 1
	#if USE_SHARED_REDUCTION_2 == 1
		atomicAdd_g_f(y2 + bx * TILE + tx, l_sum + s_A[1][tx]);
	#else
		atomicAdd_g_f(y2 + bx * TILE + tx, l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + 1)*m + bx * TILE + tx]);
	#endif // USE_SHARED_REDUCTION_2
#else // reduced later in bicgReduction2
	#if USE_SHARED_REDUCTION_2 == 1
		y2[bx * TILE + tx + by*m] = l_sum + s_A[1][tx];
	#else
		y2[bx * TILE + tx + by*m] = l_sum + A[(ROWS_PROCESSED*by + ROWS_PROCESSED - TILE + 1)*m + bx * TILE + tx];
	#endif // USE_SHARED_REDUCTION_2
#endif // ATOMICS
}

