#define REAL float

#if GRANULARITY == 1

extern "C" __global__ void gemm_batch(const REAL* A, const REAL* B, REAL* C, int n) {
	int mBlock = blockIdx.x*MGCG_GROUP_SIZE_Y+threadIdx.y;
	int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty*STRIDE_BLOCK + tx;
/* preload data */
#if CACHING_STRATEGY == 1
    int preloadStartA = blockIdx.x*MGCG_GROUP_SIZE_Y*STRIDE_BLOCK*SIZE_A*SIZE_B;
    int preloadStartB = blockIdx.x*MGCG_GROUP_SIZE_Y*STRIDE_BLOCK*SIZE_C*SIZE_A;
	__shared__ REAL bufA[MGCG_GROUP_SIZE_Y*STRIDE_BLOCK*SIZE_A*SIZE_B];
	__shared__ REAL bufB[MGCG_GROUP_SIZE_Y*STRIDE_BLOCK*SIZE_C*SIZE_A];
	for (int i = 0; i < SIZE_A*SIZE_B; i++)
		bufA[i*MGCG_GROUP_SIZE_Y*STRIDE_BLOCK+tid] = A[preloadStartA + i*MGCG_GROUP_SIZE_Y*STRIDE_BLOCK + tid];
	for (int i = 0; i < SIZE_C*SIZE_A; i++)
        bufB[i*MGCG_GROUP_SIZE_Y*STRIDE_BLOCK+tid] = B[preloadStartB + i*MGCG_GROUP_SIZE_Y*STRIDE_BLOCK + tid];
	__syncthreads();
#elif CACHING_STRATEGY == 2
    int preloadStartA = mBlock*STRIDE_BLOCK*SIZE_A*SIZE_B + tx;
    int preloadStartB = mBlock*STRIDE_BLOCK*SIZE_C*SIZE_A + tx;
    REAL bufA[SIZE_A*SIZE_B];
    REAL bufB[SIZE_C*SIZE_A];
    for (int i = 0; i < SIZE_A*SIZE_B; i++)
        bufA[i] = A[preloadStartA + i*STRIDE_BLOCK];
    for (int i = 0; i < SIZE_C*SIZE_A; i++)
        bufB[i] = B[preloadStartB + i*STRIDE_BLOCK];
#endif

/* offsets into memory */
#if CACHING_STRATEGY == 0
    int startA = mBlock*STRIDE_BLOCK*SIZE_A*SIZE_B + tx;
    int startB = mBlock*STRIDE_BLOCK*SIZE_C*SIZE_A + tx;
#else
    int startA = ty*STRIDE_BLOCK*SIZE_A*SIZE_B + tx;
    int startB = ty*STRIDE_BLOCK*SIZE_C*SIZE_A + tx;
#endif
    int startC = mBlock*STRIDE_BLOCK*SIZE_C*SIZE_B + tx;

/* compute multiplication */
    for (int i = 0; i < SIZE_B; i++)
        for (int j = 0; j < SIZE_C; j++) {
            REAL tmp = (REAL)0.0;
            for (int k = 0; k < SIZE_A; k++)
#if CACHING_STRATEGY == 0
                tmp += A[startA + (i*SIZE_A+k)*STRIDE_BLOCK] * B[startB + (k*SIZE_C+j)*STRIDE_BLOCK];
#endif
#if CACHING_STRATEGY == 1
                tmp += bufA[startA + (i*SIZE_A+k)*STRIDE_BLOCK] * bufB[startB + (k*SIZE_C+j)*STRIDE_BLOCK];
#endif
#if CACHING_STRATEGY == 2
                tmp += bufA[i*SIZE_A + k] * bufB[k*SIZE_C + j];
#endif
            C[startC + (i*SIZE_C+j)*STRIDE_BLOCK] = tmp;
        }
}

#endif

