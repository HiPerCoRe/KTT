#define REAL float

#if GRANULARITY == 1

extern "C" __global__ void gemm_batch(const REAL* A, const REAL* B, REAL* C, int n) {
	int matrix = blockIdx.x*GROUP_SIZE_X + threadIdx.x;
	int tid = threadIdx.x;
/* preload data */
#if CACHING_STRATEGY == 1
    int preloadStartA = blockIdx.x*GROUP_SIZE_X*SIZE_A*SIZE_B;
    int preloadStartB = blockIdx.x*GROUP_SIZE_X*SIZE_C*SIZE_A;
	__shared__ REAL bufA[GROUP_SIZE_X*SIZE_A*SIZE_B];
	__shared__ REAL bufB[GROUP_SIZE_X*SIZE_C*SIZE_A];
	for (int i = 0; i < SIZE_A*SIZE_B; i++)
		bufA[i*GROUP_SIZE_X+tid] = A[preloadStartA + i*GROUP_SIZE_X + tid];
	for (int i = 0; i < SIZE_C*SIZE_A; i++)
        bufB[i*GROUP_SIZE_X+tid] = B[preloadStartB + i*GROUP_SIZE_X + tid];
	__syncthreads();
#elif CACHING_STRATEGY == 2
    int preloadStartA = matrix*SIZE_A*SIZE_B;
    int preloadStartB = matrix*SIZE_C*SIZE_A;
    REAL bufA[SIZE_A*SIZE_B];
    REAL bufB[SIZE_C*SIZE_A];
    for (int i = 0; i < SIZE_A*SIZE_B; i++)
        bufA[i] = A[preloadStartA + i];
    for (int i = 0; i < SIZE_C*SIZE_A; i++)
        bufB[i] = B[preloadStartB + i];
#endif

/* offsets into memory */
#if CACHING_STRATEGY == 0
    int startA = matrix*SIZE_A*SIZE_B;
    int startB = matrix*SIZE_C*SIZE_A;
#else
    int startA = tid*SIZE_A*SIZE_B;
    int startB = tid*SIZE_C*SIZE_A;
#endif
    int startC = matrix*SIZE_C*SIZE_B;

/* compute multiplication */
    for (int i = 0; i < SIZE_B; i++)
        for (int j = 0; j < SIZE_C; j++) {
            REAL tmp = (REAL)0.0;
            for (int k = 0; k < SIZE_A; k++)
#if CACHING_STRATEGY == 0
                tmp += A[startA + i*SIZE_A + k] * B[startB + k*SIZE_C + j];
#endif
#if CACHING_STRATEGY == 1
                tmp += bufA[startA + i*SIZE_A + k] * bufB[startB + k*SIZE_C + j];
#endif
#if CACHING_STRATEGY == 2
                tmp += bufA[i*SIZE_A + k] * bufB[k*SIZE_C + j];
#endif
            C[startC + i*SIZE_C + j] = tmp;
        }
}

#endif

#if GRANULARITY == 2
extern "C" __global__ void gemm_batch(const REAL* A, const REAL* B, REAL* C, int n) {
    int matrix = blockIdx.x*MGCG_GROUP_SIZE_Y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

/* preload data */
#if CACHING_STRATEGY > 0
    int preloadStartA = blockIdx.x*MGCG_GROUP_SIZE_Y*SIZE_A*SIZE_B;
    int preloadStartB = blockIdx.x*MGCG_GROUP_SIZE_Y*SIZE_C*SIZE_A;
    int myOffset = ty*SIZE_C + tx;
    __shared__ REAL bufA[MGCG_GROUP_SIZE_Y*SIZE_A*SIZE_B];
    #if CACHING_STRATEGY == 1
    __shared__ REAL bufB[MGCG_GROUP_SIZE_Y*SIZE_C*SIZE_A];
    #endif
    for (int i = myOffset; i < SIZE_A*SIZE_B*MGCG_GROUP_SIZE_Y; i+= SIZE_C*MGCG_GROUP_SIZE_Y)
        bufA[i] = A[preloadStartA + i];
    #if CACHING_STRATEGY == 1
     for (int i = myOffset; i < SIZE_C*SIZE_A*MGCG_GROUP_SIZE_Y; i+= SIZE_C*MGCG_GROUP_SIZE_Y)
        bufB[i] = B[preloadStartB + i];
    #endif
    __syncthreads();
#endif
/* offsets into memory */
#if CACHING_STRATEGY == 0
    int startA = matrix*SIZE_A*SIZE_B;
    int startB = matrix*SIZE_C*SIZE_A;
#endif
#if CACHING_STRATEGY == 1
    int startA = ty*SIZE_A*SIZE_B;
    int startB = ty*SIZE_C*SIZE_A;
#endif
#if CACHING_STRATEGY == 2
    int startA = ty*SIZE_A*SIZE_B;
    int startB = matrix*SIZE_C*SIZE_A;
#endif
    int startC = matrix*SIZE_C*SIZE_B;

/* compute multiplication */
#if CACHING_STRATEGY == 2
    REAL tmp[SIZE_B];
    for (int i = 0; i < SIZE_B; i++) 
        tmp[i] = (REAL)0.0;
    for (int i = 0; i < SIZE_A; i++) {
        REAL myB = B[startB + i*SIZE_C + tx];//bufB[startB + i*SIZE_C + tx];
        for (int j = 0; j < SIZE_B; j++)
            tmp[j] += bufA[startA + j*SIZE_A + i] * myB;
    }
    for (int i = 0; i < SIZE_B; i++)
        C[startC + i*SIZE_C + tx] = tmp[i];
#else
    for (int i = 0; i < SIZE_B; i++) {
        REAL tmp = (REAL)0.0;
        for (int k = 0; k < SIZE_A; k++)
    #if CACHING_STRATEGY == 0
            tmp += A[startA + i*SIZE_A + k] * B[startB + k*SIZE_C + tx];
    #endif
    #if CACHING_STRATEGY == 1
            tmp += bufA[startA + i*SIZE_A + k] * bufB[startB + k*SIZE_C + tx];
    #endif
        C[startC + i*SIZE_C + tx] = tmp;
    }
#endif
}
#endif

#if GRANULARITY == 3
extern "C" __global__ void gemm_batch(const REAL* A, const REAL* B, REAL* C, int n) {
    int matrix = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

/* preload data */
#if CACHING_STRATEGY == 1
    int preloadStartA = blockIdx.x*SIZE_A*SIZE_B;
    int preloadStartB = blockIdx.x*SIZE_C*SIZE_A;
    int myOffset = ty*SIZE_C + tx;
    __shared__ REAL bufA[SIZE_A*SIZE_B];
    __shared__ REAL bufB[SIZE_C*SIZE_A];
    for (int i = myOffset; i < SIZE_A*SIZE_B; i+= SIZE_C*MGCG_GROUP_SIZE_Y)
        bufA[i] = A[preloadStartA + i];
     for (int i = myOffset; i < SIZE_C*SIZE_A; i+= SIZE_C*MGCG_GROUP_SIZE_Y)
        bufB[i] = B[preloadStartB + i];
    __syncthreads();
#endif
/* offsets into memory */
#if CACHING_STRATEGY == 0
    int startA = matrix*SIZE_A*SIZE_B;
    int startB = matrix*SIZE_C*SIZE_A;
#else
    int startA = 0;
    int startB = 0;
#endif
    int startC = matrix*SIZE_C*SIZE_B;

/* compute multiplication */
    for (int i = ty; i < SIZE_B; i+= MGCG_GROUP_SIZE_Y) {
        REAL tmp = (REAL)0.0;
        for (int k = 0; k < SIZE_A; k++)
#if CACHING_STRATEGY == 0
            tmp += A[startA + i*SIZE_A + k] * B[startB + k*SIZE_C + tx];
#endif
#if CACHING_STRATEGY == 1
            tmp += bufA[startA + i*SIZE_A + k] * bufB[startB + k*SIZE_C + tx];
#endif
        C[startC + i*SIZE_C + tx] = tmp;
    }
}
#endif
