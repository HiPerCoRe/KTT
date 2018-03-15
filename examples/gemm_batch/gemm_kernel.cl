#define REAL float

#if GRANULARITY == 1

__kernel void gemm_batch(__global const REAL* A, __global const REAL* B, __global const REAL* BT, __global REAL* C, int n) {
	int matrix = get_global_id(0);
	int tid = get_local_id(0);
/* preload data */
#if CACHING_STRATEGY == 1
    int preloadStartA = get_group_id(0)*GROUP_SIZE_X*SIZE_A*SIZE_B;
    int preloadStartB = get_group_id(0)*GROUP_SIZE_X*SIZE_C*SIZE_A;
	__local REAL bufA[GROUP_SIZE_X*SIZE_A*SIZE_B];
	__local REAL bufB[GROUP_SIZE_X*SIZE_C*SIZE_A];
	for (int i = 0; i < SIZE_A*SIZE_B; i++)
		bufA[i*GROUP_SIZE_X+tid] = A[preloadStartA + i*GROUP_SIZE_X + tid];
	for (int i = 0; i < SIZE_C*SIZE_A; i++)
    #if TRANSPOSED_B == 0
        bufB[i*GROUP_SIZE_X+tid] = B[preloadStartB + i*GROUP_SIZE_X + tid];
    #else
        bufB[i*GROUP_SIZE_X+tid] = BT[preloadStartB + i*GROUP_SIZE_X + tid];
    #endif
	barrier(CLK_LOCAL_MEM_FENCE);
#elif CACHING_STRATEGY == 2
    int preloadStartA = matrix*SIZE_A*SIZE_B;
    int preloadStartB = matrix*SIZE_C*SIZE_A;
    REAL bufA[SIZE_A*SIZE_B];
    REAL bufB[SIZE_C*SIZE_A];
    for (int i = 0; i < SIZE_A*SIZE_B; i++)
        bufA[i] = A[preloadStartA + i];
    for (int i = 0; i < SIZE_C*SIZE_A; i++)
    #if TRANSPOSED_B == 0
        bufB[i] = B[preloadStartB + i];
    #else
        bufB[i] = BT[preloadStartB + i];
    #endif
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
    #if TRANSPOSED_B == 0
                tmp += A[startA + i*SIZE_A + k] * B[startB + k*SIZE_C + j];
    #else
                tmp += A[startA + i*SIZE_A + k] * BT[startB + j*SIZE_A + k];
    #endif
#endif
#if CACHING_STRATEGY == 1
    #if TRANSPOSED_B == 0
                tmp += bufA[startA + i*SIZE_A + k] * bufB[startB + k*SIZE_C + j];
    #else
                tmp += bufA[startA + i*SIZE_A + k] * bufB[startB + j*SIZE_A + k];
    #endif
#endif
#if CACHING_STRATEGY == 2
    #if TRANSPOSED_B == 0
                tmp += bufA[i*SIZE_A + k] * bufB[k*SIZE_C + j];
    #else
                tmp += bufA[i*SIZE_A + k] * bufB[j*SIZE_A + k];
    #endif
#endif
            C[startC + i*SIZE_C + j] = tmp;
        }
}

#endif

#if GRANULARITY == 2
__kernel void gemm_batch(__global const REAL* A, __global const REAL* B, __global const REAL* BT, __global REAL* C, int n) {
    int matrix = get_group_id(0)*MG_GROUP_SIZE_Y + get_local_id(1);
    int tx = get_local_id(0);
    int ty = get_local_id(1);

/* preload data */
#if CACHING_STRATEGY == 1
    int preloadStartA = get_group_id(0)*MG_GROUP_SIZE_Y*SIZE_A*SIZE_B;
    int preloadStartB = get_group_id(0)*MG_GROUP_SIZE_Y*SIZE_C*SIZE_A;
    int myOffset = ty*SIZE_C + tx;
    __local REAL bufA[MG_GROUP_SIZE_Y*SIZE_A*SIZE_B];
    __local REAL bufB[MG_GROUP_SIZE_Y*SIZE_C*SIZE_A];
    for (int i = myOffset; i < SIZE_A*SIZE_B*MG_GROUP_SIZE_Y; i+= SIZE_C*MG_GROUP_SIZE_Y)
        bufA[i] = A[preloadStartA + i];
     for (int i = myOffset; i < SIZE_C*SIZE_A*MG_GROUP_SIZE_Y; i+= SIZE_C*MG_GROUP_SIZE_Y)
    #if TRANSPOSED_B == 0
        bufB[i] = B[preloadStartB + i];
    #else
        bufB[i] = BT[preloadStartB + i];
    #endif
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

/* offsets into memory */
#if CACHING_STRATEGY == 0
    int startA = matrix*SIZE_A*SIZE_B;
    int startB = matrix*SIZE_C*SIZE_A;
#else
    int startA = ty*SIZE_A*SIZE_B;
    int startB = ty*SIZE_C*SIZE_A;
#endif
    int startC = matrix*SIZE_C*SIZE_B;

/* compute multiplication */
    for (int i = 0; i < SIZE_B; i++) {
        REAL tmp = (REAL)0.0;
        for (int k = 0; k < SIZE_A; k++)
#if CACHING_STRATEGY == 0
    #if TRANSPOSED_B == 0
            tmp += A[startA + i*SIZE_A + k] * B[startB + k*SIZE_C + tx];
    #else
            tmp += A[startA + i*SIZE_A + k] * BT[startB + tx*SIZE_A + k];
    #endif
#endif
#if CACHING_STRATEGY == 1
    #if TRANSPOSED_B == 0
            tmp += bufA[startA + i*SIZE_A + k] * bufB[startB + k*SIZE_C + tx];
    #else
            tmp += bufA[startA + i*SIZE_A + k] * bufB[startB + tx*SIZE_A + k];
    #endif
#endif
        C[startC + i*SIZE_C + tx] = tmp;
    }

}
#endif

