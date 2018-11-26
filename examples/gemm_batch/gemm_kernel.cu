#define REAL float
#define REAL2 float2
#define REAL4 float4

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
    int matrixBatch = blockIdx.x*MGCG_GROUP_SIZE_Y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

/* preload data */
#if CACHING_STRATEGY > 0
    int preloadStartA = matrixBatch*SIZE_A*SIZE_B;
    int preloadStartB = matrixBatch*SIZE_C*SIZE_A;
    int myOffset = ty*SIZE_C + tx;
    __shared__ REAL bufA[MGCG_GROUP_SIZE_Y*(SIZE_A+PADD)*SIZE_B];
    #if CACHING_STRATEGY == 1
    __shared__ REAL bufB[MGCG_GROUP_SIZE_Y*SIZE_C*(SIZE_A+PADD)];
    #endif
    #if DIRECT_WRITE == 0
    __shared__ REAL bufC[MGCG_GROUP_SIZE_Y*SIZE_C*SIZE_B];
    #endif
    for (int i = myOffset; i < SIZE_A*SIZE_B*MGCG_GROUP_SIZE_Y; i+= SIZE_C*MGCG_GROUP_SIZE_Y) {
#if PADD == 0
        bufA[i] = A[preloadStartA + i];
#else
        int padd = i/SIZE_A;
        bufA[i+padd] = A[preloadStartA + i];
#endif
    }
    #if CACHING_STRATEGY == 1
     for (int i = myOffset; i < SIZE_C*SIZE_A*MGCG_GROUP_SIZE_Y; i+= SIZE_C*MGCG_GROUP_SIZE_Y) {
#if PADD == 0
        bufB[i] = B[preloadStartB + i];
#else
        int padd = SIZE_C*(i/(SIZE_A*SIZE_C));
        bufB[i+padd] = B[preloadStartB + i];
#endif
    }
    #endif
    __syncthreads();
#endif
/* offsets into memory */
#if CACHING_STRATEGY == 0
    int startA = matrix*SIZE_A*SIZE_B;
    int startB = matrix*SIZE_C*SIZE_A;
#endif
#if CACHING_STRATEGY == 1
    int startA = ty*(SIZE_A+PADD)*SIZE_B;
    int startB = ty*SIZE_C*(SIZE_A+PADD);
#endif
#if CACHING_STRATEGY == 2
    int startA = ty*(SIZE_A+PADD)*SIZE_B;
    int startB = matrix*SIZE_C*SIZE_A;
#endif
#if DIRECT_WRITE == 0
    int startC = matrixBatch*SIZE_C*SIZE_B;
#else
    int startC = matrix*SIZE_C*SIZE_B;
#endif

/* compute multiplication */
#if CACHING_STRATEGY == 2
    REAL tmp[SIZE_B];
    for (int i = 0; i < SIZE_B; i++) 
        tmp[i] = (REAL)0.0;
    for (int i = 0; i < SIZE_A; i++) {
        REAL myB = B[startB + i*SIZE_C + tx];//bufB[startB + i*SIZE_C + tx];
        for (int j = 0; j < SIZE_B; j++)
            tmp[j] += bufA[startA + j*(SIZE_A+PADD) + i] * myB;
    }
#if DIRECT_WRITE == 1
    for (int i = 0; i < SIZE_B; i++) {
        C[startC + i*SIZE_C + tx] = tmp[i];
    }
#else
    for (int i = 0; i < SIZE_B; i++) {
        bufC[ty*SIZE_C*SIZE_B + i*SIZE_C + tx] = tmp[i];
    }
    __syncthreads();
    for (int i = myOffset; i < SIZE_C*SIZE_B*MGCG_GROUP_SIZE_Y; i+= SIZE_C*MGCG_GROUP_SIZE_Y) {
        C[startC + i] = bufC[i];
    }
#endif
#else
    for (int i = 0; i < SIZE_B; i++) {
        REAL tmp = (REAL)0.0;
        for (int k = 0; k < SIZE_A; k++) {
    #if CACHING_STRATEGY == 0
            tmp += A[startA + i*SIZE_A + k] * B[startB + k*SIZE_C + tx];
    #endif
    #if CACHING_STRATEGY == 1
            tmp += bufA[startA + i*(SIZE_A+PADD) + k] * bufB[startB + k*SIZE_C + tx];
    #endif
        }
#if DIRECT_WRITE == 1
        C[startC + i*SIZE_C + tx] = tmp;
#else
        bufC[ty*SIZE_C*SIZE_B + i*SIZE_C + tx] = tmp;
#endif
    }
#if DIRECT_WRITE == 0
    //TODO fix redundancy
    __syncthreads();
    for (int i = myOffset; i < SIZE_C*SIZE_B*MGCG_GROUP_SIZE_Y; i+= SIZE_C*MGCG_GROUP_SIZE_Y) {
        C[startC + i] = bufC[i];
    }
#endif
#endif
}
#endif

#if GRANULARITY == 3
extern "C" __global__ void gemm_batch(const REAL* A, const REAL* B, REAL* C, int n) {
    int matrix = blockIdx.x*CG_GROUP_SIZE_Z + threadIdx.z;
    int matrixBatch = blockIdx.x*CG_GROUP_SIZE_Z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

/* preload data */
#if CACHING_STRATEGY == 1 or CACHING_STRATEGY == 2
    int preloadStartA = matrixBatch*SIZE_A*SIZE_B;
    int preloadStartB = matrixBatch*SIZE_C*SIZE_A;
    int myOffset = tz*SIZE_C*MGCG_GROUP_SIZE_Y + ty*SIZE_C + tx;
    __shared__ REAL bufA[CG_GROUP_SIZE_Z*(SIZE_A+PADD)*SIZE_B];
    __shared__ REAL bufB[CG_GROUP_SIZE_Z*SIZE_C*(SIZE_A+PADD)];
#if DIRECT_WRITE == 0
    __shared__ REAL bufC[CG_GROUP_SIZE_Z*SIZE_C*SIZE_B];
#endif
    for (int i = myOffset; i < SIZE_A*SIZE_B*CG_GROUP_SIZE_Z; i+= SIZE_C*MGCG_GROUP_SIZE_Y*CG_GROUP_SIZE_Z) {
#if PADD == 0
        bufA[i] = A[preloadStartA + i];
#else
        int padd = i/SIZE_A;
        bufA[i+padd] = A[preloadStartA + i];
#endif
    }
     for (int i = myOffset; i < SIZE_C*SIZE_A*CG_GROUP_SIZE_Z; i+= SIZE_C*MGCG_GROUP_SIZE_Y*CG_GROUP_SIZE_Z) {
#if PADD == 0
        bufB[i] = B[preloadStartB + i];
#else
        int padd = SIZE_C*(i/(SIZE_A*SIZE_C));
        bufB[i+padd] = B[preloadStartB + i];
#endif
    }
    __syncthreads();
#endif
/* offsets into memory */
#if CACHING_STRATEGY == 0
    int startA = matrix*SIZE_A*SIZE_B;
    int startB = matrix*SIZE_C*SIZE_A;
#else
    int startA = tz*(SIZE_A+PADD)*SIZE_B;
    int startB = tz*SIZE_C*(SIZE_A+PADD);
#endif
#if DIRECT_WRITE == 0
    int startC = matrixBatch*SIZE_C*SIZE_B;
#else
    int startC = matrix*SIZE_C*SIZE_B;
#endif

/* compute multiplication */
#if CACHING_STRATEGY < 2
    for (int i = ty; i < SIZE_B; i+= MGCG_GROUP_SIZE_Y) {
        REAL tmp = (REAL)0.0;
        for (int k = 0; k < SIZE_A; k++)
#if CACHING_STRATEGY == 0
            tmp += A[startA + i*SIZE_A + k] * B[startB + k*SIZE_C + tx];
#endif
#if CACHING_STRATEGY == 1
            tmp += bufA[startA + i*(SIZE_A+PADD) + k] * bufB[startB + k*SIZE_C + tx];
#endif
#if DIRECT_WRITE == 0
        bufC[tz*SIZE_C*SIZE_B + i*SIZE_C + tx] = tmp;
#else
        C[startC + i*SIZE_C + tx] = tmp;
#endif
    }
#else /* CACHING_STRATEGY == 2*/
    const int batch_base = SIZE_B/MGCG_GROUP_SIZE_Y;
    const int batch_peel = (SIZE_B+MGCG_GROUP_SIZE_Y-1)/MGCG_GROUP_SIZE_Y;
    REAL tmp[batch_peel];
    for (int i = 0; i < batch_peel; i++)
        tmp[i] = REAL(0.0);
    for (int k = 0; k < SIZE_A; k++) {
        REAL myB = bufB[startB + k*SIZE_C + tx];
        for (int i = 0; i < batch_base; i++) {
            tmp[i] +=  bufA[startA + (i*MGCG_GROUP_SIZE_Y+ty)*(SIZE_A+PADD) + k] * myB;
        }
        for (int i = batch_base; i < batch_peel; i++) {
            tmp[i] += (i*MGCG_GROUP_SIZE_Y+ty < SIZE_B) ? bufA[startA + (i*MGCG_GROUP_SIZE_Y+ty)*(SIZE_A+PADD) + k] * myB : (REAL)0.0;
        }
    }
    for (int i = 0; i < batch_peel; i++) {
        int index = i*MGCG_GROUP_SIZE_Y+ty;
        if (index < SIZE_B)
#if DIRECT_WRITE == 0
            bufC[tz*SIZE_C*SIZE_B + index*SIZE_C + tx] = tmp[i];
#else
            C[startC + index*SIZE_C + tx] = tmp[i];
#endif
    }
#endif
#if DIRECT_WRITE == 0
    __syncthreads();
    for (int i = myOffset; i < SIZE_C*SIZE_B*CG_GROUP_SIZE_Z; i+= SIZE_C*MGCG_GROUP_SIZE_Y*CG_GROUP_SIZE_Z) {
        C[startC + i] = bufC[i];
    }
#endif
}
#endif

