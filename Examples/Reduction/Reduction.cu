#if VECTOR_SIZE == 1
   typedef float VEC; 
#endif
#if VECTOR_SIZE == 2
    typedef float2 VEC;
#endif
#if VECTOR_SIZE == 4
    typedef float4 VEC;
#endif
/*#if VECTOR_SIZE == 8
    typedef float8 VEC;
#endif
#if VECTOR_SIZE == 16
    typedef float16 VEC;
#endif*/

inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

extern "C" __global__ void reduce( const VEC* in,  float* out, unsigned int n, unsigned int inOffset, unsigned int outOffset) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    __shared__ float buf[WORK_GROUP_SIZE_X];

#if UNBOUNDED_WG == 1
    unsigned int addr = i+inOffset;
    if (i < (n+VECTOR_SIZE-1)/VECTOR_SIZE) {
#if VECTOR_SIZE == 1
        buf[tid] = in[addr];
#endif
#if VECTOR_SIZE == 2
        buf[tid] = in[addr].x+in[addr].y;
#endif
#if VECTOR_SIZE == 4
        buf[tid] = in[addr].x+in[addr].y+in[addr].z+in[addr].w;
#endif
/*#if VECTOR_SIZE == 8
        buf[tid] = in[addr].s0+in[addr].s1+in[addr].s2+in[addr].s3+in[addr].s4+in[addr].s5+in[addr].s6+in[addr].s7;
#endif
#if VECTOR_SIZE == 16
        buf[tid] = in[addr].s0+in[addr].s1+in[addr].s2+in[addr].s3+in[addr].s4+in[addr].s5+in[addr].s6+in[addr].s7+in[addr].s8+in[addr].s9+in[addr].sa+in[addr].sb+in[addr].sc+in[addr].sd+in[addr].se+in[addr].sf;
#endif*/
    } else {
        buf[tid] = 0.0f;
    }
#else /*UNBOUNDED_WG != 1*/
    VEC partial;
#if VECTOR_SIZE == 1
    partial = 0.0f;
#endif
#if VECTOR_SIZE == 2
    partial.x = 0.0f;
    partial.y = 0.0f;    
#endif
#if VECTOR_SIZE == 4
    partial.x = 0.0f;
    partial.y = 0.0f;
    partial.z = 0.0f;
    partial.w = 0.0f;
#endif
/*#if VECTOR_SIZE == 8
    partial = (0,0,0,0,0,0,0,0);
#endif
#if VECTOR_SIZE == 16
    partial = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
#endif*/
    while (i < n/VECTOR_SIZE) {
        partial += in[i+inOffset];
        i += WORK_GROUP_SIZE_X*WG_NUM;
    }
#if VECTOR_SIZE == 1
    buf[tid] = partial;
#endif
#if VECTOR_SIZE == 2
    buf[tid] = partial.x + partial.y;    
#endif
#if VECTOR_SIZE == 4
    buf[tid] = partial.x + partial.y + partial.z + partial.w;
#endif
/*#if VECTOR_SIZE == 8
    buf[tid] = partial.s0 + partial.s1 + partial.s2 + partial.s3 + partial.s4 + partial.s5 + partial.s6 + partial.s7;
#endif
#if VECTOR_SIZE == 16
    buf[tid] = partial.s0 + partial.s1 + partial.s2 + partial.s3 + partial.s4 + partial.s5 + partial.s6 + partial.s7 + partial.s8 + partial.s9 + partial.sa + partial.sb + partial.sc + partial.sd + partial.se + partial.sf;
#endif*/
#endif /* UNBOUNDED_WG != 1 */

    __syncthreads();

#if WORK_GROUP_SIZE_X >= 512
    if (tid < 256)
        buf[tid] += buf[tid + 256];
    __syncthreads();
#endif
#if WORK_GROUP_SIZE_X >= 256
    if (tid < 128)
        buf[tid] += buf[tid + 128];
    __syncthreads();
#endif
#if WORK_GROUP_SIZE_X >= 128
    if (tid < 64)
        buf[tid] += buf[tid + 64];
    __syncthreads();
#endif
#if WORK_GROUP_SIZE_X >= 64
    if (tid < 32)
        buf[tid] += buf[tid + 32];
    __syncthreads();
#endif
#if WORK_GROUP_SIZE_X >= 32
    if (tid < 16)
        buf[tid] += buf[tid + 16];
    __syncthreads();
#endif
#if WORK_GROUP_SIZE_X >= 16
    if (tid < 8)
        buf[tid] += buf[tid + 8];
    __syncthreads();
#endif
#if WORK_GROUP_SIZE_X >= 8
    if (tid < 4)
        buf[tid] += buf[tid + 4];
    __syncthreads();
#endif
#if WORK_GROUP_SIZE_X >= 4
    if (tid < 2)
        buf[tid] += buf[tid + 2];
    __syncthreads();
#endif
#if WORK_GROUP_SIZE_X >= 2
    if (tid < 1) {
        buf[0] += buf[1];
    }
#endif
#if USE_ATOMICS == 1
    if (tid < 1)
        atomicAdd(out+outOffset, buf[0]);
#else
    // the last group
    if (blockIdx.x == gridDim.x-1) {
        if (tid == 0)
            out[blockIdx.x + outOffset] = buf[0];
        else if (tid < VECTOR_SIZE)
            out[blockIdx.x + outOffset + tid] = 0.0f;
    }
    // other groups
    else {
        if (tid == 0)
            out[blockIdx.x + outOffset] = buf[0];
    }
#endif
}
