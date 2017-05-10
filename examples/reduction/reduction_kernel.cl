void atomic_add_global(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
 
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

#if VECTOR_SIZE == 1
   typedef float VEC; 
#endif
#if VECTOR_SIZE == 2
    typedef float2 VEC;
#endif
#if VECTOR_SIZE == 4
    typedef float4 VEC;
#endif
#if VECTOR_SIZE == 8
    typedef float8 VEC;
#endif
#if VECTOR_SIZE == 16
    typedef float16 VEC;
#endif


__kernel void reduce(__global const VEC* in, __global float* out, unsigned int n) {
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);

    __local float buf[WORK_GROUP_SIZE_X];

#if UNBOUNDED_WG == 1
    if (i < n/VECTOR_SIZE) {
#if VECTOR_SIZE == 1
        buf[tid] = in[i];
#endif
#if VECTOR_SIZE == 2
        buf[tid] = in[i].s0+in[i].s1;
#endif
#if VECTOR_SIZE == 4
        buf[tid] = in[i].s0+in[i].s1+in[i].s2+in[i].s3;
#endif
#if VECTOR_SIZE == 8
        buf[tid] = in[i].s0+in[i].s1+in[i].s2+in[i].s3+in[i].s4+in[i].s5+in[i].s6+in[i].s7;
#endif
#if VECTOR_SIZE == 16
        buf[tid] = in[i].s0+in[i].s1+in[i].s2+in[i].s3+in[i].s4+in[i].s5+in[i].s6+in[i].s7+in[i].s8+in[i].s9+in[i].sa+in[i].sb+in[i].sc+in[i].sd+in[i].se+in[i].sf;
#endif
    } else {
        buf[tid] = 0.0f;
    }
#else /*UNBOUNDED_WG != 1*/
    VEC partial;
#if VECTOR_SIZE == 1
    partial = 0.0f;
#endif
#if VECTOR_SIZE == 2
    partial = (0,0);    
#endif
#if VECTOR_SIZE == 4
    partial = (0,0,0,0);
#endif
#if VECTOR_SIZE == 8
    partial = (0,0,0,0,0,0,0,0);
#endif
#if VECTOR_SIZE == 16
    partial = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
#endif
    while (i < n/VECTOR_SIZE) {
        partial += in[i];
        i += WORK_GROUP_SIZE_X*WG_NUM;
    }
#if VECTOR_SIZE == 1
    buf[tid] = partial;
#endif
#if VECTOR_SIZE == 2
    buf[tid] = partial.s0 + partial.s1;    
#endif
#if VECTOR_SIZE == 4
    buf[tid] = partial.s0 + partial.s1 + partial.s2 + partial.s3;
#endif
#if VECTOR_SIZE == 8
    buf[tid] = partial.s0 + partial.s1 + partial.s2 + partial.s3 + partial.s4 + partial.s5 + partial.s6 + partial.s7;
#endif
#if VECTOR_SIZE == 16
    buf[tid] = partial.s0 + partial.s1 + partial.s2 + partial.s3 + partial.s4 + partial.s5 + partial.s6 + partial.s7 + partial.s8 + partial.s9 + partial.sa + partial.sb + partial.sc + partial.sd + partial.se + partial.sf;
#endif
#endif /* UNBOUNDED_WG != 1 */

    barrier(CLK_LOCAL_MEM_FENCE);

#if WORK_GROUP_SIZE_X >= 512
    if (tid < 256)
        buf[tid] += buf[tid + 256];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if WORK_GROUP_SIZE_X >= 256
    if (tid < 128)
        buf[tid] += buf[tid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if WORK_GROUP_SIZE_X >= 128
    if (tid < 64)
        buf[tid] += buf[tid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if WORK_GROUP_SIZE_X >= 64
    if (tid < 32)
        buf[tid] += buf[tid + 32];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if WORK_GROUP_SIZE_X >= 32
    if (tid < 16)
        buf[tid] += buf[tid + 16];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if WORK_GROUP_SIZE_X >= 16
    if (tid < 8)
        buf[tid] += buf[tid + 8];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if WORK_GROUP_SIZE_X >= 8
    if (tid < 4)
        buf[tid] += buf[tid + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if WORK_GROUP_SIZE_X >= 4
    if (tid < 2)
        buf[tid] += buf[tid + 2];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if WORK_GROUP_SIZE_X >= 2
    if (tid < 1) {
        buf[0] += buf[1];
    }
#endif
    if (tid < 1) {
#if USE_ATOMICS == 1
        atomic_add_global(out, buf[0]);
#else
        out[get_global_id(0)] = buf[0];
#endif
    }
}

