#if USE_CONSTANT_MEMORY == 0
    #define MEMORY_TYPE_AOS __global const
    #define MEMORY_TYPE_SOA __global const
#elif USE_CONSTANT_MEMORY == 1
    #if USE_SOA == 0
        #define MEMORY_TYPE_AOS __constant
        #define MEMORY_TYPE_SOA __global const
    #else
        #define MEMORY_TYPE_AOS __global const
        #define MEMORY_TYPE_SOA __constant
    #endif // USE_SOA
#endif // USE_CONSTANT_MEMORY

#if VECTOR_SIZE == 1
    typedef float vector;
#elif VECTOR_SIZE == 2
    typedef float2 vector;
#elif VECTOR_SIZE == 4
    typedef float4 vector;
#elif VECTOR_SIZE == 8
    typedef float8 vector;
#elif VECTOR_SIZE == 16
    typedef float16 vector;
#endif // VECTOR_SIZE

#if VECTOR_SIZE > 1
__kernel __attribute__((vec_type_hint(vector))) 
#endif
__kernel void directCoulombSum(MEMORY_TYPE_AOS float4* atomInfo, MEMORY_TYPE_SOA vector* atomInfoX, MEMORY_TYPE_SOA vector* atomInfoY,
    MEMORY_TYPE_SOA vector* atomInfoZ, MEMORY_TYPE_SOA vector* atomInfoW, int numberOfAtoms, float gridSpacing, __global float* energyGrid)
{
    int xIndex = get_global_id(0);
    int yIndex = get_global_id(1);
    int zIndex = get_global_id(2);
        
    int sliceOffset = get_global_size(1) * get_global_size(0);
	int outIndex = sliceOffset*Z_ITERATIONS*zIndex + get_global_size(0)*yIndex + xIndex;

    float coordX = gridSpacing * xIndex;
    float coordY = gridSpacing * yIndex;
    float coordZ = gridSpacing * zIndex*(float)Z_ITERATIONS;

    vector energyValue[Z_ITERATIONS];
    for (int i = 0; i < Z_ITERATIONS; i++) 
        energyValue[i] = 0.0f;

    for (int i = 0; i < numberOfAtoms / VECTOR_SIZE; i++)
    {
        #if USE_SOA == 1
        vector dX = coordX - atomInfoX[i];
        vector dY = coordY - atomInfoY[i];
        vector dZ = coordZ - atomInfoZ[i];
        vector w = atomInfoW[i];
        #else
        vector dX = coordX - atomInfo[i].x;
        vector dY = coordY - atomInfo[i].y;
        vector dZ = coordZ - atomInfo[i].z;
        vector w = atomInfo[i].w;
        #endif /* USE_SOA */
        #if INNER_UNROLL_FACTOR > 0
        #pragma unroll INNER_UNROLL_FACTOR
        #endif
        for (int j = 0; j < Z_ITERATIONS; j++) {
            vector partialResult = half_rsqrt(dX * dX + dY * dY + dZ*dZ);
            energyValue[j] += w * partialResult;
            dZ += gridSpacing;
        }
    }

    for (int i = 0; i < Z_ITERATIONS; i++)
        #if VECTOR_SIZE == 1
        energyGrid[outIndex + sliceOffset*i] += energyValue[i];
        #elif VECTOR_SIZE == 2
        energyGrid[outIndex + sliceOffset*i] += energyValue[i].x + energyValue[i].y;
        #elif VECTOR_SIZE == 4
        energyGrid[outIndex + sliceOffset*i] += energyValue[i].x + energyValue[i].y + energyValue[i].z + energyValue[i].w;
        #elif VECTOR_SIZE == 8
        energyGrid[outIndex + sliceOffset*i] += energyValue[i].s0 + energyValue[i].s1 + energyValue[i].s2 + energyValue[i].s3 + energyValue[i].s4 + energyValue[i].s5 + energyValue[i].s6 + energyValue[i].s7;
        #elif VECTOR_SIZE == 16
        energyGrid[outIndex + sliceOffset*i] += energyValue[i].s0 + energyValue[i].s1 + energyValue[i].s2 + energyValue[i].s3 + energyValue[i].s4 + energyValue[i].s5 + energyValue[i].s6 + energyValue[i].s7 + energyValue[i].s8 + energyValue[i].s9 + energyValue[i].sa + energyValue[i].sb + energyValue[i].sc + energyValue[i].sd + energyValue[i].se + energyValue[i].sf;
        #endif
}
