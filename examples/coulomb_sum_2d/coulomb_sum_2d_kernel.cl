#if USE_CONSTANT_MEMORY == 0
    #define MEMORY_TYPE_AOS __global const
    #define MEMORY_TYPE_SOA __global const
#elif USE_CONSTANT_MEMORY == 1
    #if USE_SOA == 0
        #define MEMORY_TYPE_AOS __constant
        #define MEMORY_TYPE_SOA __global const
    #elif USE_SOA > 0
        #define MEMORY_TYPE_AOS __global const
        #define MEMORY_TYPE_SOA __constant
    #endif // USE_SOA
#endif // USE_CONSTANT_MEMORY

#if USE_SOA == 0
    #define SOA_UNROLL 0
#elif USE_SOA > 0
    #define SOA_UNROLL 1
#endif // USE_SOA

#if VECTOR_TYPE == 1
    typedef float vector;
#elif VECTOR_TYPE == 2
    typedef float2 vector;
#elif VECTOR_TYPE == 4
    typedef float4 vector;
#elif VECTOR_TYPE == 8
    typedef float8 vector;
#elif VECTOR_TYPE == 16
    typedef float16 vector;
#endif // VECTOR_TYPE

#if VECTOR_SIZE > 1
__kernel __attribute__((vec_type_hint(vector)))
#endif
__kernel void directCoulombSum(MEMORY_TYPE_AOS vector* atomInfo, MEMORY_TYPE_SOA vector* atomInfoX, MEMORY_TYPE_SOA vector* atomInfoY,
    MEMORY_TYPE_SOA vector* atomInfoZ, MEMORY_TYPE_SOA vector* atomInfoW, int numberOfAtoms, float gridSpacing, __global float* energyGrid)
{
    int xIndex = get_global_id(0) * OUTER_UNROLL_FACTOR;
    int yIndex = get_global_id(1);
    
    int outIndex = get_global_size(0) * yIndex * OUTER_UNROLL_FACTOR + xIndex;

    float coordX = gridSpacing * xIndex;
    #if OUTER_UNROLL_FACTOR > 1
    float coordX2 = gridSpacing * (xIndex + 1);
    #endif // OUTER_UNROLL_FACTOR > 1
    #if OUTER_UNROLL_FACTOR > 2
    float coordX3 = gridSpacing * (xIndex + 2);
    float coordX4 = gridSpacing * (xIndex + 3);
    #endif // OUTER_UNROLL_FACTOR > 2
    #if OUTER_UNROLL_FACTOR > 4
    float coordX5 = gridSpacing * (xIndex + 4);
    float coordX6 = gridSpacing * (xIndex + 5);
    float coordX7 = gridSpacing * (xIndex + 6);
    float coordX8 = gridSpacing * (xIndex + 7);
    #endif // OUTER_UNROLL_FACTOR > 4

    float coordY = gridSpacing * yIndex;

    #if USE_SOA < 2
    float energyValue = 0.0f;
    #if OUTER_UNROLL_FACTOR > 1
    float energyValue2 = 0.0f;
    #endif // OUTER_UNROLL_FACTOR > 1
    #if OUTER_UNROLL_FACTOR > 2
    float energyValue3 = 0.0f;
    float energyValue4 = 0.0f;
    #endif // OUTER_UNROLL_FACTOR > 2
    #if OUTER_UNROLL_FACTOR > 4
    float energyValue5 = 0.0f;
    float energyValue6 = 0.0f;
    float energyValue7 = 0.0f;
    float energyValue8 = 0.0f;
    #endif // OUTER_UNROLL_FACTOR > 4

    #elif USE_SOA == 2
    #if VECTOR_TYPE == 2
    float2 energyValueVect = ( 0.0f, 0.0f );
    float2 energyValueVect2 = ( 0.0f, 0.0f );
    float2 energyValueVect3 = ( 0.0f, 0.0f );
    float2 energyValueVect4 = ( 0.0f, 0.0f );
    float2 energyValueVect5 = ( 0.0f, 0.0f );
    float2 energyValueVect6 = ( 0.0f, 0.0f );
    float2 energyValueVect7 = ( 0.0f, 0.0f );
    float2 energyValueVect8 = ( 0.0f, 0.0f );
    #elif VECTOR_TYPE == 4
    float4 energyValueVect = ( 0.0f, 0.0f, 0.0f, 0.0f );
    float4 energyValueVect2 = ( 0.0f, 0.0f, 0.0f, 0.0f );
    float4 energyValueVect3 = ( 0.0f, 0.0f, 0.0f, 0.0f );
    float4 energyValueVect4 = ( 0.0f, 0.0f, 0.0f, 0.0f );
    float4 energyValueVect5 = ( 0.0f, 0.0f, 0.0f, 0.0f );
    float4 energyValueVect6 = ( 0.0f, 0.0f, 0.0f, 0.0f );
    float4 energyValueVect7 = ( 0.0f, 0.0f, 0.0f, 0.0f );
    float4 energyValueVect8 = ( 0.0f, 0.0f, 0.0f, 0.0f );
    #elif VECTOR_TYPE == 8
    float8 energyValueVect = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float8 energyValueVect2 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float8 energyValueVect3 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float8 energyValueVect4 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float8 energyValueVect5 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float8 energyValueVect6 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float8 energyValueVect7 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float8 energyValueVect8 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    #elif VECTOR_TYPE == 16
    float16 energyValueVect = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float16 energyValueVect2 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float16 energyValueVect3 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float16 energyValueVect4 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float16 energyValueVect5 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float16 energyValueVect6 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float16 energyValueVect7 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    float16 energyValueVect8 = ( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    #endif // VECTOR_TYPE
    #endif // USE_SOA

    #if USE_SOA < 2
    #if VECTOR_TYPE == 1
    #if INNER_UNROLL_FACTOR > 0
    #pragma unroll INNER_UNROLL_FACTOR
    #endif
    for (int i = 0; i < numberOfAtoms; i++)
    {
        #if USE_SOA == 0
        float distanceX = coordX - atomInfo[(4 * i)];
        float distanceY = coordY - atomInfo[(4 * i) + 1];
        float partialResult = half_rsqrt(distanceX * distanceX + distanceY * distanceY + atomInfo[(4 * i) + 2]);
        energyValue += atomInfo[(4 * i) + 3] * partialResult;


        #if OUTER_UNROLL_FACTOR > 1
        float distanceX2 = coordX2 - atomInfo[(4 * i)];
        float partialResult2 = half_rsqrt(distanceX2 * distanceX2 + distanceY * distanceY + atomInfo[(4 * i) + 2]);
        energyValue2 += atomInfo[(4 * i) + 3] * partialResult2;
        #endif // OUTER_UNROLL_FACTOR > 1


        #if OUTER_UNROLL_FACTOR > 2
        float distanceX3 = coordX3 - atomInfo[(4 * i)];
        float partialResult3 = half_rsqrt(distanceX3 * distanceX3 + distanceY * distanceY + atomInfo[(4 * i) + 2]);
        energyValue3 += atomInfo[(4 * i) + 3] * partialResult3;

        float distanceX4 = coordX4 - atomInfo[(4 * i)];
        float partialResult4 = half_rsqrt(distanceX4 * distanceX4 + distanceY * distanceY + atomInfo[(4 * i) + 2]);
        energyValue4 += atomInfo[(4 * i) + 3] * partialResult4;
        #endif // OUTER_UNROLL_FACTOR > 2


        #if OUTER_UNROLL_FACTOR > 4
        float distanceX5 = coordX5 - atomInfo[(4 * i)];
        float partialResult5 = half_rsqrt(distanceX5 * distanceX5 + distanceY * distanceY + atomInfo[(4 * i) + 2]);
        energyValue5 += atomInfo[(4 * i) + 3] * partialResult5;

        float distanceX6 = coordX6 - atomInfo[(4 * i)];
        float partialResult6 = half_rsqrt(distanceX6 * distanceX6 + distanceY * distanceY + atomInfo[(4 * i) + 2]);
        energyValue6 += atomInfo[(4 * i) + 3] * partialResult6;

        float distanceX7 = coordX7 - atomInfo[(4 * i)];
        float partialResult7 = half_rsqrt(distanceX7 * distanceX7 + distanceY * distanceY + atomInfo[(4 * i) + 2]);
        energyValue7 += atomInfo[(4 * i) + 3] * partialResult7;

        float distanceX8 = coordX8 - atomInfo[(4 * i)];
        float partialResult8 = half_rsqrt(distanceX8 * distanceX8 + distanceY * distanceY + atomInfo[(4 * i) + 2]);
        energyValue8 += atomInfo[(4 * i) + 3] * partialResult8;
        #endif // OUTER_UNROLL_FACTOR > 4


        #elif USE_SOA == 1
        float distanceX = coordX - atomInfoX[i];
        float distanceY = coordY - atomInfoY[i];
        float partialResult = half_rsqrt(distanceX * distanceX + distanceY * distanceY + atomInfoZ[i]);
        energyValue += atomInfoW[i] * partialResult;


        #if OUTER_UNROLL_FACTOR > 1
        float distanceX2 = coordX2 - atomInfoX[i];
        float partialResult2 = half_rsqrt(distanceX2 * distanceX2 + distanceY * distanceY + atomInfoZ[i]);
        energyValue2 += atomInfoW[i] * partialResult2;
        #endif // OUTER_UNROLL_FACTOR > 1


        #if OUTER_UNROLL_FACTOR > 2
        float distanceX3 = coordX3 - atomInfoX[i];
        float partialResult3 = half_rsqrt(distanceX3 * distanceX3 + distanceY * distanceY + atomInfoZ[i]);
        energyValue3 += atomInfoW[i] * partialResult3;

        float distanceX4 = coordX4 - atomInfoX[i];
        float partialResult4 = half_rsqrt(distanceX4 * distanceX4 + distanceY * distanceY + atomInfoZ[i]);
        energyValue4 += atomInfoW[i] * partialResult4;
        #endif // OUTER_UNROLL_FACTOR > 2


        #if OUTER_UNROLL_FACTOR > 4
        float distanceX5 = coordX5 - atomInfoX[i];
        float partialResult5 = half_rsqrt(distanceX5 * distanceX5 + distanceY * distanceY + atomInfoZ[i]);
        energyValue5 += atomInfoW[i] * partialResult5;

        float distanceX6 = coordX6 - atomInfoX[i];
        float partialResult6 = half_rsqrt(distanceX6 * distanceX6 + distanceY * distanceY + atomInfoZ[i]);
        energyValue6 += atomInfoW[i] * partialResult6;

        float distanceX7 = coordX7 - atomInfoX[i];
        float partialResult7 = half_rsqrt(distanceX7 * distanceX7 + distanceY * distanceY + atomInfoZ[i]);
        energyValue7 += atomInfoW[i] * partialResult7;

        float distanceX8 = coordX8 - atomInfoX[i];
        float partialResult8 = half_rsqrt(distanceX8 * distanceX8 + distanceY * distanceY + atomInfoZ[i]);
        energyValue8 += atomInfoW[i] * partialResult8;
        #endif // OUTER_UNROLL_FACTOR > 4
        #endif // USE_SOA
    }

    #elif VECTOR_TYPE == 2
    #if INNER_UNROLL_FACTOR > 0
    #pragma unroll INNER_UNROLL_FACTOR
    #endif
    for (int i = 0; i < numberOfAtoms / (1 + SOA_UNROLL); i++)
    {
        #if USE_SOA == 0
        float distanceX = coordX - atomInfo[(2 * i)].x;
        float distanceY = coordY - atomInfo[(2 * i)].y;
        float partialResult = half_rsqrt(distanceX * distanceX + distanceY * distanceY + atomInfo[(2 * i) + 1].x);
        energyValue += atomInfo[(2 * i) + 1].y * partialResult;


        #if OUTER_UNROLL_FACTOR > 1
        float distanceX2 = coordX2 - atomInfo[(2 * i)].x;
        float partialResult2 = half_rsqrt(distanceX2 * distanceX2 + distanceY * distanceY + atomInfo[(2 * i) + 1].x);
        energyValue2 += atomInfo[(2 * i) + 1].y * partialResult2;
        #endif // OUTER_UNROLL_FACTOR > 1


        #if OUTER_UNROLL_FACTOR > 2
        float distanceX3 = coordX3 - atomInfo[(2 * i)].x;
        float partialResult3 = half_rsqrt(distanceX3 * distanceX3 + distanceY * distanceY + atomInfo[(2 * i) + 1].x);
        energyValue3 += atomInfo[(2 * i) + 1].y * partialResult3;

        float distanceX4 = coordX4 - atomInfo[(2 * i)].x;
        float partialResult4 = half_rsqrt(distanceX4 * distanceX4 + distanceY * distanceY + atomInfo[(2 * i) + 1].x);
        energyValue4 += atomInfo[(2 * i) + 1].y * partialResult4;
        #endif // OUTER_UNROLL_FACTOR > 2


        #if OUTER_UNROLL_FACTOR > 4
        float distanceX5 = coordX5 - atomInfo[(2 * i)].x;
        float partialResult5 = half_rsqrt(distanceX5 * distanceX5 + distanceY * distanceY + atomInfo[(2 * i) + 1].x);
        energyValue5 += atomInfo[(2 * i) + 1].y * partialResult5;

        float distanceX6 = coordX6 - atomInfo[(2 * i)].x;
        float partialResult6 = half_rsqrt(distanceX6 * distanceX6 + distanceY * distanceY + atomInfo[(2 * i) + 1].x);
        energyValue6 += atomInfo[(2 * i) + 1].y * partialResult6;

        float distanceX7 = coordX7 - atomInfo[(2 * i)].x;
        float partialResult7 = half_rsqrt(distanceX7 * distanceX7 + distanceY * distanceY + atomInfo[(2 * i) + 1].x);
        energyValue7 += atomInfo[(2 * i) + 1].y * partialResult7;

        float distanceX8 = coordX8 - atomInfo[(2 * i)].x;
        float partialResult8 = half_rsqrt(distanceX8 * distanceX8 + distanceY * distanceY + atomInfo[(2 * i) + 1].x);
        energyValue8 += atomInfo[(2 * i) + 1].y * partialResult8;
        #endif // OUTER_UNROLL_FACTOR > 4


        #elif USE_SOA == 1
        float distanceX = coordX - atomInfoX[i].x;
        float distanceY = coordY - atomInfoY[i].x;
        float partialResult = half_rsqrt(distanceX * distanceX + distanceY * distanceY + atomInfoZ[i].x);
        energyValue += atomInfoW[i].x * partialResult;

        float distanceXy = coordX - atomInfoX[i].y;
        float distanceYy = coordY - atomInfoY[i].y;
        float partialResulty = half_rsqrt(distanceXy * distanceXy + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue += atomInfoW[i].y * partialResulty;


        #if OUTER_UNROLL_FACTOR > 1
        float distanceX2 = coordX2 - atomInfoX[i].x;
        float partialResult2 = half_rsqrt(distanceX2 * distanceX2 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue2 += atomInfoW[i].x * partialResult2;

        float distanceXy2 = coordX2 - atomInfoX[i].y;
        float partialResulty2 = half_rsqrt(distanceXy2 * distanceXy2 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue2 += atomInfoW[i].y * partialResulty2;
        #endif // OUTER_UNROLL_FACTOR > 1


        #if OUTER_UNROLL_FACTOR > 2
        float distanceX3 = coordX3 - atomInfoX[i].x;
        float partialResult3 = half_rsqrt(distanceX3 * distanceX3 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue3 += atomInfoW[i].x * partialResult3;

        float distanceXy3 = coordX3 - atomInfoX[i].y;
        float partialResulty3 = half_rsqrt(distanceXy3 * distanceXy3 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue3 += atomInfoW[i].y * partialResulty3;

        float distanceX4 = coordX4 - atomInfoX[i].x;
        float partialResult4 = half_rsqrt(distanceX4 * distanceX4 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue4 += atomInfoW[i].x * partialResult4;

        float distanceXy4 = coordX4 - atomInfoX[i].y;
        float partialResulty4 = half_rsqrt(distanceXy4 * distanceXy4 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue4 += atomInfoW[i].y * partialResulty4;
        #endif // OUTER_UNROLL_FACTOR > 2


        #if OUTER_UNROLL_FACTOR > 4
        float distanceX5 = coordX5 - atomInfoX[i].x;
        float partialResult5 = half_rsqrt(distanceX5 * distanceX5 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue5 += atomInfoW[i].x * partialResult5;

        float distanceXy5 = coordX5 - atomInfoX[i].y;
        float partialResulty5 = half_rsqrt(distanceXy5 * distanceXy5 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue5 += atomInfoW[i].y * partialResulty5;

        float distanceX6 = coordX6 - atomInfoX[i].x;
        float partialResult6 = half_rsqrt(distanceX6 * distanceX6 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue6 += atomInfoW[i].x * partialResult6;

        float distanceXy6 = coordX6 - atomInfoX[i].y;
        float partialResulty6 = half_rsqrt(distanceXy6 * distanceXy6 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue6 += atomInfoW[i].y * partialResulty6;

        float distanceX7 = coordX7 - atomInfoX[i].x;
        float partialResult7 = half_rsqrt(distanceX7 * distanceX7 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue7 += atomInfoW[i].x * partialResult7;

        float distanceXy7 = coordX7 - atomInfoX[i].y;
        float partialResulty7 = half_rsqrt(distanceXy7 * distanceXy7 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue7 += atomInfoW[i].y * partialResulty7;

        float distanceX8 = coordX8 - atomInfoX[i].x;
        float partialResult8 = half_rsqrt(distanceX8 * distanceX8 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue8 += atomInfoW[i].x * partialResult8;

        float distanceXy8 = coordX8 - atomInfoX[i].y;
        float partialResulty8 = half_rsqrt(distanceXy8 * distanceXy8 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue8 += atomInfoW[i].y * partialResulty8;
        #endif // OUTER_UNROLL_FACTOR > 4
        #endif // USE_SOA
    }

    #elif VECTOR_TYPE == 4
    #if INNER_UNROLL_FACTOR > 0
    #pragma unroll INNER_UNROLL_FACTOR
    #endif
    for (int i = 0; i < numberOfAtoms / (1 + SOA_UNROLL * 3); i++)
    {
        #if USE_SOA == 0
        float distanceX = coordX - atomInfo[i].x;
        float distanceY = coordY - atomInfo[i].y;
        float partialResult = half_rsqrt(distanceX * distanceX + distanceY * distanceY + atomInfo[i].z);
        energyValue += atomInfo[i].w * partialResult;


        #if OUTER_UNROLL_FACTOR > 1
        float distanceX2 = coordX2 - atomInfo[i].x;
        float partialResult2 = half_rsqrt(distanceX2 * distanceX2 + distanceY * distanceY + atomInfo[i].z);
        energyValue2 += atomInfo[i].w * partialResult2;
        #endif // OUTER_UNROLL_FACTOR > 1


        #if OUTER_UNROLL_FACTOR > 2
        float distanceX3 = coordX3 - atomInfo[i].x;
        float partialResult3 = half_rsqrt(distanceX3 * distanceX3 + distanceY * distanceY + atomInfo[i].z);
        energyValue3 += atomInfo[i].w * partialResult3;

        float distanceX4 = coordX4 - atomInfo[i].x;
        float partialResult4 = half_rsqrt(distanceX4 * distanceX4 + distanceY * distanceY + atomInfo[i].z);
        energyValue4 += atomInfo[i].w * partialResult4;
        #endif // OUTER_UNROLL_FACTOR > 2


        #if OUTER_UNROLL_FACTOR > 4
        float distanceX5 = coordX5 - atomInfo[i].x;
        float partialResult5 = half_rsqrt(distanceX5 * distanceX5 + distanceY * distanceY + atomInfo[i].z);
        energyValue5 += atomInfo[i].w * partialResult5;

        float distanceX6 = coordX6 - atomInfo[i].x;
        float partialResult6 = half_rsqrt(distanceX6 * distanceX6 + distanceY * distanceY + atomInfo[i].z);
        energyValue6 += atomInfo[i].w * partialResult6;

        float distanceX7 = coordX7 - atomInfo[i].x;
        float partialResult7 = half_rsqrt(distanceX7 * distanceX7 + distanceY * distanceY + atomInfo[i].z);
        energyValue7 += atomInfo[i].w * partialResult7;

        float distanceX8 = coordX8 - atomInfo[i].x;
        float partialResult8 = half_rsqrt(distanceX8 * distanceX8 + distanceY * distanceY + atomInfo[i].z);
        energyValue8 += atomInfo[i].w * partialResult8;
        #endif // OUTER_UNROLL_FACTOR > 4


        #elif USE_SOA == 1
        float distanceX = coordX - atomInfoX[i].x;
        float distanceY = coordY - atomInfoY[i].x;
        float partialResult = half_rsqrt(distanceX * distanceX + distanceY * distanceY + atomInfoZ[i].x);
        energyValue += atomInfoW[i].x * partialResult;

        float distanceXy = coordX - atomInfoX[i].y;
        float distanceYy = coordY - atomInfoY[i].y;
        float partialResulty = half_rsqrt(distanceXy * distanceXy + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue += atomInfoW[i].y * partialResulty;

        float distanceXz = coordX - atomInfoX[i].z;
        float distanceYz = coordY - atomInfoY[i].z;
        float partialResultz = half_rsqrt(distanceXz * distanceXz + distanceYz * distanceYz + atomInfoZ[i].z);
        energyValue += atomInfoW[i].z * partialResultz;

        float distanceXw = coordX - atomInfoX[i].w;
        float distanceYw = coordY - atomInfoY[i].w;
        float partialResultw = half_rsqrt(distanceXw * distanceXw + distanceYw * distanceYw + atomInfoZ[i].w);
        energyValue += atomInfoW[i].w * partialResultw;


        #if OUTER_UNROLL_FACTOR > 1
        float distanceX2 = coordX2 - atomInfoX[i].x;
        float partialResult2 = half_rsqrt(distanceX2 * distanceX2 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue2 += atomInfoW[i].x * partialResult2;

        float distanceXy2 = coordX2 - atomInfoX[i].y;
        float partialResulty2 = half_rsqrt(distanceXy2 * distanceXy2 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue2 += atomInfoW[i].y * partialResulty2;

        float distanceXz2 = coordX2 - atomInfoX[i].z;
        float partialResultz2 = half_rsqrt(distanceXz2 * distanceXz2 + distanceYz * distanceYz + atomInfoZ[i].z);
        energyValue2 += atomInfoW[i].z * partialResultz2;

        float distanceXw2 = coordX2 - atomInfoX[i].w;
        float partialResultw2 = half_rsqrt(distanceXw2 * distanceXw2 + distanceYw * distanceYw + atomInfoZ[i].w);
        energyValue2 += atomInfoW[i].w * partialResultw2;
        #endif // OUTER_UNROLL_FACTOR > 1


        #if OUTER_UNROLL_FACTOR > 2
        float distanceX3 = coordX3 - atomInfoX[i].x;
        float partialResult3 = half_rsqrt(distanceX3 * distanceX3 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue3 += atomInfoW[i].x * partialResult3;

        float distanceXy3 = coordX3 - atomInfoX[i].y;
        float partialResulty3 = half_rsqrt(distanceXy3 * distanceXy3 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue3 += atomInfoW[i].y * partialResulty3;

        float distanceXz3 = coordX3 - atomInfoX[i].z;
        float partialResultz3 = half_rsqrt(distanceXz3 * distanceXz3 + distanceYz * distanceYz + atomInfoZ[i].z);
        energyValue3 += atomInfoW[i].z * partialResultz3;

        float distanceXw3 = coordX3 - atomInfoX[i].w;
        float partialResultw3 = half_rsqrt(distanceXw3 * distanceXw3 + distanceYw * distanceYw + atomInfoZ[i].w);
        energyValue3 += atomInfoW[i].w * partialResultw3;

        float distanceX4 = coordX4 - atomInfoX[i].x;
        float partialResult4 = half_rsqrt(distanceX4 * distanceX4 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue4 += atomInfoW[i].x * partialResult4;

        float distanceXy4 = coordX4 - atomInfoX[i].y;
        float partialResulty4 = half_rsqrt(distanceXy4 * distanceXy4 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue4 += atomInfoW[i].y * partialResulty4;

        float distanceXz4 = coordX4 - atomInfoX[i].z;
        float partialResultz4 = half_rsqrt(distanceXz4 * distanceXz4 + distanceYz * distanceYz + atomInfoZ[i].z);
        energyValue4 += atomInfoW[i].z * partialResultz4;

        float distanceXw4 = coordX4 - atomInfoX[i].w;
        float partialResultw4 = half_rsqrt(distanceXw4 * distanceXw4 + distanceYw * distanceYw + atomInfoZ[i].w);
        energyValue4 += atomInfoW[i].w * partialResultw4;
        #endif // OUTER_UNROLL_FACTOR > 2


        #if OUTER_UNROLL_FACTOR > 4
        float distanceX5 = coordX5 - atomInfoX[i].x;
        float partialResult5 = half_rsqrt(distanceX5 * distanceX5 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue5 += atomInfoW[i].x * partialResult5;

        float distanceXy5 = coordX5 - atomInfoX[i].y;
        float partialResulty5 = half_rsqrt(distanceXy5 * distanceXy5 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue5 += atomInfoW[i].y * partialResulty5;

        float distanceXz5 = coordX5 - atomInfoX[i].z;
        float partialResultz5 = half_rsqrt(distanceXz5 * distanceXz5 + distanceYz * distanceYz + atomInfoZ[i].z);
        energyValue5 += atomInfoW[i].z * partialResultz5;

        float distanceXw5 = coordX5 - atomInfoX[i].w;
        float partialResultw5 = half_rsqrt(distanceXw5 * distanceXw5 + distanceYw * distanceYw + atomInfoZ[i].w);
        energyValue5 += atomInfoW[i].w * partialResultw5;

        float distanceX6 = coordX6 - atomInfoX[i].x;
        float partialResult6 = half_rsqrt(distanceX6 * distanceX6 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue6 += atomInfoW[i].x * partialResult6;

        float distanceXy6 = coordX6 - atomInfoX[i].y;
        float partialResulty6 = half_rsqrt(distanceXy6 * distanceXy6 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue6 += atomInfoW[i].y * partialResulty6;

        float distanceXz6 = coordX6 - atomInfoX[i].z;
        float partialResultz6 = half_rsqrt(distanceXz6 * distanceXz6 + distanceYz * distanceYz + atomInfoZ[i].z);
        energyValue6 += atomInfoW[i].z * partialResultz6;

        float distanceXw6 = coordX6 - atomInfoX[i].w;
        float partialResultw6 = half_rsqrt(distanceXw6 * distanceXw6 + distanceYw * distanceYw + atomInfoZ[i].w);
        energyValue6 += atomInfoW[i].w * partialResultw6;

        float distanceX7 = coordX7 - atomInfoX[i].x;
        float partialResult7 = half_rsqrt(distanceX7 * distanceX7 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue7 += atomInfoW[i].x * partialResult7;

        float distanceXy7 = coordX7 - atomInfoX[i].y;
        float partialResulty7 = half_rsqrt(distanceXy7 * distanceXy7 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue7 += atomInfoW[i].y * partialResulty7;

        float distanceXz7 = coordX7 - atomInfoX[i].z;
        float partialResultz7 = half_rsqrt(distanceXz7 * distanceXz7 + distanceYz * distanceYz + atomInfoZ[i].z);
        energyValue7 += atomInfoW[i].z * partialResultz7;

        float distanceXw7 = coordX7 - atomInfoX[i].w;
        float partialResultw7 = half_rsqrt(distanceXw7 * distanceXw7 + distanceYw * distanceYw + atomInfoZ[i].w);
        energyValue7 += atomInfoW[i].w * partialResultw7;

        float distanceX8 = coordX8 - atomInfoX[i].x;
        float partialResult8 = half_rsqrt(distanceX8 * distanceX8 + distanceY * distanceY + atomInfoZ[i].x);
        energyValue8 += atomInfoW[i].x * partialResult8;

        float distanceXy8 = coordX8 - atomInfoX[i].y;
        float partialResulty8 = half_rsqrt(distanceXy8 * distanceXy8 + distanceYy * distanceYy + atomInfoZ[i].y);
        energyValue8 += atomInfoW[i].y * partialResulty8;

        float distanceXz8 = coordX8 - atomInfoX[i].z;
        float partialResultz8 = half_rsqrt(distanceXz8 * distanceXz8 + distanceYz * distanceYz + atomInfoZ[i].z);
        energyValue8 += atomInfoW[i].z * partialResultz8;

        float distanceXw8 = coordX8 - atomInfoX[i].w;
        float partialResultw8 = half_rsqrt(distanceXw8 * distanceXw8 + distanceYw * distanceYw + atomInfoZ[i].w);
        energyValue8 += atomInfoW[i].w * partialResultw8;
        #endif // OUTER_UNROLL_FACTOR > 4
        #endif // USE_SOA
    }

    #elif VECTOR_TYPE == 8
    #if INNER_UNROLL_FACTOR > 0
    #pragma unroll INNER_UNROLL_FACTOR
    #endif
    for (int i = 0; i < numberOfAtoms / (2 - SOA_UNROLL) / (1 + SOA_UNROLL * 7); i++)
    {
        #if USE_SOA == 0
        float distanceX = coordX - atomInfo[i].s0;
        float distanceY = coordY - atomInfo[i].s1;
        float partialResult = half_rsqrt(distanceX * distanceX + distanceY * distanceY + atomInfo[i].s2);
        energyValue += atomInfo[i].s3 * partialResult;

        float distanceX2 = coordX - atomInfo[i].s4;
        float distanceY2 = coordY - atomInfo[i].s5;
        float partialResult2 = half_rsqrt(distanceX2 * distanceX2 + distanceY2 * distanceY2 + atomInfo[i].s6);
        energyValue += atomInfo[i].s7 * partialResult2;


        #if OUTER_UNROLL_FACTOR > 1
        float distanceX21 = coordX2 - atomInfo[i].s0;
        float partialResult21 = half_rsqrt(distanceX21 * distanceX21 + distanceY * distanceY + atomInfo[i].s2);
        energyValue2 += atomInfo[i].s3 * partialResult21;

        float distanceX22 = coordX2 - atomInfo[i].s4;
        float partialResult22 = half_rsqrt(distanceX22 * distanceX22 + distanceY2 * distanceY2 + atomInfo[i].s6);
        energyValue2 += atomInfo[i].s7 * partialResult22;
        #endif // OUTER_UNROLL_FACTOR > 1


        #if OUTER_UNROLL_FACTOR > 2
        float distanceX31 = coordX3 - atomInfo[i].s0;
        float partialResult31 = half_rsqrt(distanceX31 * distanceX31 + distanceY * distanceY + atomInfo[i].s2);
        energyValue3 += atomInfo[i].s3 * partialResult31;

        float distanceX32 = coordX3 - atomInfo[i].s4;
        float partialResult32 = half_rsqrt(distanceX32 * distanceX32 + distanceY2 * distanceY2 + atomInfo[i].s6);
        energyValue3 += atomInfo[i].s7 * partialResult32;

        float distanceX41 = coordX4 - atomInfo[i].s0;
        float partialResult41 = half_rsqrt(distanceX41 * distanceX41 + distanceY * distanceY + atomInfo[i].s2);
        energyValue4 += atomInfo[i].s3 * partialResult41;

        float distanceX42 = coordX4 - atomInfo[i].s4;
        float partialResult42 = half_rsqrt(distanceX42 * distanceX42 + distanceY2 * distanceY2 + atomInfo[i].s6);
        energyValue4 += atomInfo[i].s7 * partialResult42;
        #endif // OUTER_UNROLL_FACTOR > 2


        #if OUTER_UNROLL_FACTOR > 4
        float distanceX51 = coordX5 - atomInfo[i].s0;
        float partialResult51 = half_rsqrt(distanceX51 * distanceX51 + distanceY * distanceY + atomInfo[i].s2);
        energyValue5 += atomInfo[i].s3 * partialResult51;

        float distanceX52 = coordX5 - atomInfo[i].s4;
        float partialResult52 = half_rsqrt(distanceX52 * distanceX52 + distanceY2 * distanceY2 + atomInfo[i].s6);
        energyValue5 += atomInfo[i].s7 * partialResult52;

        float distanceX61 = coordX6 - atomInfo[i].s0;
        float partialResult61 = half_rsqrt(distanceX61 * distanceX61 + distanceY * distanceY + atomInfo[i].s2);
        energyValue6 += atomInfo[i].s3 * partialResult61;

        float distanceX62 = coordX6 - atomInfo[i].s4;
        float partialResult62 = half_rsqrt(distanceX62 * distanceX62 + distanceY2 * distanceY2 + atomInfo[i].s6);
        energyValue6 += atomInfo[i].s7 * partialResult62;

        float distanceX71 = coordX7 - atomInfo[i].s0;
        float partialResult71 = half_rsqrt(distanceX71 * distanceX71 + distanceY * distanceY + atomInfo[i].s2);
        energyValue7 += atomInfo[i].s3 * partialResult71;

        float distanceX72 = coordX7 - atomInfo[i].s4;
        float partialResult72 = half_rsqrt(distanceX72 * distanceX72 + distanceY2 * distanceY2 + atomInfo[i].s6);
        energyValue7 += atomInfo[i].s7 * partialResult72;

        float distanceX81 = coordX8 - atomInfo[i].s0;
        float partialResult81 = half_rsqrt(distanceX81 * distanceX81 + distanceY * distanceY + atomInfo[i].s2);
        energyValue8 += atomInfo[i].s3 * partialResult81;

        float distanceX82 = coordX8 - atomInfo[i].s4;
        float partialResult82 = half_rsqrt(distanceX82 * distanceX82 + distanceY2 * distanceY2 + atomInfo[i].s6);
        energyValue8 += atomInfo[i].s7 * partialResult82;
        #endif // OUTER_UNROLL_FACTOR > 4


        #elif USE_SOA == 1
        float distanceXs0 = coordX - atomInfoX[i].s0;
        float distanceYs0 = coordY - atomInfoY[i].s0;
        float partialResults0 = half_rsqrt(distanceXs0 * distanceXs0 + distanceYs0 * distanceYs0 + atomInfoZ[i].s0);
        energyValue += atomInfoW[i].s0 * partialResults0;

        float distanceXs1 = coordX - atomInfoX[i].s1;
        float distanceYs1 = coordY - atomInfoY[i].s1;
        float partialResults1 = half_rsqrt(distanceXs1 * distanceXs1 + distanceYs1 * distanceYs1 + atomInfoZ[i].s1);
        energyValue += atomInfoW[i].s1 * partialResults1;

        float distanceXs2 = coordX - atomInfoX[i].s2;
        float distanceYs2 = coordY - atomInfoY[i].s2;
        float partialResults2 = half_rsqrt(distanceXs2 * distanceXs2 + distanceYs2 * distanceYs2 + atomInfoZ[i].s2);
        energyValue += atomInfoW[i].s2 * partialResults2;

        float distanceXs3 = coordX - atomInfoX[i].s3;
        float distanceYs3 = coordY - atomInfoY[i].s3;
        float partialResults3 = half_rsqrt(distanceXs3 * distanceXs3 + distanceYs3 * distanceYs3 + atomInfoZ[i].s3);
        energyValue += atomInfoW[i].s3 * partialResults3;

        float distanceXs4 = coordX - atomInfoX[i].s4;
        float distanceYs4 = coordY - atomInfoY[i].s4;
        float partialResults4 = half_rsqrt(distanceXs4 * distanceXs4 + distanceYs4 * distanceYs4 + atomInfoZ[i].s4);
        energyValue += atomInfoW[i].s4 * partialResults4;

        float distanceXs5 = coordX - atomInfoX[i].s5;
        float distanceYs5 = coordY - atomInfoY[i].s5;
        float partialResults5 = half_rsqrt(distanceXs5 * distanceXs5 + distanceYs5 * distanceYs5 + atomInfoZ[i].s5);
        energyValue += atomInfoW[i].s5 * partialResults5;

        float distanceXs6 = coordX - atomInfoX[i].s6;
        float distanceYs6 = coordY - atomInfoY[i].s6;
        float partialResults6 = half_rsqrt(distanceXs6 * distanceXs6 + distanceYs6 * distanceYs6 + atomInfoZ[i].s6);
        energyValue += atomInfoW[i].s6 * partialResults6;

        float distanceXs7 = coordX - atomInfoX[i].s7;
        float distanceYs7 = coordY - atomInfoY[i].s7;
        float partialResults7 = half_rsqrt(distanceXs7 * distanceXs7 + distanceYs7 * distanceYs7 + atomInfoZ[i].s7);
        energyValue += atomInfoW[i].s7 * partialResults7;


        #if OUTER_UNROLL_FACTOR > 1
        float distanceXs02 = coordX2 - atomInfoX[i].s0;
        float partialResults02 = half_rsqrt(distanceXs02 * distanceXs02 + distanceYs0 * distanceYs0 + atomInfoZ[i].s0);
        energyValue2 += atomInfoW[i].s0 * partialResults02;

        float distanceXs12 = coordX2 - atomInfoX[i].s1;
        float partialResults12 = half_rsqrt(distanceXs12 * distanceXs12 + distanceYs1 * distanceYs1 + atomInfoZ[i].s1);
        energyValue2 += atomInfoW[i].s1 * partialResults12;

        float distanceXs22 = coordX2 - atomInfoX[i].s2;
        float partialResults22 = half_rsqrt(distanceXs22 * distanceXs22 + distanceYs2 * distanceYs2 + atomInfoZ[i].s2);
        energyValue2 += atomInfoW[i].s2 * partialResults22;

        float distanceXs32 = coordX2 - atomInfoX[i].s3;
        float partialResults32 = half_rsqrt(distanceXs32 * distanceXs32 + distanceYs3 * distanceYs3 + atomInfoZ[i].s3);
        energyValue2 += atomInfoW[i].s3 * partialResults32;

        float distanceXs42 = coordX2 - atomInfoX[i].s4;
        float partialResults42 = half_rsqrt(distanceXs42 * distanceXs42 + distanceYs4 * distanceYs4 + atomInfoZ[i].s4);
        energyValue2 += atomInfoW[i].s4 * partialResults42;

        float distanceXs52 = coordX2 - atomInfoX[i].s5;
        float partialResults52 = half_rsqrt(distanceXs52 * distanceXs52 + distanceYs5 * distanceYs5 + atomInfoZ[i].s5);
        energyValue2 += atomInfoW[i].s5 * partialResults52;

        float distanceXs62 = coordX2 - atomInfoX[i].s6;
        float partialResults62 = half_rsqrt(distanceXs62 * distanceXs62 + distanceYs6 * distanceYs6 + atomInfoZ[i].s6);
        energyValue2 += atomInfoW[i].s6 * partialResults62;

        float distanceXs72 = coordX2 - atomInfoX[i].s7;
        float partialResults72 = half_rsqrt(distanceXs72 * distanceXs72 + distanceYs7 * distanceYs7 + atomInfoZ[i].s7);
        energyValue2 += atomInfoW[i].s7 * partialResults72;
        #endif // OUTER_UNROLL_FACTOR > 1


        #if OUTER_UNROLL_FACTOR > 2
        float distanceXs03 = coordX3 - atomInfoX[i].s0;
        float partialResults03 = half_rsqrt(distanceXs03 * distanceXs03 + distanceYs0 * distanceYs0 + atomInfoZ[i].s0);
        energyValue3 += atomInfoW[i].s0 * partialResults03;

        float distanceXs13 = coordX3 - atomInfoX[i].s1;
        float partialResults13 = half_rsqrt(distanceXs13 * distanceXs13 + distanceYs1 * distanceYs1 + atomInfoZ[i].s1);
        energyValue3 += atomInfoW[i].s1 * partialResults13;

        float distanceXs23 = coordX3 - atomInfoX[i].s2;
        float partialResults23 = half_rsqrt(distanceXs23 * distanceXs23 + distanceYs2 * distanceYs2 + atomInfoZ[i].s2);
        energyValue3 += atomInfoW[i].s2 * partialResults23;

        float distanceXs33 = coordX3 - atomInfoX[i].s3;
        float partialResults33 = half_rsqrt(distanceXs33 * distanceXs33 + distanceYs3 * distanceYs3 + atomInfoZ[i].s3);
        energyValue3 += atomInfoW[i].s3 * partialResults33;

        float distanceXs43 = coordX3 - atomInfoX[i].s4;
        float partialResults43 = half_rsqrt(distanceXs43 * distanceXs43 + distanceYs4 * distanceYs4 + atomInfoZ[i].s4);
        energyValue3 += atomInfoW[i].s4 * partialResults43;

        float distanceXs53 = coordX3 - atomInfoX[i].s5;
        float partialResults53 = half_rsqrt(distanceXs53 * distanceXs53 + distanceYs5 * distanceYs5 + atomInfoZ[i].s5);
        energyValue3 += atomInfoW[i].s5 * partialResults53;

        float distanceXs63 = coordX3 - atomInfoX[i].s6;
        float partialResults63 = half_rsqrt(distanceXs63 * distanceXs63 + distanceYs6 * distanceYs6 + atomInfoZ[i].s6);
        energyValue3 += atomInfoW[i].s6 * partialResults63;

        float distanceXs73 = coordX3 - atomInfoX[i].s7;
        float partialResults73 = half_rsqrt(distanceXs73 * distanceXs73 + distanceYs7 * distanceYs7 + atomInfoZ[i].s7);
        energyValue3 += atomInfoW[i].s7 * partialResults73;

        float distanceXs04 = coordX4 - atomInfoX[i].s0;
        float partialResults04 = half_rsqrt(distanceXs04 * distanceXs04 + distanceYs0 * distanceYs0 + atomInfoZ[i].s0);
        energyValue4 += atomInfoW[i].s0 * partialResults04;

        float distanceXs14 = coordX4 - atomInfoX[i].s1;
        float partialResults14 = half_rsqrt(distanceXs14 * distanceXs14 + distanceYs1 * distanceYs1 + atomInfoZ[i].s1);
        energyValue4 += atomInfoW[i].s1 * partialResults14;

        float distanceXs24 = coordX4 - atomInfoX[i].s2;
        float partialResults24 = half_rsqrt(distanceXs24 * distanceXs24 + distanceYs2 * distanceYs2 + atomInfoZ[i].s2);
        energyValue4 += atomInfoW[i].s2 * partialResults24;

        float distanceXs34 = coordX4 - atomInfoX[i].s3;
        float partialResults34 = half_rsqrt(distanceXs34 * distanceXs34 + distanceYs3 * distanceYs3 + atomInfoZ[i].s3);
        energyValue4 += atomInfoW[i].s3 * partialResults34;

        float distanceXs44 = coordX4 - atomInfoX[i].s4;
        float partialResults44 = half_rsqrt(distanceXs44 * distanceXs44 + distanceYs4 * distanceYs4 + atomInfoZ[i].s4);
        energyValue4 += atomInfoW[i].s4 * partialResults44;

        float distanceXs54 = coordX4 - atomInfoX[i].s5;
        float partialResults54 = half_rsqrt(distanceXs54 * distanceXs54 + distanceYs5 * distanceYs5 + atomInfoZ[i].s5);
        energyValue4 += atomInfoW[i].s5 * partialResults54;

        float distanceXs64 = coordX4 - atomInfoX[i].s6;
        float partialResults64 = half_rsqrt(distanceXs64 * distanceXs64 + distanceYs6 * distanceYs6 + atomInfoZ[i].s6);
        energyValue4 += atomInfoW[i].s6 * partialResults64;

        float distanceXs74 = coordX4 - atomInfoX[i].s7;
        float partialResults74 = half_rsqrt(distanceXs74 * distanceXs74 + distanceYs7 * distanceYs7 + atomInfoZ[i].s7);
        energyValue4 += atomInfoW[i].s7 * partialResults74;
        #endif // OUTER_UNROLL_FACTOR > 2


        #if OUTER_UNROLL_FACTOR > 4
        float distanceXs05 = coordX5 - atomInfoX[i].s0;
        float partialResults05 = half_rsqrt(distanceXs05 * distanceXs05 + distanceYs0 * distanceYs0 + atomInfoZ[i].s0);
        energyValue5 += atomInfoW[i].s0 * partialResults05;

        float distanceXs15 = coordX5 - atomInfoX[i].s1;
        float partialResults15 = half_rsqrt(distanceXs15 * distanceXs15 + distanceYs1 * distanceYs1 + atomInfoZ[i].s1);
        energyValue5 += atomInfoW[i].s1 * partialResults15;

        float distanceXs25 = coordX5 - atomInfoX[i].s2;
        float partialResults25 = half_rsqrt(distanceXs25 * distanceXs25 + distanceYs2 * distanceYs2 + atomInfoZ[i].s2);
        energyValue5 += atomInfoW[i].s2 * partialResults25;

        float distanceXs35 = coordX5 - atomInfoX[i].s3;
        float partialResults35 = half_rsqrt(distanceXs35 * distanceXs35 + distanceYs3 * distanceYs3 + atomInfoZ[i].s3);
        energyValue5 += atomInfoW[i].s3 * partialResults35;

        float distanceXs45 = coordX5 - atomInfoX[i].s4;
        float partialResults45 = half_rsqrt(distanceXs45 * distanceXs45 + distanceYs4 * distanceYs4 + atomInfoZ[i].s4);
        energyValue5 += atomInfoW[i].s4 * partialResults45;

        float distanceXs55 = coordX5 - atomInfoX[i].s5;
        float partialResults55 = half_rsqrt(distanceXs55 * distanceXs55 + distanceYs5 * distanceYs5 + atomInfoZ[i].s5);
        energyValue5 += atomInfoW[i].s5 * partialResults55;

        float distanceXs65 = coordX5 - atomInfoX[i].s6;
        float partialResults65 = half_rsqrt(distanceXs65 * distanceXs65 + distanceYs6 * distanceYs6 + atomInfoZ[i].s6);
        energyValue5 += atomInfoW[i].s6 * partialResults65;

        float distanceXs75 = coordX5 - atomInfoX[i].s7;
        float partialResults75 = half_rsqrt(distanceXs75 * distanceXs75 + distanceYs7 * distanceYs7 + atomInfoZ[i].s7);
        energyValue5 += atomInfoW[i].s7 * partialResults75;

        float distanceXs06 = coordX6 - atomInfoX[i].s0;
        float partialResults06 = half_rsqrt(distanceXs06 * distanceXs06 + distanceYs0 * distanceYs0 + atomInfoZ[i].s0);
        energyValue6 += atomInfoW[i].s0 * partialResults06;

        float distanceXs16 = coordX6 - atomInfoX[i].s1;
        float partialResults16 = half_rsqrt(distanceXs16 * distanceXs16 + distanceYs1 * distanceYs1 + atomInfoZ[i].s1);
        energyValue6 += atomInfoW[i].s1 * partialResults16;

        float distanceXs26 = coordX6 - atomInfoX[i].s2;
        float partialResults26 = half_rsqrt(distanceXs26 * distanceXs26 + distanceYs2 * distanceYs2 + atomInfoZ[i].s2);
        energyValue6 += atomInfoW[i].s2 * partialResults26;

        float distanceXs36 = coordX6 - atomInfoX[i].s3;
        float partialResults36 = half_rsqrt(distanceXs36 * distanceXs36 + distanceYs3 * distanceYs3 + atomInfoZ[i].s3);
        energyValue6 += atomInfoW[i].s3 * partialResults36;

        float distanceXs46 = coordX6 - atomInfoX[i].s4;
        float partialResults46 = half_rsqrt(distanceXs46 * distanceXs46 + distanceYs4 * distanceYs4 + atomInfoZ[i].s4);
        energyValue6 += atomInfoW[i].s4 * partialResults46;

        float distanceXs56 = coordX6 - atomInfoX[i].s5;
        float partialResults56 = half_rsqrt(distanceXs56 * distanceXs56 + distanceYs5 * distanceYs5 + atomInfoZ[i].s5);
        energyValue6 += atomInfoW[i].s5 * partialResults56;

        float distanceXs66 = coordX6 - atomInfoX[i].s6;
        float partialResults66 = half_rsqrt(distanceXs66 * distanceXs66 + distanceYs6 * distanceYs6 + atomInfoZ[i].s6);
        energyValue6 += atomInfoW[i].s6 * partialResults66;

        float distanceXs76 = coordX6 - atomInfoX[i].s7;
        float partialResults76 = half_rsqrt(distanceXs76 * distanceXs76 + distanceYs7 * distanceYs7 + atomInfoZ[i].s7);
        energyValue6 += atomInfoW[i].s7 * partialResults76;

        float distanceXs07 = coordX7 - atomInfoX[i].s0;
        float partialResults07 = half_rsqrt(distanceXs07 * distanceXs07 + distanceYs0 * distanceYs0 + atomInfoZ[i].s0);
        energyValue7 += atomInfoW[i].s0 * partialResults07;

        float distanceXs17 = coordX7 - atomInfoX[i].s1;
        float partialResults17 = half_rsqrt(distanceXs17 * distanceXs17 + distanceYs1 * distanceYs1 + atomInfoZ[i].s1);
        energyValue7 += atomInfoW[i].s1 * partialResults17;

        float distanceXs27 = coordX7 - atomInfoX[i].s2;
        float partialResults27 = half_rsqrt(distanceXs27 * distanceXs27 + distanceYs2 * distanceYs2 + atomInfoZ[i].s2);
        energyValue7 += atomInfoW[i].s2 * partialResults27;

        float distanceXs37 = coordX7 - atomInfoX[i].s3;
        float partialResults37 = half_rsqrt(distanceXs37 * distanceXs37 + distanceYs3 * distanceYs3 + atomInfoZ[i].s3);
        energyValue7 += atomInfoW[i].s3 * partialResults37;

        float distanceXs47 = coordX7 - atomInfoX[i].s4;
        float partialResults47 = half_rsqrt(distanceXs47 * distanceXs47 + distanceYs4 * distanceYs4 + atomInfoZ[i].s4);
        energyValue7 += atomInfoW[i].s4 * partialResults47;

        float distanceXs57 = coordX7 - atomInfoX[i].s5;
        float partialResults57 = half_rsqrt(distanceXs57 * distanceXs57 + distanceYs5 * distanceYs5 + atomInfoZ[i].s5);
        energyValue7 += atomInfoW[i].s5 * partialResults57;

        float distanceXs67 = coordX7 - atomInfoX[i].s6;
        float partialResults67 = half_rsqrt(distanceXs67 * distanceXs67 + distanceYs6 * distanceYs6 + atomInfoZ[i].s6);
        energyValue7 += atomInfoW[i].s6 * partialResults67;

        float distanceXs77 = coordX7 - atomInfoX[i].s7;
        float partialResults77 = half_rsqrt(distanceXs77 * distanceXs77 + distanceYs7 * distanceYs7 + atomInfoZ[i].s7);
        energyValue7 += atomInfoW[i].s7 * partialResults77;

        float distanceXs08 = coordX8 - atomInfoX[i].s0;
        float partialResults08 = half_rsqrt(distanceXs08 * distanceXs08 + distanceYs0 * distanceYs0 + atomInfoZ[i].s0);
        energyValue8 += atomInfoW[i].s0 * partialResults08;

        float distanceXs18 = coordX8 - atomInfoX[i].s1;
        float partialResults18 = half_rsqrt(distanceXs18 * distanceXs18 + distanceYs1 * distanceYs1 + atomInfoZ[i].s1);
        energyValue8 += atomInfoW[i].s1 * partialResults18;

        float distanceXs28 = coordX8 - atomInfoX[i].s2;
        float partialResults28 = half_rsqrt(distanceXs28 * distanceXs28 + distanceYs2 * distanceYs2 + atomInfoZ[i].s2);
        energyValue8 += atomInfoW[i].s2 * partialResults28;

        float distanceXs38 = coordX8 - atomInfoX[i].s3;
        float partialResults38 = half_rsqrt(distanceXs38 * distanceXs38 + distanceYs3 * distanceYs3 + atomInfoZ[i].s3);
        energyValue8 += atomInfoW[i].s3 * partialResults38;

        float distanceXs48 = coordX8 - atomInfoX[i].s4;
        float partialResults48 = half_rsqrt(distanceXs48 * distanceXs48 + distanceYs4 * distanceYs4 + atomInfoZ[i].s4);
        energyValue8 += atomInfoW[i].s4 * partialResults48;

        float distanceXs58 = coordX8 - atomInfoX[i].s5;
        float partialResults58 = half_rsqrt(distanceXs58 * distanceXs58 + distanceYs5 * distanceYs5 + atomInfoZ[i].s5);
        energyValue8 += atomInfoW[i].s5 * partialResults58;

        float distanceXs68 = coordX8 - atomInfoX[i].s6;
        float partialResults68 = half_rsqrt(distanceXs68 * distanceXs68 + distanceYs6 * distanceYs6 + atomInfoZ[i].s6);
        energyValue8 += atomInfoW[i].s6 * partialResults68;

        float distanceXs78 = coordX8 - atomInfoX[i].s7;
        float partialResults78 = half_rsqrt(distanceXs78 * distanceXs78 + distanceYs7 * distanceYs7 + atomInfoZ[i].s7);
        energyValue8 += atomInfoW[i].s7 * partialResults78;
        #endif // OUTER_UNROLL_FACTOR > 4
        #endif // USE_SOA
    }
    #endif // VECTOR_TYPE

    #elif USE_SOA == 2
    #if INNER_UNROLL_FACTOR > 0
    #pragma unroll INNER_UNROLL_FACTOR
    #endif
    for (int i = 0; i < numberOfAtoms / VECTOR_TYPE; i++)
    {
        vector distanceX = coordX - atomInfoX[i];
        vector distanceY = coordY - atomInfoY[i];
        vector partialResult = half_rsqrt(distanceX * distanceX + distanceY * distanceY + atomInfoZ[i]);
        energyValueVect += atomInfoW[i] * partialResult;


        #if OUTER_UNROLL_FACTOR > 1
        vector distanceX2 = coordX2 - atomInfoX[i];
        vector partialResult2 = half_rsqrt(distanceX2 * distanceX2 + distanceY * distanceY + atomInfoZ[i]);
        energyValueVect2 += atomInfoW[i] * partialResult2;
        #endif // OUTER_UNROLL_FACTOR > 1


        #if OUTER_UNROLL_FACTOR > 2
        vector distanceX3 = coordX3 - atomInfoX[i];
        vector partialResult3 = half_rsqrt(distanceX3 * distanceX3 + distanceY * distanceY + atomInfoZ[i]);
        energyValueVect3 += atomInfoW[i] * partialResult3;

        vector distanceX4 = coordX4 - atomInfoX[i];
        vector partialResult4 = half_rsqrt(distanceX4 * distanceX4 + distanceY * distanceY + atomInfoZ[i]);
        energyValueVect4 += atomInfoW[i] * partialResult4;
        #endif // OUTER_UNROLL_FACTOR > 2


        #if OUTER_UNROLL_FACTOR > 4
        vector distanceX5 = coordX5 - atomInfoX[i];
        vector partialResult5 = half_rsqrt(distanceX5 * distanceX5 + distanceY * distanceY + atomInfoZ[i]);
        energyValueVect5 += atomInfoW[i] * partialResult5;

        vector distanceX6 = coordX6 - atomInfoX[i];
        vector partialResult6 = half_rsqrt(distanceX6 * distanceX6 + distanceY * distanceY + atomInfoZ[i]);
        energyValueVect6 += atomInfoW[i] * partialResult6;

        vector distanceX7 = coordX7 - atomInfoX[i];
        vector partialResult7 = half_rsqrt(distanceX7 * distanceX7 + distanceY * distanceY + atomInfoZ[i]);
        energyValueVect7 += atomInfoW[i] * partialResult7;

        vector distanceX8 = coordX8 - atomInfoX[i];
        vector partialResult8 = half_rsqrt(distanceX8 * distanceX8 + distanceY * distanceY + atomInfoZ[i]);
        energyValueVect8 += atomInfoW[i] * partialResult8;
        #endif // OUTER_UNROLL_FACTOR > 4
    }
    #endif // USE_SOA

    #if USE_SOA < 2
    energyGrid[outIndex] += energyValue;
    #if OUTER_UNROLL_FACTOR > 1
    energyGrid[outIndex + 1] += energyValue2;
    #endif // OUTER_UNROLL_FACTOR > 1
    #if OUTER_UNROLL_FACTOR > 2
    energyGrid[outIndex + 2] += energyValue3;
    energyGrid[outIndex + 3] += energyValue4;
    #endif // OUTER_UNROLL_FACTOR > 2
    #if OUTER_UNROLL_FACTOR > 4
    energyGrid[outIndex + 4] += energyValue5;
    energyGrid[outIndex + 5] += energyValue6;
    energyGrid[outIndex + 6] += energyValue7;
    energyGrid[outIndex + 7] += energyValue8;
    #endif // OUTER_UNROLL_FACTOR > 4

    #elif USE_SOA == 2
    #if VECTOR_TYPE == 2
    energyGrid[outIndex] += energyValueVect.x + energyValueVect.y;
    #if OUTER_UNROLL_FACTOR > 1
    energyGrid[outIndex + 1] += energyValueVect2.x + energyValueVect2.y;
    #endif // OUTER_UNROLL_FACTOR > 1
    #if OUTER_UNROLL_FACTOR > 2
    energyGrid[outIndex + 2] += energyValueVect3.x + energyValueVect3.y;
    energyGrid[outIndex + 3] += energyValueVect4.x + energyValueVect4.y;
    #endif // OUTER_UNROLL_FACTOR > 2
    #if OUTER_UNROLL_FACTOR > 4
    energyGrid[outIndex + 4] += energyValueVect5.x + energyValueVect5.y;
    energyGrid[outIndex + 5] += energyValueVect6.x + energyValueVect6.y;
    energyGrid[outIndex + 6] += energyValueVect7.x + energyValueVect7.y;
    energyGrid[outIndex + 7] += energyValueVect8.x + energyValueVect8.y;
    #endif // OUTER_UNROLL_FACTOR > 4

    #elif VECTOR_TYPE == 4
    energyGrid[outIndex] += energyValueVect.x + energyValueVect.y + energyValueVect.z + energyValueVect.w;
    #if OUTER_UNROLL_FACTOR > 1
    energyGrid[outIndex + 1] += energyValueVect2.x + energyValueVect2.y + energyValueVect2.z + energyValueVect2.w;
    #endif // OUTER_UNROLL_FACTOR > 1
    #if OUTER_UNROLL_FACTOR > 2
    energyGrid[outIndex + 2] += energyValueVect3.x + energyValueVect3.y + energyValueVect3.z + energyValueVect3.w;
    energyGrid[outIndex + 3] += energyValueVect4.x + energyValueVect4.y + energyValueVect4.z + energyValueVect4.w;
    #endif // OUTER_UNROLL_FACTOR > 2
    #if OUTER_UNROLL_FACTOR > 4
    energyGrid[outIndex + 4] += energyValueVect5.x + energyValueVect5.y + energyValueVect5.z + energyValueVect5.w;
    energyGrid[outIndex + 5] += energyValueVect6.x + energyValueVect6.y + energyValueVect6.z + energyValueVect6.w;
    energyGrid[outIndex + 6] += energyValueVect7.x + energyValueVect7.y + energyValueVect7.z + energyValueVect7.w;
    energyGrid[outIndex + 7] += energyValueVect8.x + energyValueVect8.y + energyValueVect8.z + energyValueVect8.w;
    #endif // OUTER_UNROLL_FACTOR > 4

    #elif VECTOR_TYPE == 8
    energyGrid[outIndex] += energyValueVect.s0 + energyValueVect.s1 + energyValueVect.s2 + energyValueVect.s3 + energyValueVect.s4 + energyValueVect.s5 + energyValueVect.s6 + energyValueVect.s7;
    #if OUTER_UNROLL_FACTOR > 1
    energyGrid[outIndex + 1] += energyValueVect2.s0 + energyValueVect2.s1 + energyValueVect2.s2 + energyValueVect2.s3 + energyValueVect2.s4 + energyValueVect2.s5 + energyValueVect2.s6 + energyValueVect2.s7;
    #endif // OUTER_UNROLL_FACTOR > 1
    #if OUTER_UNROLL_FACTOR > 2
    energyGrid[outIndex + 2] += energyValueVect3.s0 + energyValueVect3.s1 + energyValueVect3.s2 + energyValueVect3.s3 + energyValueVect3.s4 + energyValueVect3.s5 + energyValueVect3.s6 + energyValueVect3.s7;
    energyGrid[outIndex + 3] += energyValueVect4.s0 + energyValueVect4.s1 + energyValueVect4.s2 + energyValueVect4.s3 + energyValueVect4.s4 + energyValueVect4.s5 + energyValueVect4.s6 + energyValueVect4.s7;
    #endif // OUTER_UNROLL_FACTOR > 2
    #if OUTER_UNROLL_FACTOR > 4
    energyGrid[outIndex + 4] += energyValueVect5.s0 + energyValueVect5.s1 + energyValueVect5.s2 + energyValueVect5.s3 + energyValueVect5.s4 + energyValueVect5.s5 + energyValueVect5.s6 + energyValueVect5.s7;
    energyGrid[outIndex + 5] += energyValueVect6.s0 + energyValueVect6.s1 + energyValueVect6.s2 + energyValueVect6.s3 + energyValueVect6.s4 + energyValueVect6.s5 + energyValueVect6.s6 + energyValueVect6.s7;
    energyGrid[outIndex + 6] += energyValueVect7.s0 + energyValueVect7.s1 + energyValueVect7.s2 + energyValueVect7.s3 + energyValueVect7.s4 + energyValueVect7.s5 + energyValueVect7.s6 + energyValueVect7.s7;
    energyGrid[outIndex + 7] += energyValueVect8.s0 + energyValueVect8.s1 + energyValueVect8.s2 + energyValueVect8.s3 + energyValueVect8.s4 + energyValueVect8.s5 + energyValueVect8.s6 + energyValueVect8.s7;
    #endif // OUTER_UNROLL_FACTOR > 4

    #elif VECTOR_TYPE == 16
    energyGrid[outIndex] += energyValueVect.s0 + energyValueVect.s1 + energyValueVect.s2 + energyValueVect.s3 + energyValueVect.s4 + energyValueVect.s5 + energyValueVect.s6 + energyValueVect.s7 + energyValueVect.s8 + energyValueVect.s9 + energyValueVect.sa + energyValueVect.sb + energyValueVect.sc + energyValueVect.sd + energyValueVect.se + energyValueVect.sf;
    #if OUTER_UNROLL_FACTOR > 1
    energyGrid[outIndex + 1] += energyValueVect2.s0 + energyValueVect2.s1 + energyValueVect2.s2 + energyValueVect2.s3 + energyValueVect2.s4 + energyValueVect2.s5 + energyValueVect2.s6 + energyValueVect2.s7 + energyValueVect2.s8 + energyValueVect2.s9 + energyValueVect2.sa + energyValueVect2.sb + energyValueVect2.sc + energyValueVect2.sd + energyValueVect2.se + energyValueVect2.sf;
    #endif // OUTER_UNROLL_FACTOR > 1
    #if OUTER_UNROLL_FACTOR > 2
    energyGrid[outIndex + 2] += energyValueVect3.s0 + energyValueVect3.s1 + energyValueVect3.s2 + energyValueVect3.s3 + energyValueVect3.s4 + energyValueVect3.s5 + energyValueVect3.s6 + energyValueVect3.s7 + energyValueVect3.s8 + energyValueVect3.s9 + energyValueVect3.sa + energyValueVect3.sb + energyValueVect3.sc + energyValueVect3.sd + energyValueVect3.se + energyValueVect3.sf;
    energyGrid[outIndex + 3] += energyValueVect4.s0 + energyValueVect4.s1 + energyValueVect4.s2 + energyValueVect4.s3 + energyValueVect4.s4 + energyValueVect4.s5 + energyValueVect4.s6 + energyValueVect4.s7 + energyValueVect4.s8 + energyValueVect4.s9 + energyValueVect4.sa + energyValueVect4.sb + energyValueVect4.sc + energyValueVect4.sd + energyValueVect4.se + energyValueVect4.sf;
    #endif // OUTER_UNROLL_FACTOR > 2
    #if OUTER_UNROLL_FACTOR > 4
    energyGrid[outIndex + 4] += energyValueVect5.s0 + energyValueVect5.s1 + energyValueVect5.s2 + energyValueVect5.s3 + energyValueVect5.s4 + energyValueVect5.s5 + energyValueVect5.s6 + energyValueVect5.s7 + energyValueVect5.s8 + energyValueVect5.s9 + energyValueVect5.sa + energyValueVect5.sb + energyValueVect5.sc + energyValueVect5.sd + energyValueVect5.se + energyValueVect5.sf;
    energyGrid[outIndex + 5] += energyValueVect6.s0 + energyValueVect6.s1 + energyValueVect6.s2 + energyValueVect6.s3 + energyValueVect6.s4 + energyValueVect6.s5 + energyValueVect6.s6 + energyValueVect6.s7 + energyValueVect6.s8 + energyValueVect6.s9 + energyValueVect6.sa + energyValueVect6.sb + energyValueVect6.sc + energyValueVect6.sd + energyValueVect6.se + energyValueVect6.sf;
    energyGrid[outIndex + 6] += energyValueVect7.s0 + energyValueVect7.s1 + energyValueVect7.s2 + energyValueVect7.s3 + energyValueVect7.s4 + energyValueVect7.s5 + energyValueVect7.s6 + energyValueVect7.s7 + energyValueVect7.s8 + energyValueVect7.s9 + energyValueVect7.sa + energyValueVect7.sb + energyValueVect7.sc + energyValueVect7.sd + energyValueVect7.se + energyValueVect7.sf;
    energyGrid[outIndex + 7] += energyValueVect8.s0 + energyValueVect8.s1 + energyValueVect8.s2 + energyValueVect8.s3 + energyValueVect8.s4 + energyValueVect8.s5 + energyValueVect8.s6 + energyValueVect8.s7 + energyValueVect8.s8 + energyValueVect8.s9 + energyValueVect8.sa + energyValueVect8.sb + energyValueVect8.sc + energyValueVect8.sd + energyValueVect8.se + energyValueVect8.sf;
    #endif // OUTER_UNROLL_FACTOR > 4
    #endif // VECTOR_TYPE
    #endif // USE_SOA
}
