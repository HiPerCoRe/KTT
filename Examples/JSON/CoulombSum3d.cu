extern "C" __global__ void directCoulombSum(const float4* atomInfo, const float* atomInfoX, const float* atomInfoY, const float* atomInfoZ, const float* atomInfoW, int numberOfAtoms, float gridSpacing, int gridSize, float* energyGrid)
{
    int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
    int zIndex = (blockIdx.z*blockDim.z + threadIdx.z) * Z_ITERATIONS;

    if ((xIndex >= gridSize) || (yIndex >= gridSize) || (xIndex >= gridSize))
        return;
        
    int sliceOffset = gridSize * gridSize;
	int outIndex = sliceOffset*zIndex + gridSize*yIndex + xIndex;

    float coordX = gridSpacing * xIndex;
    float coordY = gridSpacing * yIndex;
    float coordZ = gridSpacing * zIndex;

    float energyValue[Z_ITERATIONS];
    for (int i = 0; i < Z_ITERATIONS; i++) 
        energyValue[i] = 0.0f;

    for (int i = 0; i < numberOfAtoms / VECTOR_SIZE; i++)
    {
        #if USE_SOA == 1
        float dX = coordX - atomInfoX[i];
        float dY = coordY - atomInfoY[i];
        float dZ = coordZ - atomInfoZ[i];
        float w = atomInfoW[i];
        #else
        float dX = coordX - atomInfo[i].x;
        float dY = coordY - atomInfo[i].y;
        float dZ = coordZ - atomInfo[i].z;
        float w = atomInfo[i].w;
        #endif /* USE_SOA */
        #if INNER_UNROLL_FACTOR > 0
        #pragma unroll INNER_UNROLL_FACTOR
        #endif
        for (int j = 0; j < Z_ITERATIONS; j++) {
            float partialResult = rsqrt(dX * dX + dY * dY + dZ*dZ);
            energyValue[j] += w * partialResult;
            dZ += gridSpacing;
        }
    }

    for (int i = 0; i < Z_ITERATIONS; i++)
        if (zIndex + i < gridSize)
            energyGrid[outIndex + sliceOffset*i] += energyValue[i];
}
