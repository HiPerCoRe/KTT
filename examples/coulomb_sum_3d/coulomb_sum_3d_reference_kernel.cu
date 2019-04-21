extern "C" __global__ void directCoulombSumReference(float4* atomInfo, int numberOfAtoms, float gridSpacing, float* energyGrid)
{
    int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
    int zIndex = blockIdx.z*blockDim.z + threadIdx.z;
    
	int outIndex = blockDim.y*gridDim.y * blockDim.x*gridDim.x * zIndex + blockDim.x*gridDim.x * yIndex + xIndex;

    float coordX = gridSpacing * xIndex;
    float coordY = gridSpacing * yIndex;
    float coordZ = gridSpacing * zIndex;

    float energyValue = 0.0f;

    for (int i = 0; i < numberOfAtoms; i++)
    {
        float dX = coordX - atomInfo[i].x;
        float dY = coordY - atomInfo[i].y;
        float dZ = coordZ - atomInfo[i].z;
        float partialResult = rsqrt(dX * dX + dY * dY + dZ*dZ);
        energyValue += atomInfo[i].w * partialResult;
    }

    energyGrid[outIndex] += energyValue;
}
