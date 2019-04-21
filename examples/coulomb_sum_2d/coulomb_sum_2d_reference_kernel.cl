__kernel void directCoulombSumReference(__global float4* atomInfo, int numberOfAtoms, float gridSpacing, __global float* energyGrid)
{
    int xIndex = get_global_id(0);
    int yIndex = get_global_id(1);
        
    int outIndex = get_global_size(0) * yIndex + xIndex;

    float currentEnergy = energyGrid[outIndex];

    float coordX = gridSpacing * xIndex;
    float coordY = gridSpacing * yIndex;
    float energyValue = 0.0f;

    for (int i = 0; i < numberOfAtoms; i++)
    {
        float distanceX = coordX - atomInfo[i].x;
        float distanceY = coordY - atomInfo[i].y;
        float partialResult = half_rsqrt(distanceX * distanceX + distanceY * distanceY + atomInfo[i].z);
        energyValue += atomInfo[i].w * partialResult;
    }

    energyGrid[outIndex] += currentEnergy + energyValue;
}
