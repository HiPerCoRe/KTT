__kernel void directCoulombSumReference(__global float4* atomInfo, int numberOfAtoms, float gridSpacing, __global float* energyGrid)
{
    int xIndex = get_global_id(0);
    int yIndex = get_global_id(1);
    int zIndex = get_global_id(2);
   
	int outIndex = get_global_size(1) * get_global_size(0) * zIndex + get_global_size(0) * yIndex + xIndex;

    float coordX = gridSpacing * xIndex;
    float coordY = gridSpacing * yIndex;
    //XXX this code work correctly when zIndex is fixed to constant here and in coulomb_sum_tunable.h, e.g. float coordZ = gridSpacing * 1;
    float coordZ = gridSpacing * zIndex;

    float energyValue = 0.0f;

    for (int i = 0; i < numberOfAtoms; i++)
    {
        float dX = coordX - atomInfo[i].x;
        float dY = coordY - atomInfo[i].y;
        float dZ = coordZ - atomInfo[i].z;
        float partialResult = half_rsqrt(dX * dX + dY * dY + dZ*dZ);
        energyValue += atomInfo[i].w * partialResult;
    }
    
    energyGrid[outIndex] += energyValue;
}
