__kernel void reduce(__global float* input, __global float* result)
{
    __local float partials[WG_SIZE];

    const size_t localId = get_local_id(0);
    const size_t localSize = get_local_size(0);
    const size_t groupId = get_group_id(0);

    partials[localId] = input[groupId * localSize + localId];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t stride = localSize / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            partials[localId] += partials[localId + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Plain store, not an atomic add: each slot belongs to exactly one work group.
    if (localId == 0) {
        result[groupId] = partials[0];
    }
}
