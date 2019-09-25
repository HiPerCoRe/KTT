// This kernel code based on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

#if SORT_VECTOR == 2
#define SORTVECTYPE uint2
#elif SORT_VECTOR == 4
#define SORTVECTYPE uint4
#elif SORT_VECTOR == 8
#define SORTVECTYPE uint8
#endif

#if SCAN_VECTOR == 2
#define SCANVECTYPE uint2
#elif SCAN_VECTOR == 4
#define SCANVECTYPE uint4
#elif SCAN_VECTOR == 8
#define SCANVECTYPE uint8
#endif

uint scanLSB(const uint val, __local uint* s_data)
{
    // Set first half of shared mem to 0's
    int idx = get_local_id(0);
    s_data[idx] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Set 2nd half to thread local sum
    idx += get_local_size(0);

    // scan in local memory

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;     barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 1; i < SORT_BLOCK_SIZE; i*=2) {
      t = s_data[idx - i];  barrier(CLK_LOCAL_MEM_FENCE);;
      s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);;
    }
    return s_data[idx] - val;  // convert inclusive -> exclusive
}

SORTVECTYPE scan4(SORTVECTYPE idata, __local uint* ptr)
{
    SORTVECTYPE val4 = idata;
    SORTVECTYPE sum;

    // Scan the elements in idata within this thread
#if SORT_VECTOR == 2
    sum.x = val4.x;
    uint val = val4.y + sum.x;
#elif SORT_VECTOR == 4
    sum.x = val4.x;
    sum.y = val4.y + sum.x;
    sum.z = val4.z + sum.y;
    uint val = val4.w + sum.z;
#elif SORT_VECTOR == 8
    sum.s0 = val4.s0;
    sum.s1 = val4.s1 + sum.s0;
    sum.s2 = val4.s2 + sum.s1;
    sum.s3 = val4.s3 + sum.s2;
    sum.s4 = val4.s4 + sum.s3;
    sum.s5 = val4.s5 + sum.s4;
    sum.s6 = val4.s6 + sum.s5;
    uint val = val4.s7 + sum.s6;
#endif

    // Now scan those sums across the local work group
    val = scanLSB(val, ptr);

#if SORT_VECTOR == 2
    val4.x = val;
    val4.y = val + sum.x;
#elif SORT_VECTOR == 4
    val4.x = val;
    val4.y = val + sum.x;
    val4.z = val + sum.y;
    val4.w = val + sum.z;
#elif SORT_VECTOR == 8
    val4.s0 = val;
    val4.s1 = val + sum.s0;
    val4.s2 = val + sum.s1;
    val4.s3 = val + sum.s2;
    val4.s4 = val + sum.s3;
    val4.s5 = val + sum.s4;
    val4.s6 = val + sum.s5;
    val4.s7 = val + sum.s6;
#endif

    return val4;
}

//----------------------------------------------------------------------------
//
// radixSortBlocks sorts all blocks of data independently in shared
// memory.  Each thread block (CTA) sorts one block of SORT_VECTOR*CTA_SIZE elements
//
// The radix sort is done in two stages.  This stage calls radixSortBlock
// on each block independently, sorting on the basis of bits
// (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------

__kernel void radixSortBlocks(const uint nbits, 
                              const uint startbit,
                              __global SORTVECTYPE* keysOut,
                              __global SORTVECTYPE* valuesOut,
                              __global SORTVECTYPE* keysIn,  
                              __global SORTVECTYPE* valuesIn)
{
    __local uint sMem[SORT_VECTOR*SORT_BLOCK_SIZE];

    // Get Indexing information
    const uint i = get_global_id(0);
    const uint tid = get_local_id(0);
    const uint localSize = get_local_size(0);

    // Load keys and vals from global memory
    SORTVECTYPE key, value;
    key = keysIn[i];
    value = valuesIn[i];

    __local uint numtrue;

    // For each of the 4 bits
    for(uint shift = startbit; shift < (startbit + nbits); ++shift)
    {
        // Check if the LSB is 0
        SORTVECTYPE lsb;
#if SORT_VECTOR == 2
        lsb.x = !((key.x >> shift) & 0x1);
        lsb.y = !((key.y >> shift) & 0x1);
#elif SORT_VECTOR == 4
        lsb.x = !((key.x >> shift) & 0x1);
        lsb.y = !((key.y >> shift) & 0x1);
        lsb.z = !((key.z >> shift) & 0x1);
        lsb.w = !((key.w >> shift) & 0x1);
#elif SORT_VECTOR == 8
        lsb.s0 = !((key.s0 >> shift) & 0x1);
        lsb.s1 = !((key.s1 >> shift) & 0x1);
        lsb.s2 = !((key.s2 >> shift) & 0x1);
        lsb.s3 = !((key.s3 >> shift) & 0x1);
        lsb.s4 = !((key.s4 >> shift) & 0x1);
        lsb.s5 = !((key.s5 >> shift) & 0x1);
        lsb.s6 = !((key.s6 >> shift) & 0x1);
        lsb.s7 = !((key.s7 >> shift) & 0x1);
#endif

        // Do an exclusive scan of how many elems have 0's in the LSB
        // When this is finished, address.n will contain the number of
        // elems with 0 in the LSB which precede elem n
        SORTVECTYPE address = scan4(lsb, sMem);

        // Store the total number of elems with an LSB of 0
        // to shared mem
        if (tid == localSize - 1)
        {
#if SORT_VECTOR == 2
          numtrue = address.y + lsb.y;
#elif SORT_VECTOR == 4
            numtrue = address.w + lsb.w;
#elif SORT_VECTOR == 8
            numtrue = address.s7 + lsb.s7;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);;

        // Determine rank -- position in the block
        // If you are a 0 --> your position is the scan of 0's
        // If you are a 1 --> your position is calculated as below
        SORTVECTYPE rank;
        const int idx = tid*SORT_VECTOR;
#if SORT_VECTOR == 2
        rank.x = lsb.x ? address.x : numtrue + idx     - address.x;
        rank.y = lsb.y ? address.y : numtrue + idx + 1 - address.y;
#elif SORT_VECTOR == 4
        rank.x = lsb.x ? address.x : numtrue + idx     - address.x;
        rank.y = lsb.y ? address.y : numtrue + idx + 1 - address.y;
        rank.z = lsb.z ? address.z : numtrue + idx + 2 - address.z;
        rank.w = lsb.w ? address.w : numtrue + idx + 3 - address.w;
#elif SORT_VECTOR == 8
        rank.s0 = lsb.s0 ? address.s0 : numtrue + idx     - address.s0;
        rank.s1 = lsb.s1 ? address.s1 : numtrue + idx + 1 - address.s1;
        rank.s2 = lsb.s2 ? address.s2 : numtrue + idx + 2 - address.s2;
        rank.s3 = lsb.s3 ? address.s3 : numtrue + idx + 3 - address.s3;
        rank.s4 = lsb.s4 ? address.s4 : numtrue + idx + 4 - address.s4;
        rank.s5 = lsb.s5 ? address.s5 : numtrue + idx + 5 - address.s5;
        rank.s6 = lsb.s6 ? address.s6 : numtrue + idx + 6 - address.s6;
        rank.s7 = lsb.s7 ? address.s7 : numtrue + idx + 7 - address.s7;
#endif

        // Scatter keys into local mem
#if SORT_VECTOR == 2
        sMem[(rank.x & 1) * localSize + (rank.x >> 1)] = key.x;
        sMem[(rank.y & 1) * localSize + (rank.y >> 1)] = key.y;
#elif SORT_VECTOR == 4
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = key.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = key.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = key.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = key.w;
#elif SORT_VECTOR == 8
        sMem[(rank.s0 & 7) * localSize + (rank.s0 >> 3)] = key.s0;
        sMem[(rank.s1 & 7) * localSize + (rank.s1 >> 3)] = key.s1;
        sMem[(rank.s2 & 7) * localSize + (rank.s2 >> 3)] = key.s2;
        sMem[(rank.s3 & 7) * localSize + (rank.s3 >> 3)] = key.s3;
        sMem[(rank.s4 & 7) * localSize + (rank.s4 >> 3)] = key.s4;
        sMem[(rank.s5 & 7) * localSize + (rank.s5 >> 3)] = key.s5;
        sMem[(rank.s6 & 7) * localSize + (rank.s6 >> 3)] = key.s6;
        sMem[(rank.s7 & 7) * localSize + (rank.s7 >> 3)] = key.s7;
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
#if SORT_VECTOR == 2
        key.x = sMem[tid];
        key.y = sMem[tid +     localSize];
#elif SORT_VECTOR == 4
        key.x = sMem[tid];
        key.y = sMem[tid +     localSize];
        key.z = sMem[tid + 2 * localSize];
        key.w = sMem[tid + 3 * localSize];
#elif SORT_VECTOR == 8
        key.s0 = sMem[tid];
        key.s1 = sMem[tid +     localSize];
        key.s2 = sMem[tid + 2 * localSize];
        key.s3 = sMem[tid + 3 * localSize];
        key.s4 = sMem[tid + 4 * localSize];
        key.s5 = sMem[tid + 5 * localSize];
        key.s6 = sMem[tid + 6 * localSize];
        key.s7 = sMem[tid + 7 * localSize];
#endif
        barrier(CLK_LOCAL_MEM_FENCE);;

        // Scatter values into local mem
#if SORT_VECTOR == 2
        sMem[(rank.x & 1) * localSize + (rank.x >> 1)] = value.x;
        sMem[(rank.y & 1) * localSize + (rank.y >> 1)] = value.y;
#elif SORT_VECTOR == 4
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = value.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = value.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = value.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = value.w;
#elif SORT_VECTOR == 8
        sMem[(rank.s0 & 7) * localSize + (rank.s0 >> 3)] = value.s0;
        sMem[(rank.s1 & 7) * localSize + (rank.s1 >> 3)] = value.s1;
        sMem[(rank.s2 & 7) * localSize + (rank.s2 >> 3)] = value.s2;
        sMem[(rank.s3 & 7) * localSize + (rank.s3 >> 3)] = value.s3;
        sMem[(rank.s4 & 7) * localSize + (rank.s4 >> 3)] = value.s4;
        sMem[(rank.s5 & 7) * localSize + (rank.s5 >> 3)] = value.s5;
        sMem[(rank.s6 & 7) * localSize + (rank.s6 >> 3)] = value.s6;
        sMem[(rank.s7 & 7) * localSize + (rank.s7 >> 3)] = value.s7;
#endif
        barrier(CLK_LOCAL_MEM_FENCE);;

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
#if SORT_VECTOR == 2
        value.x = sMem[tid];
        value.y = sMem[tid +     localSize];
#elif SORT_VECTOR == 4
        value.x = sMem[tid];
        value.y = sMem[tid +     localSize];
        value.z = sMem[tid + 2 * localSize];
        value.w = sMem[tid + 3 * localSize];
#elif SORT_VECTOR == 8
        value.s0 = sMem[tid];
        value.s1 = sMem[tid +     localSize];
        value.s2 = sMem[tid + 2 * localSize];
        value.s3 = sMem[tid + 3 * localSize];
        value.s4 = sMem[tid + 4 * localSize];
        value.s5 = sMem[tid + 5 * localSize];
        value.s6 = sMem[tid + 6 * localSize];
        value.s7 = sMem[tid + 7 * localSize];
#endif
        barrier(CLK_LOCAL_MEM_FENCE);;
    }
    keysOut[i]   = key;
    valuesOut[i] = value;
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each
// block counts the number of keys that fall into each radix in the group, and
// finds the starting offset of each radix in the block.  It then writes the
// radix counts to the counters array, and the starting offsets to the
// blockOffsets array.
//
//----------------------------------------------------------------------------
__kernel void findRadixOffsets(__global SCANVECTYPE* keys, 
        __global uint* counters,
        __global uint* blockOffsets, 
        uint startbit, 
        uint numElements, 
        uint totalBlocks)
{
    __local uint  sStartPointers[16];
    __local uint sRadix1[SCAN_VECTOR*SCAN_BLOCK_SIZE];

    uint gid = get_group_id(0);
    uint tid = get_local_id(0);
    uint localSize = get_local_size(0);
    uint i = get_global_id(0);

    SCANVECTYPE radix2;
    radix2 = keys[i];

#if SCAN_VECTOR == 2
    sRadix1[2 * tid]     = (radix2.x >> startbit) & 0xF;
    sRadix1[2 * tid + 1] = (radix2.y >> startbit) & 0xF;
#elif SCAN_VECTOR == 4
    sRadix1[4 * tid]      = (radix2.x >> startbit) & 0xF;
    sRadix1[4 * tid + 1]  = (radix2.y >> startbit) & 0xF;
    sRadix1[4 * tid + 2]  = (radix2.z >> startbit) & 0xF;
    sRadix1[4 * tid + 3]  = (radix2.w >> startbit) & 0xF;
#elif SCAN_VECTOR == 8
    sRadix1[8 * tid]      = (radix2.s0 >> startbit) & 0xF;
    sRadix1[8 * tid + 1]  = (radix2.s1 >> startbit) & 0xF;
    sRadix1[8 * tid + 2]  = (radix2.s2 >> startbit) & 0xF;
    sRadix1[8 * tid + 3]  = (radix2.s3 >> startbit) & 0xF;
    sRadix1[8 * tid + 4]  = (radix2.s4 >> startbit) & 0xF;
    sRadix1[8 * tid + 5]  = (radix2.s5 >> startbit) & 0xF;
    sRadix1[8 * tid + 6]  = (radix2.s6 >> startbit) & 0xF;
    sRadix1[8 * tid + 7]  = (radix2.s7 >> startbit) & 0xF;
#endif

    // Finds the position where the sRadix1 entries differ and stores start
    // index for each radix.
    if(tid < 16)
    {
        sStartPointers[tid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);;

    if((tid > 0) && (sRadix1[tid] != sRadix1[tid - 1]) )
    {
        sStartPointers[sRadix1[tid]] = tid;
    }
    if(sRadix1[tid + localSize] != sRadix1[tid + localSize - 1])
    {
        sStartPointers[sRadix1[tid + localSize]] = tid + localSize;
    }

#if SCAN_VECTOR == 4 || SCAN_VECTOR == 8
    if (sRadix1[tid + 2*localSize] != sRadix1[tid + 2*localSize - 1])
    {
      sStartPointers[sRadix1[tid + 2*localSize]] = tid + 2*localSize;
    }
    if (sRadix1[tid + 3*localSize] != sRadix1[tid + 3*localSize - 1])
    {
      sStartPointers[sRadix1[tid + 3*localSize]] = tid + 3*localSize;
    }
#endif
#if SCAN_VECTOR == 8
    if (sRadix1[tid + 4*localSize] != sRadix1[tid + 4*localSize - 1])
    {
      sStartPointers[sRadix1[tid + 4*localSize]] = tid + 4*localSize;
    }
    if (sRadix1[tid + 5*localSize] != sRadix1[tid + 5*localSize - 1])
    {
      sStartPointers[sRadix1[tid + 5*localSize]] = tid + 5*localSize;
    }
    if (sRadix1[tid + 6*localSize] != sRadix1[tid + 6*localSize - 1])
    {
      sStartPointers[sRadix1[tid + 6*localSize]] = tid + 6*localSize;
    }
    if (sRadix1[tid + 7*localSize] != sRadix1[tid + 7*localSize - 1])
    {
      sStartPointers[sRadix1[tid + 7*localSize]] = tid + 7*localSize;
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);;

    if(tid < 16)
    {
        blockOffsets[gid*16 + tid] = sStartPointers[tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);;

    // Compute the sizes of each block.
    if((tid > 0) && (sRadix1[tid] != sRadix1[tid - 1]) )
    {
        sStartPointers[sRadix1[tid - 1]] =
            tid - sStartPointers[sRadix1[tid - 1]];
    }
    if(sRadix1[tid + localSize] != sRadix1[tid + localSize - 1] )
    {
        sStartPointers[sRadix1[tid + localSize - 1]] =
            tid + localSize - sStartPointers[sRadix1[tid +
                                                         localSize - 1]];
    }
#if SCAN_VECTOR == 4 || SCAN_VECTOR == 8
    if(sRadix1[tid + 2*localSize] != sRadix1[tid + 2*localSize - 1] )
    {
        sStartPointers[sRadix1[tid + 2*localSize - 1]] =
            tid + 2*localSize - sStartPointers[sRadix1[tid + 2*localSize - 1]];
    }
    if(sRadix1[tid + 3*localSize] != sRadix1[tid + 3*localSize - 1] )
    {
        sStartPointers[sRadix1[tid + 3*localSize - 1]] =
            tid + 3*localSize - sStartPointers[sRadix1[tid + 3*localSize - 1]];
    }
#endif
#if SCAN_VECTOR == 8
    if(sRadix1[tid + 4*localSize] != sRadix1[tid + 4*localSize - 1] )
    {
        sStartPointers[sRadix1[tid + 4*localSize - 1]] =
            tid + 4*localSize - sStartPointers[sRadix1[tid + 4*localSize - 1]];
    }
    if(sRadix1[tid + 5*localSize] != sRadix1[tid + 5*localSize - 1] )
    {
        sStartPointers[sRadix1[tid + 5*localSize - 1]] =
            tid + 5*localSize - sStartPointers[sRadix1[tid + 5*localSize - 1]];
    }
    if(sRadix1[tid + 6*localSize] != sRadix1[tid + 6*localSize - 1] )
    {
        sStartPointers[sRadix1[tid + 6*localSize - 1]] =
            tid + 6*localSize - sStartPointers[sRadix1[tid + 6*localSize - 1]];
    }
    if(sRadix1[tid + 7*localSize] != sRadix1[tid + 7*localSize - 1] )
    {
        sStartPointers[sRadix1[tid + 7*localSize - 1]] =
            tid + 7*localSize - sStartPointers[sRadix1[tid + 7*localSize - 1]];
    }
#endif

#if SCAN_VECTOR == 2
    if(tid == localSize - 1)
    {
        sStartPointers[sRadix1[2 * localSize - 1]] =
            2 * localSize - sStartPointers[sRadix1[2 * localSize - 1]];
    }
#elif SCAN_VECTOR == 4
    if (tid == localSize - 1)
    {
        sStartPointers[sRadix1[4 * localSize - 1]] =
            4 * localSize - sStartPointers[sRadix1[4 * localSize - 1]];
    }
#elif SCAN_VECTOR == 8
    if (tid == localSize - 1)
    {
        sStartPointers[sRadix1[8 * localSize - 1]] =
            8 * localSize - sStartPointers[sRadix1[8 * localSize - 1]];
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);;

    if(tid < 16)
    {
        counters[tid * totalBlocks + gid] = sStartPointers[tid];
    }
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets
// have been found. On compute version 1.1 and earlier GPUs, this code depends
// on SORT_BLOCK_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
//----------------------------------------------------------------------------
__kernel void reorderData(uint  startbit,
                          __global uint  *outKeys,
                          __global uint  *outValues,
                          __global SCANVECTYPE *keys,
                          __global SCANVECTYPE *values,
                          __global uint  *blockOffsets,
                          __global uint  *offsets,
                          __global uint  *sizes,
                          uint  totalBlocks)
{
    __local SCANVECTYPE sKeys2[SCAN_BLOCK_SIZE];
    __local SCANVECTYPE sValues2[SCAN_BLOCK_SIZE];
    __local uint  sOffsets[16];
    __local uint  sBlockOffsets[16];
    __local uint* sKeys1   = (__local uint*) &sKeys2;
    __local uint* sValues1 = (__local uint*) &sValues2;

    uint gid = get_group_id(0);
    uint tid = get_local_id(0);
    uint localSize = get_local_size(0);
    uint i = get_global_id(0);

    sKeys2[tid]   = keys[i];
    sValues2[tid] = values[i];

    if(tid < 16)
    {
        sOffsets[tid]      = offsets[tid * totalBlocks +
                                             gid];
        sBlockOffsets[tid] = blockOffsets[gid * 16 + tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint radix = (sKeys1[tid] >> startbit) & 0xF;
    uint globalOffset = sOffsets[radix] + tid - sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[tid];
    outValues[globalOffset] = sValues1[tid];

    radix = (sKeys1[tid + localSize] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + tid + localSize -
                   sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[tid + localSize];
    outValues[globalOffset] = sValues1[tid + localSize];

#if SCAN_VECTOR == 4 || SCAN_VECTOR == 8
    radix = (sKeys1[tid + 2*localSize] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + tid + 2*localSize -
                   sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[tid + 2*localSize];
    outValues[globalOffset] = sValues1[tid + 2*localSize];

    radix = (sKeys1[tid + 3*localSize] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + tid + 3*localSize -
                   sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[tid + 3*localSize];
    outValues[globalOffset] = sValues1[tid + 3*localSize];
#endif
#if SCAN_VECTOR == 8
    radix = (sKeys1[tid + 4*localSize] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + tid + 4*localSize -
                   sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[tid + 4*localSize];
    outValues[globalOffset] = sValues1[tid + 4*localSize];

    radix = (sKeys1[tid + 5*localSize] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + tid + 5*localSize -
                   sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[tid + 5*localSize];
    outValues[globalOffset] = sValues1[tid + 5*localSize];

    radix = (sKeys1[tid + 6*localSize] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + tid + 6*localSize -
                   sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[tid + 6*localSize];
    outValues[globalOffset] = sValues1[tid + 6*localSize];

    radix = (sKeys1[tid + 7*localSize] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + tid + 7*localSize -
                   sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[tid + 7*localSize];
    outValues[globalOffset] = sValues1[tid + 7*localSize];
#endif
}

uint scanLocalMem(const uint val, __local uint* s_data)
{
    // Set first half of shared mem to 0
    int idx = get_local_id(0);
    s_data[idx] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Set 2nd half to thread local sum
    idx += get_local_size(0);

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;     barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 1; i < SCAN_BLOCK_SIZE; i*=2) {
      t = s_data[idx - i];  barrier(CLK_LOCAL_MEM_FENCE);
      s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);
    }
    return s_data[idx-1];
}

__kernel void
scan(__global uint *g_odata,  __global uint* g_idata, 
     __global uint* g_blockSums, const int n,
     const int fullBlock, const int storeSum)
{
    __local uint s_data[2*SCAN_BLOCK_SIZE];

    // Load data into shared mem
    SORTVECTYPE tempData;
    SORTVECTYPE threadScanT;
    uint res;
    __global SORTVECTYPE* inData  = (__global SORTVECTYPE*) g_idata;

    const int gid = get_global_id(0);
    const int tid = get_local_id(0);
    const int i = gid * SORT_VECTOR;

    // If possible, read from global mem in a SORTVECTYPE chunk
#if SORT_VECTOR == 2
    if (fullBlock || i + 1 < n)
    {
        // scan the 2 elems read in from global
        tempData       = inData[gid];
        threadScanT.x = tempData.x;
        threadScanT.y = tempData.y + threadScanT.x;
        res = threadScanT.y;
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.x = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.x;
        res = threadScanT.y;
    }
#elif SORT_VECTOR == 4
    if (fullBlock || i + 3 < n)
    {
        // scan the 4 elems read in from global
        tempData       = inData[gid];
        threadScanT.x = tempData.x;
        threadScanT.y = tempData.y + threadScanT.x;
        threadScanT.z = tempData.z + threadScanT.y;
        threadScanT.w = tempData.w + threadScanT.z;
        res = threadScanT.w;
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.x = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.x;
        threadScanT.z = ((i+2 < n) ? g_idata[i+2] : 0.0f) + threadScanT.y;
        threadScanT.w = ((i+3 < n) ? g_idata[i+3] : 0.0f) + threadScanT.z;
        res = threadScanT.w;
    }
#elif SORT_VECTOR == 8
    if (fullBlock || i + 7 < n)
    {
        // scan the 8 elems read in from global
        tempData       = inData[gid];
        threadScanT.s0 = tempData.s0;
        threadScanT.s1 = tempData.s1 + threadScanT.s0;
        threadScanT.s2 = tempData.s2 + threadScanT.s1;
        threadScanT.s3 = tempData.s3 + threadScanT.s2;
        threadScanT.s4 = tempData.s4 + threadScanT.s3;
        threadScanT.s5 = tempData.s5 + threadScanT.s4;
        threadScanT.s6 = tempData.s6 + threadScanT.s5;
        threadScanT.s7 = tempData.s7 + threadScanT.s6;
        res = threadScanT.s7;
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.s0 = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.s1 = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.s0;
        threadScanT.s2 = ((i+2 < n) ? g_idata[i+2] : 0.0f) + threadScanT.s1;
        threadScanT.s3 = ((i+3 < n) ? g_idata[i+3] : 0.0f) + threadScanT.s2;
        threadScanT.s4 = ((i+4 < n) ? g_idata[i+4] : 0.0f) + threadScanT.s3;
        threadScanT.s5 = ((i+5 < n) ? g_idata[i+5] : 0.0f) + threadScanT.s4;
        threadScanT.s6 = ((i+6 < n) ? g_idata[i+6] : 0.0f) + threadScanT.s5;
        threadScanT.s7 = ((i+7 < n) ? g_idata[i+7] : 0.0f) + threadScanT.s6;
        res = threadScanT.s7;
    }
#endif

    res = scanLocalMem(res, s_data);
    barrier(CLK_LOCAL_MEM_FENCE);

    // If we have to store the sum for the block, have the last work item
    // in the block write it out
    if (storeSum && tid == get_local_size(0)-1) {
#if SORT_VECTOR == 2
        g_blockSums[get_group_id(0)] = res + threadScanT.y;
#elif SORT_VECTOR == 4
        g_blockSums[get_group_id(0)] = res + threadScanT.w;
#elif SORT_VECTOR == 8
        g_blockSums[get_group_id(0)] = res + threadScanT.s7;
#endif
    }

    // write results to global memory
    __global SORTVECTYPE* outData = (__global SORTVECTYPE*) g_odata;

#if SORT_VECTOR == 2
    tempData.x = res;
    tempData.y = res + threadScanT.x;

    if (fullBlock || i + 1 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.x; }
    }
#elif SORT_VECTOR == 4
    tempData.x = res;
    tempData.y = res + threadScanT.x;
    tempData.z = res + threadScanT.y;
    tempData.w = res + threadScanT.z;

    if (fullBlock || i + 3 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.x;
        if ((i+1) < n) { g_odata[i+1] = tempData.y;
        if ((i+2) < n) { g_odata[i+2] = tempData.z; } } }
    }
#elif SORT_VECTOR == 8
    tempData.s0 = res;
    tempData.s1 = res + threadScanT.s0;
    tempData.s2 = res + threadScanT.s1;
    tempData.s3 = res + threadScanT.s2;
    tempData.s4 = res + threadScanT.s3;
    tempData.s5 = res + threadScanT.s4;
    tempData.s6 = res + threadScanT.s5;
    tempData.s7 = res + threadScanT.s6;

    if (fullBlock || i + 7 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.s0;
        if ((i+1) < n) { g_odata[i+1] = tempData.s1;
        if ((i+2) < n) { g_odata[i+2] = tempData.s2;
        if ((i+3) < n) { g_odata[i+3] = tempData.s3;
        if ((i+4) < n) { g_odata[i+4] = tempData.s4;
        if ((i+5) < n) { g_odata[i+5] = tempData.s5;
        if ((i+6) < n) { g_odata[i+6] = tempData.s6; } } } } } } }
    }
#endif
}

__kernel void
vectorAddUniform4(__global uint *d_vector, __global const uint *d_uniforms, 
        const int n)
{
    __local uint uni[1];

    if (get_local_id(0) == 0)
    {
        uni[0] = d_uniforms[get_group_id(0)];
    }

    unsigned int address = get_local_id(0) + (get_group_id(0) *
            get_local_size(0) * SORT_VECTOR);

    barrier(CLK_LOCAL_MEM_FENCE);

    // SORT_VECTOR elems per thread
    for (int i = 0; i < SORT_VECTOR && address < n; i++)
    {
        d_vector[address] += uni[0];
        address += get_local_size(0);
    }
}

