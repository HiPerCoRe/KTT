// This kernel code based on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

typedef unsigned int uint;

typedef struct __builtin_align__(16) {
    uint4 a;
    uint4 b;
  } my_uint8;

#if SORT_VECTOR == 2
#define SORTVECTYPE uint2
#elif SORT_VECTOR == 4
#define SORTVECTYPE uint4
#elif SORT_VECTOR == 8
#define SORTVECTYPE my_uint8
#endif

#if SCAN_VECTOR == 2
#define SCANVECTYPE uint2
#elif SCAN_VECTOR == 4
#define SCANVECTYPE uint4
#elif SCAN_VECTOR == 8
#define SCANVECTYPE my_uint8
#endif

__device__ uint scanLSB(const uint val, uint* s_data)
{
    // Set first half of shared mem to 0's
    int idx = threadIdx.x;
    s_data[idx] = 0;
    __syncthreads();

    // Set 2nd half to thread local sum
    idx += blockDim.x;

    // scan in local memory

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;     __syncthreads();
    for (uint i = 1; i < SORT_BLOCK_SIZE; i*=2) {
      t = s_data[idx - i];  __syncthreads();
      s_data[idx] += t;      __syncthreads();
    }
    return s_data[idx] - val;  // convert inclusive -> exclusive
}

__device__ SORTVECTYPE scan4(SORTVECTYPE idata, uint* ptr)
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
    sum.a.x = val4.a.x;
    sum.a.y = val4.a.y + sum.a.x;
    sum.a.z = val4.a.z + sum.a.y;
    sum.a.w = val4.a.w + sum.a.z;
    sum.b.x = val4.b.x + sum.a.w;
    sum.b.y = val4.b.y + sum.b.x;
    sum.b.z = val4.b.z + sum.b.y;
    uint val = val4.b.w + sum.b.z;
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
    val4.a.x = val;
    val4.a.y = val + sum.a.x;
    val4.a.z = val + sum.a.y;
    val4.a.w = val + sum.a.z;
    val4.b.x = val + sum.a.w;
    val4.b.y = val + sum.b.x;
    val4.b.z = val + sum.b.y;
    val4.b.w = val + sum.b.z;
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

extern "C" __global__ void radixSortBlocks(const uint nbits, const uint startbit,
                              SORTVECTYPE* keysOut, SORTVECTYPE* valuesOut,
                              SORTVECTYPE* keysIn,  SORTVECTYPE* valuesIn)
{
    __shared__ uint sMem[SORT_VECTOR*SORT_BLOCK_SIZE];

    // Get Indexing information
    const uint i = threadIdx.x + (blockIdx.x * blockDim.x);
    const uint tid = threadIdx.x;
    const uint localSize = blockDim.x;

    // Load keys and vals from global memory
    SORTVECTYPE key, value;
    key = keysIn[i];
    value = valuesIn[i];

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
        lsb.a.x = !((key.a.x >> shift) & 0x1);
        lsb.a.y = !((key.a.y >> shift) & 0x1);
        lsb.a.z = !((key.a.z >> shift) & 0x1);
        lsb.a.w = !((key.a.w >> shift) & 0x1);
        lsb.b.x = !((key.b.x >> shift) & 0x1);
        lsb.b.y = !((key.b.y >> shift) & 0x1);
        lsb.b.z = !((key.b.z >> shift) & 0x1);
        lsb.b.w = !((key.b.w >> shift) & 0x1);
#endif

        // Do an exclusive scan of how many elems have 0's in the LSB
        // When this is finished, address.n will contain the number of
        // elems with 0 in the LSB which precede elem n
        SORTVECTYPE address = scan4(lsb, sMem);

        __shared__ uint numtrue;

        // Store the total number of elems with an LSB of 0
        // to shared mem
        if (tid == localSize - 1)
        {
#if SORT_VECTOR == 2
          numtrue = address.y + lsb.y;
#elif SORT_VECTOR == 4
            numtrue = address.w + lsb.w;
#elif SORT_VECTOR == 8
            numtrue = address.b.w + lsb.b.w;
#endif
        }
        __syncthreads();

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
        rank.a.x = lsb.a.x ? address.a.x : numtrue + idx     - address.a.x;
        rank.a.y = lsb.a.y ? address.a.y : numtrue + idx + 1 - address.a.y;
        rank.a.z = lsb.a.z ? address.a.z : numtrue + idx + 2 - address.a.z;
        rank.a.w = lsb.a.w ? address.a.w : numtrue + idx + 3 - address.a.w;
        rank.b.x = lsb.b.x ? address.b.x : numtrue + idx + 4 - address.b.x;
        rank.b.y = lsb.b.y ? address.b.y : numtrue + idx + 5 - address.b.y;
        rank.b.z = lsb.b.z ? address.b.z : numtrue + idx + 6 - address.b.z;
        rank.b.w = lsb.b.w ? address.b.w : numtrue + idx + 7 - address.b.w;
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
        sMem[(rank.a.x & 7) * localSize + (rank.a.x >> 3)] = key.a.x;
        sMem[(rank.a.y & 7) * localSize + (rank.a.y >> 3)] = key.a.y;
        sMem[(rank.a.z & 7) * localSize + (rank.a.z >> 3)] = key.a.z;
        sMem[(rank.a.w & 7) * localSize + (rank.a.w >> 3)] = key.a.w;
        sMem[(rank.b.x & 7) * localSize + (rank.b.x >> 3)] = key.b.x;
        sMem[(rank.b.y & 7) * localSize + (rank.b.y >> 3)] = key.b.y;
        sMem[(rank.b.z & 7) * localSize + (rank.b.z >> 3)] = key.b.z;
        sMem[(rank.b.w & 7) * localSize + (rank.b.w >> 3)] = key.b.w;
#endif
        __syncthreads();

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
        key.a.x = sMem[tid];
        key.a.y = sMem[tid +     localSize];
        key.a.z = sMem[tid + 2 * localSize];
        key.a.w = sMem[tid + 3 * localSize];
        key.b.x = sMem[tid + 4 * localSize];
        key.b.y = sMem[tid + 5 * localSize];
        key.b.z = sMem[tid + 6 * localSize];
        key.b.w = sMem[tid + 7 * localSize];
#endif
        __syncthreads();

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
        sMem[(rank.a.x & 7) * localSize + (rank.a.x >> 3)] = value.a.x;
        sMem[(rank.a.y & 7) * localSize + (rank.a.y >> 3)] = value.a.y;
        sMem[(rank.a.z & 7) * localSize + (rank.a.z >> 3)] = value.a.z;
        sMem[(rank.a.w & 7) * localSize + (rank.a.w >> 3)] = value.a.w;
        sMem[(rank.b.x & 7) * localSize + (rank.b.x >> 3)] = value.b.x;
        sMem[(rank.b.y & 7) * localSize + (rank.b.y >> 3)] = value.b.y;
        sMem[(rank.b.z & 7) * localSize + (rank.b.z >> 3)] = value.b.z;
        sMem[(rank.b.w & 7) * localSize + (rank.b.w >> 3)] = value.b.w;
#endif
        __syncthreads();

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
        value.a.x = sMem[tid];
        value.a.y = sMem[tid +     localSize];
        value.a.z = sMem[tid + 2 * localSize];
        value.a.w = sMem[tid + 3 * localSize];
        value.b.x = sMem[tid + 4 * localSize];
        value.b.y = sMem[tid + 5 * localSize];
        value.b.z = sMem[tid + 6 * localSize];
        value.b.w = sMem[tid + 7 * localSize];
#endif
        __syncthreads();
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
extern "C" __global__ void findRadixOffsets(SCANVECTYPE* keys, uint* counters,
        uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks)
{
    __shared__ uint  sStartPointers[16];
    __shared__ uint sRadix1[SCAN_VECTOR*SCAN_BLOCK_SIZE];

    uint gid = blockIdx.x;
    uint tid = threadIdx.x;
    uint localSize = blockDim.x;
    uint i = threadIdx.x + (blockIdx.x * blockDim.x);

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
    sRadix1[8 * tid]      = (radix2.a.x >> startbit) & 0xF;
    sRadix1[8 * tid + 1]  = (radix2.a.y >> startbit) & 0xF;
    sRadix1[8 * tid + 2]  = (radix2.a.z >> startbit) & 0xF;
    sRadix1[8 * tid + 3]  = (radix2.a.w >> startbit) & 0xF;
    sRadix1[8 * tid + 4]  = (radix2.b.x >> startbit) & 0xF;
    sRadix1[8 * tid + 5]  = (radix2.b.y >> startbit) & 0xF;
    sRadix1[8 * tid + 6]  = (radix2.b.z >> startbit) & 0xF;
    sRadix1[8 * tid + 7]  = (radix2.b.w >> startbit) & 0xF;
#endif

    // Finds the position where the sRadix1 entries differ and stores start
    // index for each radix.
    if(tid < 16)
    {
        sStartPointers[tid] = 0;
    }
    __syncthreads();

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
    __syncthreads();

    if(tid < 16)
    {
        blockOffsets[gid*16 + tid] = sStartPointers[tid];
    }
    __syncthreads();

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
    __syncthreads();

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
extern "C" __global__ void reorderData(uint  startbit,
                            uint  *outKeys,
                            uint  *outValues,
                            SCANVECTYPE *keys,
                            SCANVECTYPE *values,
                            uint  *blockOffsets,
                            uint  *offsets,
                            uint  *sizes,
                            uint  totalBlocks)
{
    __shared__ SCANVECTYPE sKeys2[SCAN_BLOCK_SIZE];
    __shared__ SCANVECTYPE sValues2[SCAN_BLOCK_SIZE];
    __shared__ uint  sOffsets[16];
    __shared__ uint  sBlockOffsets[16];
    uint* sKeys1   = (uint*) sKeys2;
    uint* sValues1 = (uint*) sValues2;

    uint gid = blockIdx.x;
    uint tid = threadIdx.x;
    uint localSize = blockDim.x;
    uint i = threadIdx.x + (blockIdx.x * blockDim.x);

    sKeys2[tid]   = keys[i];
    sValues2[tid] = values[i];

    if(tid < 16)
    {
        sOffsets[tid]      = offsets[tid * totalBlocks +
                                             gid];
        sBlockOffsets[tid] = blockOffsets[gid * 16 + tid];
    }
    __syncthreads();

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

__device__ uint scanLocalMem(const uint val, uint* s_data)
{
    // Set first half of shared mem to 0
    int idx = threadIdx.x;
    s_data[idx] = 0.0f;
    __syncthreads();

    // Set 2nd half to thread local sum
    idx += blockDim.x;

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;     __syncthreads();
    for (uint i = 1; i < SCAN_BLOCK_SIZE; i*=2) {
      t = s_data[idx - i];  __syncthreads();
      s_data[idx] += t;      __syncthreads();
    }
    return s_data[idx-1];
}

extern "C" __global__ void
scan(uint *g_odata, uint* g_idata, uint* g_blockSums, const int n,
     const bool fullBlock, const bool storeSum)
{
    __shared__ uint s_data[2*SCAN_BLOCK_SIZE];

    // Load data into shared mem
    SORTVECTYPE tempData;
    SORTVECTYPE threadScanT;
    uint res;
    SORTVECTYPE* inData  = (SORTVECTYPE*) g_idata;

    const int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;
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
        threadScanT.a.x = tempData.a.x;
        threadScanT.a.y = tempData.a.y + threadScanT.a.x;
        threadScanT.a.z = tempData.a.z + threadScanT.a.y;
        threadScanT.a.w = tempData.a.w + threadScanT.a.z;
        threadScanT.b.x = tempData.b.x + threadScanT.a.w;
        threadScanT.b.y = tempData.b.y + threadScanT.b.x;
        threadScanT.b.z = tempData.b.z + threadScanT.b.y;
        threadScanT.b.w = tempData.b.w + threadScanT.b.z;
        res = threadScanT.b.w;
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.a.x = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.a.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.a.x;
        threadScanT.a.z = ((i+2 < n) ? g_idata[i+2] : 0.0f) + threadScanT.a.y;
        threadScanT.a.w = ((i+3 < n) ? g_idata[i+3] : 0.0f) + threadScanT.a.z;
        threadScanT.b.x = ((i+4 < n) ? g_idata[i+4] : 0.0f) + threadScanT.a.w;
        threadScanT.b.y = ((i+5 < n) ? g_idata[i+5] : 0.0f) + threadScanT.b.x;
        threadScanT.b.z = ((i+6 < n) ? g_idata[i+6] : 0.0f) + threadScanT.b.y;
        threadScanT.b.w = ((i+7 < n) ? g_idata[i+7] : 0.0f) + threadScanT.b.z;
        res = threadScanT.b.w;
    }
#endif

    res = scanLocalMem(res, s_data);
    __syncthreads();

    // If we have to store the sum for the block, have the last work item
    // in the block write it out
    if (storeSum && tid == blockDim.x-1) {
#if SORT_VECTOR == 2
        g_blockSums[blockIdx.x] = res + threadScanT.y;
#elif SORT_VECTOR == 4
        g_blockSums[blockIdx.x] = res + threadScanT.w;
#elif SORT_VECTOR == 8
        g_blockSums[blockIdx.x] = res + threadScanT.b.w;
#endif
    }

    // write results to global memory
    SORTVECTYPE* outData = (SORTVECTYPE*) g_odata;

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
    tempData.a.x = res;
    tempData.a.y = res + threadScanT.a.x;
    tempData.a.z = res + threadScanT.a.y;
    tempData.a.w = res + threadScanT.a.z;
    tempData.b.x = res + threadScanT.a.w;
    tempData.b.y = res + threadScanT.b.x;
    tempData.b.z = res + threadScanT.b.y;
    tempData.b.w = res + threadScanT.b.z;

    if (fullBlock || i + 7 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.a.x;
        if ((i+1) < n) { g_odata[i+1] = tempData.a.y;
        if ((i+2) < n) { g_odata[i+2] = tempData.a.z;
        if ((i+3) < n) { g_odata[i+3] = tempData.a.w;
        if ((i+4) < n) { g_odata[i+4] = tempData.b.x;
        if ((i+5) < n) { g_odata[i+5] = tempData.b.y;
        if ((i+6) < n) { g_odata[i+6] = tempData.b.z; } } } } } } }
    }
#endif
}

extern "C" __global__ void
vectorAddUniform4(uint *d_vector, const uint *d_uniforms, const int n)
{
    __shared__ uint uni[1];

    if (threadIdx.x == 0)
    {
        uni[0] = d_uniforms[blockIdx.x];
    }

    unsigned int address = threadIdx.x + (blockIdx.x *
            blockDim.x * SORT_VECTOR);

    __syncthreads();

    // SORT_VECTOR elems per thread
    for (int i = 0; i < SORT_VECTOR && address < n; i++)
    {
        d_vector[address] += uni[0];
        address += blockDim.x;
    }
}
