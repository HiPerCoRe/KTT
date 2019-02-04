// This kernel code based on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

typedef unsigned int uint;
__device__ uint scanLSB(const uint val, uint* s_data)
{
    // Shared mem is 256 uints long, set first half to 0's
    int idx = threadIdx.x;
    s_data[idx] = 0;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += blockDim.x; // += 128 in this case

    // Unrolled scan in local memory

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
    /*t = s_data[idx -  2];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  4];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  8];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 16];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 32];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 64];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 128]; __syncthreads();
    s_data[idx] +=t;       __syncthreads();
*/
    return s_data[idx] - val;  // convert inclusive -> exclusive
}

__device__ uint4 scan4(uint4 idata, uint* ptr)
{
    uint4 val4 = idata;
    uint4 sum;

    // Scan the 4 elements in idata within this thread
    sum.x = val4.x;
    sum.y = val4.y + sum.x;
    sum.z = val4.z + sum.y;
    uint val = val4.w + sum.z;

    // Now scan those sums across the local work group
    val = scanLSB(val, ptr);

    val4.x = val;
    val4.y = val + sum.x;
    val4.z = val + sum.y;
    val4.w = val + sum.z;

    return val4;
}

//----------------------------------------------------------------------------
//
// radixSortBlocks sorts all blocks of data independently in shared
// memory.  Each thread block (CTA) sorts one block of 4*CTA_SIZE elements
//
// The radix sort is done in two stages.  This stage calls radixSortBlock
// on each block independently, sorting on the basis of bits
// (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------

extern "C" __global__ void radixSortBlocks(const uint nbits, const uint startbit,
                              uint4* keysOut, uint4* valuesOut,
                              uint4* keysIn,  uint4* valuesIn)
{
    __shared__ uint sMem[4*SORT_BLOCK_SIZE]; //512

    // Get Indexing information
    const uint i = threadIdx.x + (blockIdx.x * blockDim.x);
    const uint tid = threadIdx.x;
    const uint localSize = blockDim.x;

    // Load keys and vals from global memory
    uint4 key, value;
    key = keysIn[i];
    value = valuesIn[i];
    //if (i<10)printf("in %u %u-%u %u-%u %u-%u %u-%u\n", i, keysIn[i].x, valuesIn[i].x, keysIn[i].y, valuesIn[i].y, keysIn[i].z, valuesIn[i].z, keysIn[i].w, valuesIn[i].w);

    // For each of the 4 bits
    for(uint shift = startbit; shift < (startbit + nbits); ++shift)
    {
        // Check if the LSB is 0
        uint4 lsb;
        lsb.x = !((key.x >> shift) & 0x1);
        lsb.y = !((key.y >> shift) & 0x1);
        lsb.z = !((key.z >> shift) & 0x1);
        lsb.w = !((key.w >> shift) & 0x1);

        // Do an exclusive scan of how many elems have 0's in the LSB
        // When this is finished, address.n will contain the number of
        // elems with 0 in the LSB which precede elem n
        uint4 address = scan4(lsb, sMem);

        __shared__ uint numtrue;

        // Store the total number of elems with an LSB of 0
        // to shared mem
        if (tid == localSize - 1)
        {
            numtrue = address.w + lsb.w;
        }
        __syncthreads();

        // Determine rank -- position in the block
        // If you are a 0 --> your position is the scan of 0's
        // If you are a 1 --> your position is calculated as below
        uint4 rank;
        const int idx = tid*4;
        rank.x = lsb.x ? address.x : numtrue + idx     - address.x;
        rank.y = lsb.y ? address.y : numtrue + idx + 1 - address.y;
        rank.z = lsb.z ? address.z : numtrue + idx + 2 - address.z;
        rank.w = lsb.w ? address.w : numtrue + idx + 3 - address.w;

        // Scatter keys into local mem
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = key.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = key.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = key.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = key.w;
        __syncthreads();

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
        key.x = sMem[tid];
        key.y = sMem[tid +     localSize];
        key.z = sMem[tid + 2 * localSize];
        key.w = sMem[tid + 3 * localSize];
        __syncthreads();

        // Scatter values into local mem
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = value.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = value.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = value.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = value.w;
        __syncthreads();

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
        value.x = sMem[tid];
        value.y = sMem[tid +     localSize];
        value.z = sMem[tid + 2 * localSize];
        value.w = sMem[tid + 3 * localSize];
        __syncthreads();
    }
    keysOut[i]   = key;
    valuesOut[i] = value;
    //if (i<10)printf("out %u %u-%u %u-%u %u-%u %u-%u\n", i, keysOut[i].x, valuesOut[i].x, keysOut[i].y, valuesOut[i].y, keysOut[i].z, valuesOut[i].z, keysOut[i].w, valuesOut[i].w);

}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each
// block counts the number of keys that fall into each radix in the group, and
// finds the starting offset of each radix in the block.  It then writes the
// radix counts to the counters array, and the starting offsets to the
// blockOffsets array.
//
//----------------------------------------------------------------------------
extern "C" __global__ void findRadixOffsets(uint2* keys, uint* counters,
        uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks)
{
    __shared__ uint  sStartPointers[16];
    __shared__ uint sRadix1[2*SCAN_BLOCK_SIZE];

    uint gid = blockIdx.x;
    uint tid = threadIdx.x;
    uint localSize = blockDim.x;
    uint i = threadIdx.x + (blockIdx.x * blockDim.x);

    uint2 radix2;
    radix2 = keys[i];

    sRadix1[2 * tid]     = (radix2.x >> startbit) & 0xF;
    sRadix1[2 * tid + 1] = (radix2.y >> startbit) & 0xF;

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
        if (gid == 0) printf("find radix %u - %u\n", sRadix1[tid], sStartPointers[sRadix1[tid]]);
    }
    if(sRadix1[tid + localSize] != sRadix1[tid + localSize - 1])
    {
        sStartPointers[sRadix1[tid + localSize]] = tid + localSize;
        if (gid == 0) printf("find radix %u - %u\n", sRadix1[tid+localSize], sStartPointers[sRadix1[tid+localSize]]);
    }
    __syncthreads();

    if(tid < 16)
    {
        blockOffsets[gid*16 + tid] = sStartPointers[tid];
        if (gid*16+tid < 10) printf("blockOffesets %u %u\n", gid*16 + tid, blockOffsets[gid*16+tid]);
    }
    __syncthreads();

    // Compute the sizes of each block.
    if((tid > 0) && (sRadix1[tid] != sRadix1[tid - 1]) )
    {
        sStartPointers[sRadix1[tid - 1]] =
            tid - sStartPointers[sRadix1[tid - 1]];
        if (gid == 0) printf("find radix size %u - %u\n", sRadix1[tid-1], sStartPointers[sRadix1[tid-1]]);
    }
    if(sRadix1[tid + localSize] != sRadix1[tid + localSize - 1] )
    {
        sStartPointers[sRadix1[tid + localSize - 1]] =
            tid + localSize - sStartPointers[sRadix1[tid +
                                                         localSize - 1]];
    }

    if(tid == localSize - 1)
    {
        sStartPointers[sRadix1[2 * localSize - 1]] =
            2 * localSize - sStartPointers[sRadix1[2 * localSize - 1]];
    }
    __syncthreads();

    if(tid < 16)
    {
        counters[tid * totalBlocks + gid] = sStartPointers[tid];
        if (tid*totalBlocks+gid < 10) printf("counters %u %u\n", tid*totalBlocks+gid, counters[tid*totalBlocks+gid]);
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
                            uint2 *keys,
                            uint2 *values,
                            uint  *blockOffsets,
                            uint  *offsets,
                            uint  *sizes,
                            uint  totalBlocks)
{
    __shared__ uint2 sKeys2[SCAN_BLOCK_SIZE];
    __shared__ uint2 sValues2[SCAN_BLOCK_SIZE];
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
        if (gid == 0) printf("reorder %u offsets %u blockOffsets %u\n", tid, sOffsets[tid], sBlockOffsets[tid]);
    }
    __syncthreads();

    uint radix = (sKeys1[tid] >> startbit) & 0xF;
    uint globalOffset = sOffsets[radix] + tid - sBlockOffsets[radix];
    if (i<10) printf("reorder i-%u radix-%u globalOffset %u = offsets %u + thread.x %u -  blockoffsets %u. %u %u\n", i, radix, globalOffset, sOffsets[radix], tid, sBlockOffsets[radix], sKeys1[tid], sValues1[tid]);

    outKeys[globalOffset]   = sKeys1[tid];
    outValues[globalOffset] = sValues1[tid];

    radix = (sKeys1[tid + localSize] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + tid + localSize -
                   sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[tid + localSize];
    outValues[globalOffset] = sValues1[tid + localSize];
    if (i<10) printf("reorder i-%u radix-%u offsets-%u blockoffsets-%u globaloffset - %u %u-%u %u-%u\n", i, radix, sOffsets[radix], sBlockOffsets[radix], globalOffset, sKeys1[tid+localSize], outKeys[globalOffset], sValues1[tid+localSize], outValues[globalOffset]);
    if (i<10) printf("reorder out %u %u-%u %u-%u %u-%u\n", i, keys[i].x, values[i].x, keys[i].y, values[i].y, outKeys[i], outValues[i]);


}

__device__ uint scanLocalMem(const uint val, uint* s_data)
{
    // Shared mem is 512 uints long, set first half to 0
    int idx = threadIdx.x;
    s_data[idx] = 0.0f;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += blockDim.x; // += 256

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
    /*t = s_data[idx -  2];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  4];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  8];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 16];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 32];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 64];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 128]; __syncthreads();
    s_data[idx] += t;      __syncthreads();
    */

    return s_data[idx-1];
}

extern "C" __global__ void
scan(uint *g_odata, uint* g_idata, uint* g_blockSums, const int n,
     const bool fullBlock, const bool storeSum)
{
    __shared__ uint s_data[2*SCAN_BLOCK_SIZE];

    // Load data into shared mem
    uint4 tempData;
    uint4 threadScanT;
    uint res;
    uint4* inData  = (uint4*) g_idata;

    const int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;
    const int i = gid * 4;
    //if (gid < 10) printf("scan in %u %u %u %u %u\n", gid, inData[gid].x, inData[gid].y, inData[gid].z, inData[gid].w);

    // If possible, read from global mem in a uint4 chunk
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
    //if (i < 20) printf("res %u %u\n", i, res);

    res = scanLocalMem(res, s_data);
    __syncthreads();
    //if (i < 20) printf("res out %u %u\n", i, res);

    // If we have to store the sum for the block, have the last work item
    // in the block write it out
    if (storeSum && tid == blockDim.x-1) {
        g_blockSums[blockIdx.x] = res + threadScanT.w;
        //printf("blockSums out %u %u \n", blockIdx.x, g_blockSums[blockIdx.x]);
    }

    // write results to global memory
    uint4* outData = (uint4*) g_odata;

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
    //if (gid < 10) printf("scan out %d %u %u %u %u %u\n", n, gid, outData[gid].x, outData[gid].y, outData[gid].z, outData[gid].w);
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
            blockDim.x * 4);

    __syncthreads();

    // 4 elems per thread
    for (int i = 0; i < 4 && address < n; i++)
    {
        d_vector[address] += uni[0];
    //if (threadIdx.x == 0) printf("vector out %u <> %u, %u %u\n", address, n, uni[0], d_vector[address]);
        address += blockDim.x;
    }
}
