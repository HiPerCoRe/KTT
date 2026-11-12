typedef unsigned int uint;

#define FPTYPE uint
//#define FPVECTNUM 16 -- this will be defined by KTT as it is a tuning parameter
#define FPVECTYPE uint4


// Compute a per block histogram of the occurrences of each
// digit, using a 4-bit radix (i.e. 16 possible digits).
extern "C" __global__ void
reduce(const FPTYPE * in, //input vector of unsorted number
       FPTYPE * isums, //vector of all group histograms, global memory, size n_groups*n_digits
       const int n, //size of the input vector
       const int shift) //which 'digit' are we tackling now?
{
    __shared__ unsigned int lmem[LOCAL_SIZE];
    // First, calculate the bounds of the region of the array
    // that this block will sum.  We need these regions to match
    // perfectly with those in the bottom-level scan, so we index
    // as if vector types of length FPVECTNUM were in use.  This prevents
    // errors due to slightly misaligned regions.
    int region_size = ((n / FPVECTNUM) / gridDim.x) * FPVECTNUM;
    int block_start = blockIdx.x * region_size;

    // Give the last block any extra elements
    int block_stop  = (blockIdx.x == gridDim.x - 1) ?
        n : block_start + region_size;

    // Calculate starting index for this thread/work item
    int tid = threadIdx.x;
    int i = block_start + tid;

    // The per thread histogram, initially 0's.
    int digit_counts[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0 };

    // Reduce multiple elements per thread
    while (i < block_stop)
    {
        // This statement
        // 1) Loads the value in from global memory
        // 2) Shifts to the right to have the 4 bits of interest
        //    in the least significant places
        // 3) Masks any more significant bits away. This leaves us
        // with the relevant digit (which is also the index into the
        // histogram). Next increment the histogram to count this occurrence.
        digit_counts[(in[i] >> shift) & 0xFU]++;
        i += blockDim.x;
    }

    for (int d = 0; d < 16; d++)
    {
        // Load this thread's sum into local/shared memory
        lmem[tid] = digit_counts[d];
        __syncthreads();

        // Reduce the contents of shared/local memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                lmem[tid] += lmem[tid + s];
            }
            __syncthreads();
        }

        // Write result for this block to global memory
        if (tid == 0)
        {
            isums[(d * gridDim.x) + blockIdx.x] = lmem[0];
        }
    }
}

// This kernel scans the contents of local memory using a work
// inefficient, but highly parallel Kogge-Stone style scan.
// Set exclusive to 1 for an exclusive scan or 0 for an inclusive scan
__device__ inline FPTYPE scanLocalMem(FPTYPE val, FPTYPE* lmem, int exclusive)
{
    // Set first half of local memory to zero to make room for scanning
    int idx = threadIdx.x;
    lmem[idx] = 0;

    // Set second half to block sums from global memory, but don't go out
    // of bounds
    idx += blockDim.x;
    lmem[idx] = val;
    __syncthreads();

    // Now, perform Kogge-Stone scan
    FPTYPE t;
    for (int i = 1; i < blockDim.x; i *= 2)
    {
        t = lmem[idx -  i]; __syncthreads();
        lmem[idx] += t;     __syncthreads();
    }
    return lmem[idx-exclusive];
}

// This single group kernel takes the per block histograms
// from the reduction and performs an exclusive scan on them.
extern "C" __global__ void
top_scan(FPTYPE * isums, //vector of histograms of all groups, global memory, size n_groups*n_digits
         const int n) //number of groups
{
    __shared__ unsigned int lmem[LOCAL_SIZE*2];
    __shared__ int s_seed;
    s_seed = 0; __syncthreads();

    // Decide if this is the last thread that needs to
    // propagate the seed value
    int last_thread = (threadIdx.x < n &&
                      (threadIdx.x+1) == n) ? 1 : 0;

    for (int d = 0; d < 16; d++)
    {
        FPTYPE val = 0;
        // Load each block's count for digit d from global memory
        if (threadIdx.x < n)
        {
            val = isums[(n * d) + threadIdx.x];
        }
        // Exclusive scan the counts in local memory
        FPTYPE res = scanLocalMem(val, lmem, 1);
        // Write scanned value out to global
        //this constructs the seed vector of indices for this digit per group
        if (threadIdx.x < n)
        {
            isums[(n * d) + threadIdx.x] = res + s_seed;
        }
        __syncthreads();

        if (last_thread)
        {
            s_seed += res + val; //how should the next digit's index be shifted?
        }
        __syncthreads();
    }
}


extern "C" __global__ void
bottom_scan(const FPTYPE * in, //numbers to be sorted
            const FPTYPE * isums, //vector of indices used for sorting
            FPTYPE * out, //(partially) sorted vector
            const int n, //size of the in and out vector
            const int shift)
{
    __shared__ unsigned int lmem[LOCAL_SIZE*2];
    // Use local memory to cache the scanned seeds from the global isums
    __shared__ FPTYPE l_scanned_seeds[16];

    //We will need sort of partial seed vectors, as we have not saved them into global memory, we need to recompute those.
    // Keep a shared histogram of all instances seen by the current
    // block
    __shared__ FPTYPE l_block_counts[16];

    // Keep a private histogram as well
    int histogram[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0  };

    // Prepare for reading 4-element vectors
    // Assume n is divisible by 4
    // with vectors of different length, the same applies for 8 and 16
    FPVECTYPE *in4  = (FPVECTYPE*) in;
    FPVECTYPE *out4 = (FPVECTYPE*) out;
    int n4 = n / FPVECTNUM; //vector type is 4 wide

    int region_size = n4 / gridDim.x;
    int block_start = blockIdx.x * region_size;
    // Give the last block any extra elements
    int block_stop  = (blockIdx.x == gridDim.x - 1) ?
        n4 : block_start + region_size;

    // Calculate starting index for this thread/work item
    int i = block_start + threadIdx.x;
    int window = block_start;
    

    // Set the histogram in local memory to zero
    // and read in the scanned seeds from gmem
    if (threadIdx.x < 16)
    {
        l_block_counts[threadIdx.x] = 0;
        l_scanned_seeds[threadIdx.x] =
            isums[(threadIdx.x*gridDim.x)+blockIdx.x];
    }
    __syncthreads();

    // Scan multiple elements per thread
    while (window < block_stop)
    {
        // Reset histogram
        for (int q = 0; q < 16; q++) histogram[q] = 0;
        FPVECTYPE val_4;
        FPVECTYPE key_4;

        if (i < block_stop) // Make sure we don't read out of bounds
        {
            val_4 = in4[i];
#if FPVECTNUM == 4
            // Mask the keys to get the appropriate digit
            key_4.x = (val_4.x >> shift) & 0xFU;
            key_4.y = (val_4.y >> shift) & 0xFU;
            key_4.z = (val_4.z >> shift) & 0xFU;
            key_4.w = (val_4.w >> shift) & 0xFU;

            // Update the histogram
            histogram[key_4.x]++;
            histogram[key_4.y]++;
            histogram[key_4.z]++;
            histogram[key_4.w]++;

#elif FPVECTNUM == 8
            // Mask the keys to get the appropriate digit
            key_4.s0 = (val_4.s0 >> shift) & 0xFU;
            key_4.s1 = (val_4.s1 >> shift) & 0xFU;
            key_4.s2 = (val_4.s2 >> shift) & 0xFU;
            key_4.s3 = (val_4.s3 >> shift) & 0xFU;
            key_4.s4 = (val_4.s4 >> shift) & 0xFU;
            key_4.s5 = (val_4.s5 >> shift) & 0xFU;
            key_4.s6 = (val_4.s6 >> shift) & 0xFU;
            key_4.s7 = (val_4.s7 >> shift) & 0xFU;

            // Update the histogram
            histogram[key_4.s0]++;
            histogram[key_4.s1]++;
            histogram[key_4.s2]++;
            histogram[key_4.s3]++;
            histogram[key_4.s4]++;
            histogram[key_4.s5]++;
            histogram[key_4.s6]++;
            histogram[key_4.s7]++;

#elif FPVECTNUM == 16
            // Mask the keys to get the appropriate digit
            key_4.s0 = (val_4.s0 >> shift) & 0xFU;
            key_4.s1 = (val_4.s1 >> shift) & 0xFU;
            key_4.s2 = (val_4.s2 >> shift) & 0xFU;
            key_4.s3 = (val_4.s3 >> shift) & 0xFU;
            key_4.s4 = (val_4.s4 >> shift) & 0xFU;
            key_4.s5 = (val_4.s5 >> shift) & 0xFU;
            key_4.s6 = (val_4.s6 >> shift) & 0xFU;
            key_4.s7 = (val_4.s7 >> shift) & 0xFU;
            key_4.s8 = (val_4.s8 >> shift) & 0xFU;
            key_4.s9 = (val_4.s9 >> shift) & 0xFU;
            key_4.sa = (val_4.sa >> shift) & 0xFU;
            key_4.sb = (val_4.sb >> shift) & 0xFU;
            key_4.sc = (val_4.sc >> shift) & 0xFU;
            key_4.sd = (val_4.sd >> shift) & 0xFU;
            key_4.se = (val_4.se >> shift) & 0xFU;
            key_4.sf = (val_4.sf >> shift) & 0xFU;

            // Update the histogram
            histogram[key_4.s0]++;
            histogram[key_4.s1]++;
            histogram[key_4.s2]++;
            histogram[key_4.s3]++;
            histogram[key_4.s4]++;
            histogram[key_4.s5]++;
            histogram[key_4.s6]++;
            histogram[key_4.s7]++;
            histogram[key_4.s8]++;
            histogram[key_4.s9]++;
            histogram[key_4.sa]++;
            histogram[key_4.sb]++;
            histogram[key_4.sc]++;
            histogram[key_4.sd]++;
            histogram[key_4.se]++;
            histogram[key_4.sf]++;
#endif
        }

        // Scan the digit counts in local memory
        for (int digit = 0; digit < 16; digit++)
        {
        // recomputing partial seed vector from per thread histograms
            histogram[digit] = scanLocalMem(histogram[digit], lmem, 1);
            __syncthreads();
        }

        if (i < block_stop) // Make sure we don't write out of bounds
        {
            int address;
#if FPVECTNUM == 4
            address = histogram[key_4.x] + l_scanned_seeds[key_4.x] + l_block_counts[key_4.x];
            out[address] = val_4.x;
            histogram[key_4.x]++;

            address = histogram[key_4.y] + l_scanned_seeds[key_4.y] + l_block_counts[key_4.y];
            out[address] = val_4.y;
            histogram[key_4.y]++;

            address = histogram[key_4.z] + l_scanned_seeds[key_4.z] + l_block_counts[key_4.z];
            out[address] = val_4.z;
            histogram[key_4.z]++;

            address = histogram[key_4.w] + l_scanned_seeds[key_4.w] + l_block_counts[key_4.w];
            out[address] = val_4.w;
            histogram[key_4.w]++;

#elif FPVECTNUM == 8
            address = histogram[key_4.s0] + l_scanned_seeds[key_4.s0] + l_block_counts[key_4.s0];
            out[address] = val_4.s0;
            histogram[key_4.s0]++;

            address = histogram[key_4.s1] + l_scanned_seeds[key_4.s1] + l_block_counts[key_4.s1];
            out[address] = val_4.s1;
            histogram[key_4.s1]++;

            address = histogram[key_4.s2] + l_scanned_seeds[key_4.s2] + l_block_counts[key_4.s2];
            out[address] = val_4.s2;
            histogram[key_4.s2]++;

            address = histogram[key_4.s3] + l_scanned_seeds[key_4.s3] + l_block_counts[key_4.s3];
            out[address] = val_4.s3;
            histogram[key_4.s3]++;

            address = histogram[key_4.s4] + l_scanned_seeds[key_4.s4] + l_block_counts[key_4.s4];
            out[address] = val_4.s4;
            histogram[key_4.s4]++;

            address = histogram[key_4.s5] + l_scanned_seeds[key_4.s5] + l_block_counts[key_4.s5];
            out[address] = val_4.s5;
            histogram[key_4.s5]++;

            address = histogram[key_4.s6] + l_scanned_seeds[key_4.s6] + l_block_counts[key_4.s6];
            out[address] = val_4.s6;
            histogram[key_4.s6]++;

            address = histogram[key_4.s7] + l_scanned_seeds[key_4.s7] + l_block_counts[key_4.s7];
            out[address] = val_4.s7;
            histogram[key_4.s7]++;

#elif FPVECTNUM == 16
            address = histogram[key_4.s0] + l_scanned_seeds[key_4.s0] + l_block_counts[key_4.s0];
            out[address] = val_4.s0;
            histogram[key_4.s0]++;

            address = histogram[key_4.s1] + l_scanned_seeds[key_4.s1] + l_block_counts[key_4.s1];
            out[address] = val_4.s1;
            histogram[key_4.s1]++;

            address = histogram[key_4.s2] + l_scanned_seeds[key_4.s2] + l_block_counts[key_4.s2];
            out[address] = val_4.s2;
            histogram[key_4.s2]++;

            address = histogram[key_4.s3] + l_scanned_seeds[key_4.s3] + l_block_counts[key_4.s3];
            out[address] = val_4.s3;
            histogram[key_4.s3]++;

            address = histogram[key_4.s4] + l_scanned_seeds[key_4.s4] + l_block_counts[key_4.s4];
            out[address] = val_4.s4;
            histogram[key_4.s4]++;

            address = histogram[key_4.s5] + l_scanned_seeds[key_4.s5] + l_block_counts[key_4.s5];
            out[address] = val_4.s5;
            histogram[key_4.s5]++;

            address = histogram[key_4.s6] + l_scanned_seeds[key_4.s6] + l_block_counts[key_4.s6];
            out[address] = val_4.s6;
            histogram[key_4.s6]++;

            address = histogram[key_4.s7] + l_scanned_seeds[key_4.s7] + l_block_counts[key_4.s7];
            out[address] = val_4.s7;
            histogram[key_4.s7]++;

            address = histogram[key_4.s8] + l_scanned_seeds[key_4.s8] + l_block_counts[key_4.s8];
            out[address] = val_4.s8;
            histogram[key_4.s8]++;

            address = histogram[key_4.s9] + l_scanned_seeds[key_4.s9] + l_block_counts[key_4.s9];
            out[address] = val_4.s9;
            histogram[key_4.s9]++;

            address = histogram[key_4.sa] + l_scanned_seeds[key_4.sa] + l_block_counts[key_4.sa];
            out[address] = val_4.sa;
            histogram[key_4.sa]++;

            address = histogram[key_4.sb] + l_scanned_seeds[key_4.sb] + l_block_counts[key_4.sb];
            out[address] = val_4.sb;
            histogram[key_4.sb]++;

            address = histogram[key_4.sc] + l_scanned_seeds[key_4.sc] + l_block_counts[key_4.sc];
            out[address] = val_4.sc;
            histogram[key_4.sc]++;

            address = histogram[key_4.sd] + l_scanned_seeds[key_4.sd] + l_block_counts[key_4.sd];
            out[address] = val_4.sd;
            histogram[key_4.sd]++;

            address = histogram[key_4.se] + l_scanned_seeds[key_4.se] + l_block_counts[key_4.se];
            out[address] = val_4.se;
            histogram[key_4.se]++;

            address = histogram[key_4.sf] + l_scanned_seeds[key_4.sf] + l_block_counts[key_4.sf];
            out[address] = val_4.sf;
            histogram[key_4.sf]++;
#endif
        }

        // Before proceeding, make sure everyone has finished their current
        // indexing computations.
        __syncthreads();
        // Now update the seed array.
        if (threadIdx.x == blockDim.x-1)
        {
            for (int q = 0; q < 16; q++)
            {
            // save partial seed vectors in local memory if we need to advance window
                 l_block_counts[q] += histogram[q];
            }
        }
        __syncthreads();

        // Advance window
        window += blockDim.x;
        i += blockDim.x;
    }
}

