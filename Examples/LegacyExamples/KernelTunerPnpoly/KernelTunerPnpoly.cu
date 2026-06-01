__constant__ float2 d_vertices[VERTICES];

/*
 * This file contains the implementation of a CUDA Kernel for the
 * point-in-polygon problem using the crossing number algorithm
 *
 * The kernel cn_pnpoly is can be tuned using the following parameters:
 *    * BLOCK_SIZE_X                any sensible thread block size
 *    * TILE_SIZE                   any sensible tile size value
 *    * BETWEEN_METHOD              any of [0, 1, 2, 3]
 *    * USE_PRECOMPUTED_SLOPES      enable or disable [0, 1]
 *    * USE_METHOD                  any of [0, 1, 2]
 *
 * The kernel cn_pnpoly_naive is used for correctness checking.
 *
 * The algorithm used here is adapted from:
 *     'Inclusion of a Point in a Polygon', Dan Sunday, 2001
 *     (http://geomalgorithms.com/a03-_inclusion.html)
 *
 * Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */

/*
 * The is_between method returns a boolean that is True when the a is between c and b.
 * Since the kernel is instruction bound, the exact way in which you compute is_between
 * can have a dramatic effect on performance.
 * Note that the way the different methods handle coincidents of a with b and c differs slightly.
 */
__device__ __forceinline__ int is_between(float a, float b, float c) {
    #if BETWEEN_METHOD == 0
        return (b > a) != (c > a);
    #elif BETWEEN_METHOD == 1
        return ((b <= a) && (c > a)) || ((b > a) && (c <= a));
    #elif BETWEEN_METHOD == 2
        return ((a - b) == 0.0f) || ((a - b) * (a - c) < 0.0f);
    #elif BETWEEN_METHOD == 3
        //Interestingly enough method 3 exactly the same as method 2, only in a different order.
        //the performance difference between method 2 and 3 can be huge depending on all the other optimization parameters.
        return ((a - b) * (a - c) < 0.0f) || (a - b == 0.0f);
    #endif
}



/*
 * The Point-in-Polygon kernel
 */
__global__ void Pnpoly(int* bitmap, float2* points, float2* vertices, int n) {
    int i = blockIdx.x * BLOCK_SIZE_X * TILE_SIZE + threadIdx.x;
    if (i < n) {

        int c[TILE_SIZE];
        float2 lpoints[TILE_SIZE];
        #pragma unroll
        for (int ti=0; ti<TILE_SIZE; ti++) {
            c[ti] = 0;
            if (i+BLOCK_SIZE_X*ti < n) {
                lpoints[ti] = points[i+BLOCK_SIZE_X*ti];
            }
        }

        int k = VERTICES-1;

        for (int j=0; j<VERTICES; k = j++) {    // edge from vj to vk
            float2 vj = d_vertices[j]; //TODO faster with d_vertices, but needs constant data movement in KTT
            float2 vk = d_vertices[k]; //TODO faster with d_vertices, but needs constant data movement in KTT

            float slope = (vk.x-vj.x) / (vk.y-vj.y);

            #pragma unroll
            for (int ti=0; ti<TILE_SIZE; ti++) {

                float2 p = lpoints[ti];

                #if USE_METHOD == 0
                if (  is_between(p.y, vj.y, vk.y) &&         //if p is between vj and vk vertically
                     (p.x < slope * (p.y-vj.y) + vj.x)
                        ) {  //if p.x crosses the line vj-vk when moved in positive x-direction
                    c[ti] = !c[ti];
                }

                #elif USE_METHOD == 1
                //Same as method 0, but attempts to reduce divergence by avoiding the use of an if-statement.
                //Whether this is more efficient is data dependent because there will be no divergence using method 0, when none
                //of the threads within a warp evaluate is_between as true
                int b = is_between(p.y, vj.y, vk.y);
                c[ti] += b && (p.x < vj.x + slope * (p.y - vj.y));

                #elif USE_METHOD == 2
                if (  is_between(p.y, vj.y, vk.y) &&         //if p is between vj and vk vertically
                     (p.x < slope * (p.y-vj.y) + vj.x)
                        ) {  //if p.x crosses the line vj-vk when moved in positive x-direction
                    c[ti] += 1;
                }

                #endif


            }

        }

        #pragma unroll
        for (int ti=0; ti<TILE_SIZE; ti++) {
            //could do an if statement here if 1s are expected to be rare
            if (i+BLOCK_SIZE_X*ti < n) {
                #if USE_METHOD == 0
                bitmap[i+BLOCK_SIZE_X*ti] = c[ti];
                #else
                bitmap[i+BLOCK_SIZE_X*ti] = c[ti] & 1;
                #endif
            }
        }
    }

}

