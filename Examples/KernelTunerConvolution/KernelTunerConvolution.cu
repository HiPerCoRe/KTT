#define input_height (IMAGE_HEIGHT + HFS*2)
#define input_width (IMAGE_WIDTH + HFS*2)

#define i_end min(BLOCK_SIZE_Y*TILE_SIZE_Y+HFS*2, input_height)
#define j_end min(BLOCK_SIZE_X*TILE_SIZE_X+HFS*2, input_width)

    /*
        * If requested, we can use the __ldg directive to load data through the
         * read-only cache. 
          */
#if READ_ONLY == 1
#define LDG(x, y) __ldg(x+y)
#elif READ_ONLY == 0
#define LDG(x, y) x[y]
#endif

    __constant__ float d_filter[33*33]; //large enough for the largest filter

    /*
        * If use_padding == 1, we introduce (only when necessary) a number of padding
        * columns in shared memory to avoid shared memory bank conflicts
        *
        * padding columns are only inserted when BLOCK_SIZE_X is not a multiple of 32 (the assumed number of memory banks)
        * and when the width of the data needed is not a multiple of 32. The latter is because some (HFS*2+1)s never
        * cause bank conflicts.
        * 
        * If not passed as a tunable parameter, padding is on by default
        */
#define shared_mem_width (BLOCK_SIZE_X*TILE_SIZE_X+HFS*2)
#if use_padding == 1
    #if (((BLOCK_SIZE_X % 32)!=0) && (((shared_mem_width-BLOCK_SIZE_X)%32) != 0))
        // next line uses &31 instead of %32, because % in C is remainder not modulo
        #define padding_columns ((32 - (HFS*2 + BLOCK_SIZE_X*TILE_SIZE_X - BLOCK_SIZE_X)) & 31)
        #undef shared_mem_width
        #define shared_mem_width (BLOCK_SIZE_X*TILE_SIZE_X+HFS*2+padding_columns)
    #endif
#endif

    __global__ void Convolution(float *output, float *input, float *filter) {
        int ty = threadIdx.y;
        int tx = threadIdx.x;
        int by = blockIdx.y * BLOCK_SIZE_Y * TILE_SIZE_Y;
        int bx = blockIdx.x * BLOCK_SIZE_X * TILE_SIZE_X;

        //if (bx == 0 && by == 0 && tx == 0 && ty == 0)
        //    printf("XXX %i %i %i %i\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

        //shared memory to hold all input data need by this thread block
        __shared__ float sh_input[BLOCK_SIZE_Y*TILE_SIZE_Y+HFS*2][shared_mem_width];

        //load all input data needed by this thread block into shared memory
        #pragma unroll
        for (int i=ty; i<i_end; i+=BLOCK_SIZE_Y) {
            #pragma unroll
            for (int j=tx; j<j_end; j+=BLOCK_SIZE_X) {
                #if ((IMAGE_HEIGHT%(BLOCK_SIZE_Y*TILE_SIZE_Y)!=0) || (IMAGE_WIDTH%(BLOCK_SIZE_X*TILE_SIZE_X)!=0))
                int y = by+i;
                int x = bx+j;
                if (y < input_height && x < input_width) {
                    sh_input[i][j] = LDG(input, y*input_width+x);
                    //if (blockIdx.x == gridDim.x-1 && blockIdx.y == 0)
                    //    printf("XXX %i (%i %i %i) -> %i %i : %f\n", y*input_width+x, x, y, input_width, j, i, sh_input[i][j]);
                }
                #else
                sh_input[i][j] = LDG(input, (by+i)*input_width + (bx+j));
                #endif
            }
        }
        __syncthreads();

        //thread-local registers to hold local sums
        float sum[TILE_SIZE_Y][TILE_SIZE_X];
        #pragma unroll
        for (int yi=0; yi<TILE_SIZE_Y; yi++) {
            #pragma unroll
            for (int xi=0; xi<TILE_SIZE_X; xi++) {
                sum[yi][xi] = 0.0f;
            }
        }

        //for each filter weight
        #pragma unroll
        for (int i=0; i < (HFS*2+1); i++) {
            #pragma unroll
            for (int j=0; j < (HFS*2+1); j++) {
                #pragma unroll
                for (int yi=0; yi<TILE_SIZE_Y; yi++) {   
                    #pragma unroll
                    for (int xi=0; xi<TILE_SIZE_X; xi++) {
                        sum[yi][xi] += sh_input[ty+yi*BLOCK_SIZE_Y+i][tx+xi*BLOCK_SIZE_X+j] * filter[i*(HFS*2+1)+j];
                    }
                }
            }
        }

        //store results to global memory
        #pragma unroll
        for (int yi=0; yi<TILE_SIZE_Y; yi++) {   
            #pragma unroll
            for (int xi=0; xi<TILE_SIZE_X; xi++) {
                #if ((IMAGE_HEIGHT%(BLOCK_SIZE_Y*TILE_SIZE_Y)!=0) || (IMAGE_WIDTH%(BLOCK_SIZE_X*TILE_SIZE_X)!=0))
                int y = by+ty+yi*BLOCK_SIZE_Y;
                int x = bx+tx+xi*BLOCK_SIZE_X;
                if (y < IMAGE_HEIGHT && x < IMAGE_WIDTH) {
                    output[y * IMAGE_WIDTH + x] = sum[yi][xi];
                }
                #else
                output[(by+ty+yi*BLOCK_SIZE_Y) * IMAGE_WIDTH + bx+tx+xi*BLOCK_SIZE_X] = sum[yi][xi];
            #endif
        }
    }
}

