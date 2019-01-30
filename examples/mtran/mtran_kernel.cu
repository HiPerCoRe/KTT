#if VECTOR_TYPE == 1
    typedef float vector;
#elif VECTOR_TYPE == 2
    typedef float2 vector;
#elif VECTOR_TYPE == 4
    typedef float4 vector;
#endif

#if LOCAL_MEM == 1
extern "C" __global__ void mtran(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int width,
    const int height)
{
	__shared__ float tile[TILE_SIZE_Y][TILE_SIZE_X+PADD_LOCAL];
	int lx = threadIdx.x;
        int ly = threadIdx.y;
	int gx = blockIdx.x;
	int gy = blockIdx.y;
        int x = gx*TILE_SIZE_X + lx;
        int yy = gy*TILE_SIZE_Y;
        for (int y = ly; y <  TILE_SIZE_Y; y += WORK_GROUP_SIZE_Y)
        {
                tile[y][lx] = input[(yy+y)*width + x];
        }
	__syncthreads();
	int id = ly*WORK_GROUP_SIZE_X+lx;
	int tlx = id%TILE_SIZE_Y;
	int tly = id/TILE_SIZE_Y;
	for (int i = tly; i < TILE_SIZE_X; i += (WORK_GROUP_SIZE_X*WORK_GROUP_SIZE_Y)/TILE_SIZE_Y)
	{
		output[(blockIdx.x*TILE_SIZE_X+i)*width + blockIdx.y*TILE_SIZE_Y + tlx] = tile[tlx][i];
	}
}
#else
extern "C" __global__ void mtran(
  #if CR == 1
    vector* __restrict__ output,
    const float* __restrict__ input, 
  #else
    float* __restrict__ output,
    const vector* __restrict__ input,
  #endif
    const int width,
    const int height)
{
	int x = blockIdx.x*TILE_SIZE_X + threadIdx.x;
	int xt = x*VECTOR_TYPE;
    int yy = blockIdx.y*TILE_SIZE_Y + threadIdx.y;
    int lx = threadIdx.x;
    for (int y = yy; y <  yy+TILE_SIZE_Y; y += WORK_GROUP_SIZE_Y)
	{
  #if CR == 1
		vector v;
    #if VECTOR_TYPE == 1
		v = input[xt*height + y];
    #endif
    #if VECTOR_TYPE == 2
        v.x = input[xt*height + y];
		v.y = input[(xt+1)*height + y];
    #endif
    #if VECTOR_TYPE == 4
        v.x = input[xt*height + y];
        v.y = input[(xt+1)*height + y];
		v.z = input[(xt+2)*height + y];
		v.w = input[(xt+3)*height + y];
    #endif
		output[y*(width/VECTOR_TYPE) + x] = v;
  #else
		vector v = input[y*(width/VECTOR_TYPE) + x];
    #if VECTOR_TYPE == 1
        output[xt*height + y] = v;
    #endif
    #if VECTOR_TYPE == 2
		output[xt*height + y] = v.x;
		output[(xt+1)*height + y] = v.y;
    #endif
    #if VECTOR_TYPE == 4
        output[xt*height + y] = v.x;
        output[(xt+1)*height + y] = v.y;
		output[(xt+2)*height + y] = v.z;
		output[(xt+3)*height + y] = v.w;
    #endif
  #endif
	}
}
#endif
