#if VECTOR_TYPE == 1
    typedef float vector;
#elif VECTOR_TYPE == 2
    typedef float2 vector;
#elif VECTOR_TYPE == 4
    typedef float4 vector;
#elif VECTOR_TYPE == 8
    typedef float8 vector;
#elif VECTOR_TYPE == 16
    typedef float16 vector;
#endif


#if LOCAL_MEM == 1
__kernel void mtran(
    __global float* restrict output,
    __global const float* restrict input,
    const int width,
    const int height)
{
	__local float tile[TILE_SIZE_Y][TILE_SIZE_X+PADD_LOCAL];
	int lx = get_local_id(0);
        int ly = get_local_id(1);
	int gx = get_group_id(0);
	int gy = get_group_id(1);
        int x = gx*TILE_SIZE_X + lx;
        int yy = gy*TILE_SIZE_Y;
        for (int y = ly; y <  TILE_SIZE_Y; y += WORK_GROUP_SIZE_Y)
        {
                tile[y][lx] = input[(yy+y)*width + x];
        }
	barrier(CLK_LOCAL_MEM_FENCE);
	int id = ly*WORK_GROUP_SIZE_X+lx;
	int tlx = id%TILE_SIZE_Y;
	int tly = id/TILE_SIZE_Y;
	for (int i = tly; i < TILE_SIZE_X; i += (WORK_GROUP_SIZE_X*WORK_GROUP_SIZE_Y)/TILE_SIZE_Y)
	{
		output[(get_group_id(0)*TILE_SIZE_X+i)*width + get_group_id(1)*TILE_SIZE_Y + tlx] = tile[tlx][i];
	}
}
#else
__kernel void mtran(
#if CR == 1
    __global vector* restrict output,
    __global const float* restrict input, 
#else
    __global float* restrict output,
    __global const vector* restrict input,
#endif
    const int width,
    const int height)
{
	int x = get_group_id(0)*TILE_SIZE_X + get_local_id(0);
	int xt = x*VECTOR_TYPE;
        int yy = get_group_id(1)*TILE_SIZE_Y + get_local_id(1);
        int lx = get_local_id(0);
#if PREFETCH == 1
	//XXX fix vectorization + prefetching
	for (int y = yy; y <  yy+TILE_SIZE_Y*VECTOR_TYPE; y += WORK_GROUP_SIZE_Y)
        {
#if CR == 1
                prefetch(&(input[x*height + y]), 1);
#else
		prefetch(&(output[x*height + y]), 1);
#endif                  
        }
	for (int y = yy; y <  yy+TILE_SIZE_Y; y += WORK_GROUP_SIZE_Y)
        {
#if CR == 1
                prefetch(&(output[y*width + x]), VECTOR_TYPE);
#else
                prefetch(&(input[y*width + x]), VECTOR_TYPE);
#endif
        }
#endif
        for (int y = yy; y <  yy+TILE_SIZE_Y; y += WORK_GROUP_SIZE_Y)
	{
#if CR == 1
#if PREFETCH == 2
		for (int i = 0; i < VECTOR_TYPE; i++)
			prefetch(&(input[(xt+i)*height + y]), TILE_SIZE_Y);
                //prefetch(&(output[y*(width/VECTOR_TYPE) + x]), VECTOR_TYPE);
#endif
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
#if VECTOR_TYPE == 8
                v.s0 = input[xt*height + y];
                v.s1 = input[(xt+1)*height + y];
                v.s2 = input[(xt+2)*height + y];
                v.s3 = input[(xt+3)*height + y];
		v.s4 = input[(xt+4)*height + y];
                v.s5 = input[(xt+5)*height + y];
                v.s6 = input[(xt+6)*height + y];
                v.s7 = input[(xt+7)*height + y];
#endif
#if VECTOR_TYPE == 16
                v.s0 = input[xt*height + y];
                v.s1 = input[(xt+1)*height + y];
                v.s2 = input[(xt+2)*height + y];
                v.s3 = input[(xt+3)*height + y];
                v.s4 = input[(xt+4)*height + y];
                v.s5 = input[(xt+5)*height + y];
                v.s6 = input[(xt+6)*height + y];
                v.s7 = input[(xt+7)*height + y];
                v.s8 = input[(xt+8)*height + y];
                v.s9 = input[(xt+9)*height + y];
                v.sa = input[(xt+10)*height + y];
                v.sb = input[(xt+11)*height + y];
                v.sc = input[(xt+12)*height + y];
                v.sd = input[(xt+13)*height + y];
                v.se = input[(xt+14)*height + y];
                v.sf = input[(xt+15)*height + y];

#endif
		output[y*(width/VECTOR_TYPE) + x] = v;
#else
#if PREFETCH == 2
		//prefetch(&(input[y*(width/VECTOR_TYPE) + x]), VECTOR_TYPE);
		for (int i = 0; i < VECTOR_TYPE; i++)
	                prefetch(&(output[(xt+i)*height + y]), 1);
#endif
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
#if VECTOR_TYPE == 8
                output[xt*height + y] = v.s0;
                output[(xt+1)*height + y] = v.s1;
                output[(xt+2)*height + y] = v.s2;
                output[(xt+3)*height + y] = v.s3;
		output[(xt+4)*height + y] = v.s4;
                output[(xt+5)*height + y] = v.s5;
                output[(xt+6)*height + y] = v.s6;
                output[(xt+7)*height + y] = v.s7;
#endif
#if VECTOR_TYPE == 16
                output[xt*height + y] = v.s0;
                output[(xt+1)*height + y] = v.s1;
                output[(xt+2)*height + y] = v.s2;
                output[(xt+3)*height + y] = v.s3;
                output[(xt+4)*height + y] = v.s4;
                output[(xt+5)*height + y] = v.s5;
                output[(xt+6)*height + y] = v.s6;
                output[(xt+7)*height + y] = v.s7;
                output[(xt+8)*height + y] = v.s8;
                output[(xt+9)*height + y] = v.s9;
                output[(xt+10)*height + y] = v.sa;
                output[(xt+11)*height + y] = v.sb;
                output[(xt+12)*height + y] = v.sc;
                output[(xt+13)*height + y] = v.sd;
                output[(xt+14)*height + y] = v.se;
                output[(xt+15)*height + y] = v.sf;
#endif
#endif
	}
}
#endif
