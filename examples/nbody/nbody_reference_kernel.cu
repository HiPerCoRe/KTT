inline __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}

inline __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

inline __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

inline __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

extern "C" __global__ void nbody_kernel(float dt1,
	float4* pos_old, 
	float4* pos_new,
	float4* oldVel,
	float4* newVel,
	float damping, 
	float softeningSqr)
{
	const float4 dt = make_float4(dt1, dt1, dt1, 0.0f);//(float4){.x=dt1,.y=dt1,.z=dt1,.w=0.0f};
	int gti = blockIdx.x*blockDim.x + threadIdx.x;
	int ti = threadIdx.x;
	int n = blockDim.x*gridDim.x;
	int nt = blockDim.x;
	int nb = n/nt;
	__shared__ float4 pblock[1024]; // FIXME
	float4 p = pos_old[gti];
	float4 v = oldVel[gti];
	float4 a = make_float4(0.0f);//{.x=0.0f,.y=0.0f,.z=0.0f,.w=0.0f};
	
	for(int jb=0; jb < nb; jb++) { /* Foreach block ... */
		pblock[ti] = pos_old[jb*nt+ti]; /* Cache ONE particle position */
		__syncthreads(); /* Wait for others in the work-group */
		for(int j=0; j<nt; j++) { /* For ALL cached particle positions ... */
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 - p;
			float invr = rsqrtf(d.x*d.x + d.y*d.y + d.z*d.z + softeningSqr);
			float f = p2.w*invr*invr*invr;
			a += f*d; /* Accumulate acceleration */
		}
		__syncthreads(); /* Wait for others in work-group */
	}
	p += dt*v + damping*dt*dt*a;
	v += dt*a;

	pos_new[gti] = p;
	newVel[gti] = v;
}

