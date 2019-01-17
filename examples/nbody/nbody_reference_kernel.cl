__kernel void nbody_kernel(float dt1,
	__global float4* pos_old, 
	__global float4* pos_new,
	__global float4* oldVel,
	__global float4* newVel,
	float damping, 
	float softeningSqr)
{
	
	const float4 dt = (float4)(dt1,dt1,dt1,0.0f);
	int gti = get_global_id(0);
	int ti = get_local_id(0);
	int n = get_global_size(0);
	int nt = get_local_size(0);
	int nb = n/nt;
	__local float4 pblock[1024]; // FIXME
	float4 p = pos_old[gti];
	float4 v = oldVel[gti];
	float4 a = (float4)(0.0f,0.0f,0.0f,0.0f);
	
	for(int jb=0; jb < nb; jb++) { /* Foreach block ... */
		pblock[ti] = pos_old[jb*nt+ti]; /* Cache ONE particle position */
		barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in the work-group */
		for(int j=0; j<nt; j++) { /* For ALL cached particle positions ... */
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 - p;
			float invr = half_rsqrt(d.x*d.x + d.y*d.y + d.z*d.z + softeningSqr);
			float f = p2.w*invr*invr*invr;
			a += f*d; /* Accumulate acceleration */
		}
		barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in work-group */
	}
	p += dt*v + damping*dt*dt*a;
	v += dt*a;

	pos_new[gti] = p;
	newVel[gti] = v;
}
