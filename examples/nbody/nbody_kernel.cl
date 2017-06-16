// altered source from bodysystemcuda.cu
float3 bodyBodyInteraction(float3 ai,
                    float4 bi,
                    float4 bj,
					float softeningSqr)
{
    float3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += softeningSqr;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float invDist = rsqrt(distSqr);
    float invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

/*
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
*/

float3 computeBodyAccel(float4 bodyPos,
                 __global float4 *__restrict__ positions,
                 int numTiles, 
				 float softeningSqr)
{
    float4 sharedPos[WORK_GROUP_SIZE_X]; //= SharedMemory<typename vec4<T>::Type>();

    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++)
    {
        sharedPos[get_local_id(0)] = positions[tile * get_local_size(0) + get_local_id(0)];

        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        // This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

        for (unsigned int counter = 0; counter < get_local_size(0); counter++)
        {
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter], softeningSqr);
        }

        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    return acc;
}

__kernel void nbody_kernel(float dt1,
	__global float4* pos_old, 
	__global float4* pos_new,
	__global float4* vel,
	float damping, float softeningSqr)
{
	
	int n = get_global_size(0); // number of bodies
	int nt = get_local_size(0); // number of threads in block
	int nb = n/nt; // number of tiles
	
    int index = get_global_id(0);

    float4 position = pos_old[index];

    float3 accel = computeBodyAccel(position,
								   pos_old,
								   nb, softeningSqr);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * dt1
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    float4 velocity = vel[index];

    velocity.x += accel.x * dt1;
    velocity.y += accel.y * dt1;
    velocity.z += accel.z * dt1;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * dt1
    position.x += velocity.x * dt1;
    position.y += velocity.y * dt1;
    position.z += velocity.z * dt1;

    // store new position and velocity
    pos_new[index] = position;
    vel[index]    = velocity;
}
