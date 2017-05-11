// arrays for temporal results
// new acceleration

float3 getAcceleration(float pI[3], float pJX, float pJY, float pJZ, float pJMass, float softeningSqr) {
	float3 d;
	float3 a;
    // r_ij  [3 FLOPS]
    d.x = pJX - pI[0];
    d.y = pJY - pI[1];
    d.z = pJZ - pI[2];

    // distSqr = dot(d_ij, d_ij) + EPS^2  [6 FLOPS]
    float distSqr = d.x * d.x + d.y * d.y + d.z * d.z + softeningSqr;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float invDist = rsqrt(distSqr);
    float invDistCube =  invDist * invDist * invDist;

    // f = m_j * invDistCube [1 FLOP]
    float f = pJMass * invDistCube;

    // a_i =  a_i + f * r_ij [6 FLOPS]
    a.x = d.x * f;
    a.y = d.y * f;
    a.z = d.z * f;

    return a;
}

	
// method to process initial block, i.e. part of the array where the threads has to do the most specific work, arrays get innitialized etc.
void processStartBlock(float bodyAcc[3], float* bodyMass, float oldPosX[WORK_GROUP_SIZE_X],float oldPosY[WORK_GROUP_SIZE_X],float oldPosZ[WORK_GROUP_SIZE_X],float oldVelX[WORK_GROUP_SIZE_X],float oldVelY[WORK_GROUP_SIZE_X],float oldVelZ[WORK_GROUP_SIZE_X],float mass[WORK_GROUP_SIZE_X],
	__global float4* pos_old, 
	__global float4* vel,
	float softeningSqr, float bodyPos[3], float bodyVel[3], int start, int end) {
		
    int tid = get_local_id(0);
    int length = end - start + 1;

    // each thread loads a bit of the memory, but only the allowed part (this block can be final and not complete)
    if (tid < length) {
        // fill buffers for body positions etc
        oldPosX[tid] = pos_old[start + tid].x;
        oldPosY[tid] = pos_old[start + tid].y;
        oldPosZ[tid] = pos_old[start + tid].z;
        oldVelX[tid] = vel[start + tid].x;
        oldVelY[tid] = vel[start + tid].y;
        oldVelZ[tid] = vel[start + tid].z;
		mass[tid] = pos_old[start + tid].w;
        // save 'your' body info in register
        bodyPos[0] = oldPosX[tid];
        bodyPos[1] = oldPosY[tid];
        bodyPos[2] = oldPosZ[tid];
        bodyVel[0] = oldVelX[tid];
        bodyVel[1] = oldVelY[tid];
        bodyVel[2] = oldVelZ[tid];
		bodyAcc[0] = bodyAcc[1] = bodyAcc[1] = 0.f;
		*bodyMass = mass[tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // now calculate the acceleration between thread body and other bodies
    // within the block. Each thread will use only the bodies that are after it
    for(int i =  1; i < length - tid ; i++) {
        int index = i + tid;
		float3 acc = getAcceleration(bodyPos, oldPosX[index], oldPosY[index], oldPosZ[index], mass[index], softeningSqr);
		bodyAcc[0] += acc.x;
		bodyAcc[1] += acc.y;
		bodyAcc[2] += acc.z;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// method to process complete block, i.e. part of the bodies array where each body's acceleration is added to result
void processCompleteBlock(float bodyAcc[3],float oldPosX[WORK_GROUP_SIZE_X],float oldPosY[WORK_GROUP_SIZE_X],float oldPosZ[WORK_GROUP_SIZE_X],float oldVelX[WORK_GROUP_SIZE_X],float oldVelY[WORK_GROUP_SIZE_X],float oldVelZ[WORK_GROUP_SIZE_X],float mass[WORK_GROUP_SIZE_X],
	__global float4* pos_old, 
	__global float4* vel,
	float softeningSqr, float bodyPos[3], int start) {
    int tid = get_local_id(0);

    // load new values to buffer. We know that all threads can be used now, so no condition is necessary
	oldPosX[tid] = pos_old[start + tid].x;
	oldPosY[tid] = pos_old[start + tid].y;
	oldPosZ[tid] = pos_old[start + tid].z;
	oldVelX[tid] = vel[start + tid].x;
	oldVelY[tid] = vel[start + tid].y;
	oldVelZ[tid] = vel[start + tid].z;
	mass[tid] = pos_old[start + tid].w;
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate the acceleration between the thread body and each other body loaded to buffer
    # pragma unroll INNER_UNROLL_FACTOR
    for(int i =  0; i < WORK_GROUP_SIZE_X; i++) {
        int index = i;
		float3 acc = getAcceleration(bodyPos, oldPosX[index], oldPosY[index], oldPosZ[index], mass[index], softeningSqr);
		bodyAcc[0] += acc.x;
		bodyAcc[1] += acc.y;
		bodyAcc[2] += acc.z;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// method to process final block, i.e. part of the molecule array where the algorithm terminates
void processFinalBlock(float bodyAcc[3],float oldPosX[WORK_GROUP_SIZE_X],float oldPosY[WORK_GROUP_SIZE_X],float oldPosZ[WORK_GROUP_SIZE_X],float oldVelX[WORK_GROUP_SIZE_X],float oldVelY[WORK_GROUP_SIZE_X],float oldVelZ[WORK_GROUP_SIZE_X],float mass[WORK_GROUP_SIZE_X],
	__global float4* pos_old, 
	__global float4* vel,
	float softeningSqr, float bodyPos[3], int start, int end) {
    int tid = get_local_id(0);
    int length = end - start + 1;
    int topIndex = length;
    if (length < 0) {
        return;
    }
    // load new atoms, but only the threads that can work now
    if (tid < topIndex) {
		oldPosX[tid] = pos_old[start + tid].x;
		oldPosY[tid] = pos_old[start + tid].y;
		oldPosZ[tid] = pos_old[start + tid].z;
		oldVelX[tid] = vel[start + tid].x;
		oldVelY[tid] = vel[start + tid].y;
		oldVelZ[tid] = vel[start + tid].z;
		mass[tid] = pos_old[start + tid].w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate the distance between the thread atom and each other atom loaded to buffer
    int count = length / 2;
    int tmp  = 2 * count;
    #pragma unroll 2
    for(int i =  0; i < tmp; i++) {
        int index = i;
		float3 acc = getAcceleration(bodyPos, oldPosX[index], oldPosY[index], oldPosZ[index], mass[index], softeningSqr);
		bodyAcc[0] += acc.x;
		bodyAcc[1] += acc.y;
		bodyAcc[2] += acc.z;
    }

    for(int i =  tmp; i < length; i++) {
        int index = i;
		float3 acc = getAcceleration(bodyPos, oldPosX[index], oldPosY[index], oldPosZ[index], mass[index], softeningSqr);
		bodyAcc[0] += acc.x;
		bodyAcc[1] += acc.y;
		bodyAcc[2] += acc.z;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
	
	
// main method doing the calculation
__kernel void nbody_kernel(float timeDelta,
	__global float4* pos_old, 
	__global float4* pos_new,
	__global float4* vel,
	float damping, 
	float softeningSqr) {
		
	// arrays for bodies block
	// old position
	__local float oldPosX[WORK_GROUP_SIZE_X];
	__local float oldPosY[WORK_GROUP_SIZE_X];
	__local float oldPosZ[WORK_GROUP_SIZE_X];
	// old velocity
	__local float oldVelX[WORK_GROUP_SIZE_X];
	__local float oldVelY[WORK_GROUP_SIZE_X];
	__local float oldVelZ[WORK_GROUP_SIZE_X];
	// mass
	__local float mass[WORK_GROUP_SIZE_X];
	
	int n = get_global_size(0);
    // each thread holds a position of the body it represents
    float bodyPos[3];
    float bodyVel[3];
	float bodyAcc[3];
	float bodyMass;

    // process the first block, initialize local variables and prepare arrays
    // start point is the first position in the block, end point is either the last item of the array or last item of the block
    processStartBlock(bodyAcc, &bodyMass, oldPosX, oldPosY, oldPosZ, oldVelX, oldVelY, oldVelZ, mass,
		pos_old, 
		vel,
		softeningSqr
		, bodyPos, bodyVel, get_group_id(0) * WORK_GROUP_SIZE_X, min(WORK_GROUP_SIZE_X * ((int)get_group_id(0) + 1) - 1, n - 1));

    // after processing the initial block, process all internal ones
    int i = get_group_id(0) + 1;
    for (; i < (n-1)/WORK_GROUP_SIZE_X; i++) {
        // start is the first body in the block being processed
        processCompleteBlock(bodyAcc, oldPosX, oldPosY, oldPosZ, oldVelX, oldVelY, oldVelZ, mass,
		pos_old, vel, softeningSqr, bodyPos, i * WORK_GROUP_SIZE_X);
    }
    // at the end, do the final block
    processFinalBlock(bodyAcc, oldPosX, oldPosY, oldPosZ, oldVelX, oldVelY, oldVelZ, mass,
		pos_old, 
		vel,
		softeningSqr, bodyPos,i * WORK_GROUP_SIZE_X, n-1);

	int gtid = get_global_id(0);

	// calculate resulting position 	
	float resPosX = bodyPos[0] + timeDelta * bodyVel[0] + damping * timeDelta * timeDelta * bodyAcc[0];
	float resPosY = bodyPos[1] + timeDelta * bodyVel[1] + damping * timeDelta * timeDelta * bodyAcc[1];
	float resPosZ = bodyPos[2] + timeDelta * bodyVel[2] + damping * timeDelta * timeDelta * bodyAcc[2];
	pos_new[gtid] = (float4)(resPosX, resPosY, resPosZ, bodyMass);
	// calculate resulting velocity	
	float resVelX = bodyVel[0] + timeDelta * bodyAcc[0];
	float resVelY = bodyVel[1] + timeDelta * bodyAcc[1];
	float resVelZ = bodyVel[2] + timeDelta * bodyAcc[2];
	vel[gtid] = (float4)(resVelX, resVelY, resPosZ, 0.f);
}