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
    float distSqr = (d.x * d.x) + (d.y * d.y) + (d.z * d.z) + softeningSqr;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float invDist = rsqrt(distSqr);
    // float invDistCube =  invDist * invDist * invDist;

    // f = m_j * invDistCube [1 FLOP]
    float f = pJMass * invDist * invDist * invDist;

    // a_i =  a_i + f * r_ij [6 FLOPS]
    a.x = d.x * f;
    a.y = d.y * f;
    a.z = d.z * f;

    return a;
}

void loadThreadData(
	__global float4* pos_old, __global float4* vel,
	float bodyPos[3], float bodyVel[3], float bodyAcc[3], float* bodyMass,
	int start, int end) 
{
	int tid = get_local_id(0);
    int length = end - start + 1;
	if (tid < length) {
        // store 'thread specific' body info to registers
        bodyPos[0] = pos_old[start + tid].x;
        bodyPos[1] = pos_old[start + tid].y;
        bodyPos[2] = pos_old[start + tid].z;
		
        bodyVel[0] = vel[start + tid].x;
        bodyVel[1] = vel[start + tid].y;
        bodyVel[2] = vel[start + tid].z;
		
		*bodyMass = pos_old[start + tid].w;
		// erase acceleration buffer
		bodyAcc[0] = bodyAcc[1] = bodyAcc[2] = 0.f;
    }
}

// method to process complete block, i.e. part of the bodies array where each body's acceleration is added to result
void processCompleteBlock(float bodyAcc[3],float oldPosX[WORK_GROUP_SIZE_X],float oldPosY[WORK_GROUP_SIZE_X],float oldPosZ[WORK_GROUP_SIZE_X],float mass[WORK_GROUP_SIZE_X],
	__global float4* pos_old, 
	float softeningSqr, float bodyPos[3], int start) {
    int tid = get_local_id(0);

    // load new values to buffer. We know that all threads can be used now, so no condition is necessary
	oldPosX[tid] = pos_old[start + tid].x;
	oldPosY[tid] = pos_old[start + tid].y;
	oldPosZ[tid] = pos_old[start + tid].z;
	mass[tid] = pos_old[start + tid].w;
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate the acceleration between the thread body and each other body loaded to buffer
    # pragma unroll INNER_UNROLL_FACTOR1
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
void processFinalBlock(float bodyAcc[3],float oldPosX[WORK_GROUP_SIZE_X],float oldPosY[WORK_GROUP_SIZE_X],float oldPosZ[WORK_GROUP_SIZE_X],float mass[WORK_GROUP_SIZE_X],
	__global float4* pos_old, 
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
		mass[tid] = pos_old[start + tid].w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate the distance between the thread atom and each other atom loaded to buffer
    int count = length / INNER_UNROLL_FACTOR2;
    int tmp  = INNER_UNROLL_FACTOR2 * count;
    # pragma unroll INNER_UNROLL_FACTOR2
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
	__global float4* newVel,
	__global float4* pos_old, 
	__global float4* pos_new,
	__global float4* oldVel,
	float damping, 
	float softeningSqr) {
		
	// arrays for bodies block
	// old position
	__local float oldPosX[WORK_GROUP_SIZE_X];
	__local float oldPosY[WORK_GROUP_SIZE_X];
	__local float oldPosZ[WORK_GROUP_SIZE_X];
	// mass
	__local float mass[WORK_GROUP_SIZE_X];
	
	int n = get_global_size(0);
    // each thread holds a position of the body it represents
    float bodyPos[3];
    float bodyVel[3];
	float bodyAcc[3];
	float bodyMass;

	loadThreadData(pos_old, oldVel, bodyPos, bodyVel, bodyAcc, &bodyMass,
		get_group_id(0) * WORK_GROUP_SIZE_X, // start index
		min(WORK_GROUP_SIZE_X * ((int)get_group_id(0) + 1) - 1, n - 1)); // end index

    // after processing the initial block, process all internal ones
    
	int blocks = n / WORK_GROUP_SIZE_X;
    for (int i = 0; i < blocks; i++) {
        // start is the first body in the block being processed
        processCompleteBlock(bodyAcc, oldPosX, oldPosY, oldPosZ, mass,
			pos_old, softeningSqr, bodyPos, i * WORK_GROUP_SIZE_X);
    }
    // at the end, do the final block
    processFinalBlock(bodyAcc, oldPosX, oldPosY, oldPosZ, mass,
		pos_old, 
		softeningSqr, bodyPos,blocks * WORK_GROUP_SIZE_X, n-1);

	int gtid = get_global_id(0);

	// calculate resulting position 	
	float resPosX = bodyPos[0] + timeDelta * bodyVel[0] + damping * timeDelta * timeDelta * bodyAcc[0];
	float resPosY = bodyPos[1] + timeDelta * bodyVel[1] + damping * timeDelta * timeDelta * bodyAcc[1];
	float resPosZ = bodyPos[2] + timeDelta * bodyVel[2] + damping * timeDelta * timeDelta * bodyAcc[2];
	pos_new[gtid] = (float4)(resPosX, resPosY, resPosZ, bodyMass);
	// calculate resulting velocity	
	// float resVelX = bodyVel[0] + timeDelta * bodyAcc[0];
	// float resVelY = bodyVel[1] + timeDelta * bodyAcc[1];
	// float resVelZ = bodyVel[2] + timeDelta * bodyAcc[2];
	// vel{gtid] = (float4)(resVelX, resVelY, resVelZ, 0.f);
	newVel[gtid].x = bodyVel[0] + timeDelta * bodyAcc[0];
	newVel[gtid].y = bodyVel[1] + timeDelta * bodyAcc[1];
	newVel[gtid].z = bodyVel[2] + timeDelta * bodyAcc[2];
	newVel[gtid].w = 0.f;
}