#if USE_CONSTANT_MEMORY == 0
    #define MEMORY_TYPE_AOS __global  const
    #define MEMORY_TYPE_SOA __global  const
#elif USE_CONSTANT_MEMORY == 1
    #if USE_SOA == 0
        #define MEMORY_TYPE_AOS __constant
        #define MEMORY_TYPE_SOA __global  const
    #elif USE_SOA > 0
        #define MEMORY_TYPE_AOS __global  const
        #define MEMORY_TYPE_SOA __constant
    #endif // USE_SOA
#endif // USE_CONSTANT_MEMORY

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
#endif // VECTOR_TYPE

// method to calculate acceleration caused by body J
void updateAcc(vector bodyAcc[3], float bodyPos[3], // position of body I
	vector bufferPosX, vector bufferPosY, vector bufferPosZ, vector bufferMass, // position and mass of body J
	float softeningSqr) // to avoid infinities and zero division
{
	#if USE_SOA == 0
	{
		float3 d;
		d.x = bufferPosX - bodyPos[0];
		d.y = bufferPosY - bodyPos[1];
		d.z = bufferPosZ - bodyPos[2];

		float distSqr = (d.x * d.x) + (d.y * d.y) + (d.z * d.z) + softeningSqr;
		float invDist = rsqrt(distSqr);
		float f = bufferMass * invDist * invDist * invDist;

		bodyAcc[0] += d.x * f;
		bodyAcc[1] += d.y * f;
		bodyAcc[2] += d.z * f;
	}
	#else // USE_SOA != 0
	{
		vector distanceX = bufferPosX - bodyPos[0];
		vector distanceY = bufferPosY - bodyPos[1];
		vector distanceZ = bufferPosZ - bodyPos[2];
	   
		vector invDist = rsqrt(distanceX * distanceX + distanceY * distanceY + distanceZ * distanceZ + softeningSqr);
		vector f = bufferMass * invDist * invDist * invDist;

		bodyAcc[0] += distanceX * f;
		bodyAcc[1] += distanceY * f;
		bodyAcc[2] += distanceZ * f;
	}
	#endif // USE_SOA == 0
}

// method to load thread specific data from global memory
void loadThreadData(
	MEMORY_TYPE_AOS float4* oldBodyInfo, // global data; [X,Y,Z,mass]
	MEMORY_TYPE_SOA float* oldPosX,
	MEMORY_TYPE_SOA float* oldPosY,
	MEMORY_TYPE_SOA float* oldPosZ,
	MEMORY_TYPE_SOA float* mass,
	MEMORY_TYPE_AOS float4* oldVel, // velocity info
	MEMORY_TYPE_SOA float* oldVelX,
	MEMORY_TYPE_SOA float* oldVelY,
	MEMORY_TYPE_SOA float* oldVelZ,
	float bodyPos[3], float bodyVel[3], float* bodyMass, // thread data
	int start, int end) // indices
{
	int tid = get_local_id(0);
    int length = end - start + 1;
	
	if (tid < length) {
		#if USE_SOA == 0
		{
			// store 'thread specific' body info to registers
			bodyPos[0] = oldBodyInfo[start + tid].x;
			bodyPos[1] = oldBodyInfo[start + tid].y;
			bodyPos[2] = oldBodyInfo[start + tid].z;
			
			bodyVel[0] = oldVel[start + tid].x;
			bodyVel[1] = oldVel[start + tid].y;
			bodyVel[2] = oldVel[start + tid].z;
			
			*bodyMass = oldBodyInfo[start + tid].w;
		}
		#else // USE_SOA != 0
		{
			// store 'thread specific' body info to registers
			bodyPos[0] = oldPosX[start + tid];
			bodyPos[1] = oldPosY[start + tid];
			bodyPos[2] = oldPosZ[start + tid];
			
			bodyVel[0] = oldVelX[start + tid];
			bodyVel[1] = oldVelY[start + tid];
			bodyVel[2] = oldVelZ[start + tid];
			
			*bodyMass = mass[start + tid];
		}
		#endif // USE_SOA == 0
    }
}

// method will copy one item (X, Y, Z, mass) from input data to buffers
void fillBuffers(
	MEMORY_TYPE_AOS float4* oldBodyInfo, // global (input) data; [X,Y,Z,mass]
	MEMORY_TYPE_SOA vector* oldPosX,
	MEMORY_TYPE_SOA vector* oldPosY,
	MEMORY_TYPE_SOA vector* oldPosZ,
	MEMORY_TYPE_SOA vector* mass,
	vector bufferPosX[WORK_GROUP_SIZE_X], // buffers
	vector bufferPosY[WORK_GROUP_SIZE_X],
	vector bufferPosZ[WORK_GROUP_SIZE_X],
	vector bufferMass[WORK_GROUP_SIZE_X],
	int offset)
{
	 int tid = get_local_id(0);
	#if USE_SOA == 0
	{
		bufferPosX[tid] = oldBodyInfo[offset + tid].x;
		bufferPosY[tid] = oldBodyInfo[offset + tid].y;
		bufferPosZ[tid] = oldBodyInfo[offset + tid].z;
		bufferMass[tid] = oldBodyInfo[offset + tid].w;
	}
	#else // USE_SOA != 0
	{
		bufferPosX[tid] = oldPosX[offset + tid];
		bufferPosY[tid] = oldPosY[offset + tid];
		bufferPosZ[tid] = oldPosZ[offset + tid];
		bufferMass[tid] = mass[offset + tid];
	}
	#endif // USE_SOA == 0
}

// method to process complete block, i.e. part of the bodies array where
// each body's acceleration is added to result
void processCompleteBlock(
	MEMORY_TYPE_AOS float4* oldBodyInfo, // global data; [X,Y,Z,mass]
	MEMORY_TYPE_SOA vector* oldPosX,
	MEMORY_TYPE_SOA vector* oldPosY,
	MEMORY_TYPE_SOA vector* oldPosZ,
	MEMORY_TYPE_SOA vector* mass,
	vector bufferPosX[WORK_GROUP_SIZE_X], // buffers
	vector bufferPosY[WORK_GROUP_SIZE_X],
	vector bufferPosZ[WORK_GROUP_SIZE_X],
	vector bufferMass[WORK_GROUP_SIZE_X],
	vector bodyAcc[3], // thread specific data
	float bodyPos[3], 
	float softeningSqr, // used by acceleration
	int start) // initial index, included
{
    int tid = get_local_id(0);
    // load new values to buffer.
	// We know that all threads can be used now, so no condition is necessary
	fillBuffers(oldBodyInfo, oldPosX, oldPosY, oldPosZ, mass, bufferPosX, bufferPosY, bufferPosZ, bufferMass, start);
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate the acceleration between the thread body and each other body loaded to buffer
    # pragma unroll INNER_UNROLL_FACTOR1
    for(int i =  0; i < WORK_GROUP_SIZE_X; i++) {
        int index = i;
		updateAcc(bodyAcc, bodyPos,
			bufferPosX[index], bufferPosY[index], bufferPosZ[index], bufferMass[index],
			softeningSqr);
    }
    barrier(CLK_LOCAL_MEM_FENCE); // sync threads
}

// method to process final block, i.e. last section of the body array, which is shorter than WORK_GROUP_SIZE_X
void processFinalBlock(
	MEMORY_TYPE_AOS float4* oldBodyInfo, // global data; [X,Y,Z,mass]
	MEMORY_TYPE_SOA vector* oldPosX,
	MEMORY_TYPE_SOA vector* oldPosY,
	MEMORY_TYPE_SOA vector* oldPosZ,
	MEMORY_TYPE_SOA vector* mass,
	vector bufferPosX[WORK_GROUP_SIZE_X], // buffers
	vector bufferPosY[WORK_GROUP_SIZE_X],
	vector bufferPosZ[WORK_GROUP_SIZE_X],
	vector bufferMass[WORK_GROUP_SIZE_X],
	vector bodyAcc[3], // thread specific data
	float bodyPos[3], 
	float softeningSqr, // used by acceleration
	int start, int end) // initial (included) / end index
{
    int tid = get_local_id(0);
    int length = end - start + 1;
    int topIndex = length;
    if (length < 0) {
        return; 
    } // continue just with threads that won't access wrong memory
	
    // load new values to buffer
    if (tid < topIndex) {
		fillBuffers(oldBodyInfo, oldPosX, oldPosY, oldPosZ, mass, bufferPosX, bufferPosY, bufferPosZ, bufferMass, start);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate the acceleration between the thread body and each other body loaded to buffer
    int count = length / INNER_UNROLL_FACTOR2;
    int tmp  = INNER_UNROLL_FACTOR2 * count;
    # pragma unroll INNER_UNROLL_FACTOR2
    for(int i =  0; i < tmp; i++) {
        int index = i;
		updateAcc(bodyAcc, bodyPos,
			bufferPosX[index], bufferPosY[index], bufferPosZ[index], bufferMass[index],
			softeningSqr);
    }
	// finish those not processed in the block above, if any
    for(int i =  tmp; i < length; i++) {
        int index = i;
		updateAcc(bodyAcc, bodyPos,
			bufferPosX[index], bufferPosY[index], bufferPosZ[index], bufferMass[index],
			softeningSqr);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
	
// kernel calculating new position and velocity for n-bodies
__kernel void nbody_kernel(float timeDelta,
	MEMORY_TYPE_AOS float4* oldBodyInfo, // pos XYZ, mass
	MEMORY_TYPE_SOA vector* oldPosX,
	MEMORY_TYPE_SOA vector* oldPosY,
	MEMORY_TYPE_SOA vector* oldPosZ,
	MEMORY_TYPE_SOA vector* mass,
	__global float4* newBodyInfo,
	MEMORY_TYPE_AOS float4* oldVel, // XYZ, W unused
	MEMORY_TYPE_SOA vector* oldVelX,
	MEMORY_TYPE_SOA vector* oldVelY,
	MEMORY_TYPE_SOA vector* oldVelZ,
	__global float4* newVel, // XYZ, W set to 0.f
	float damping, 
	float softeningSqr)
{
	// indices
	int n = get_global_size(0);
	int gtid = get_global_id(0);
		
	// buffers for bodies info processed by the work group
	__local vector bufferPosX[WORK_GROUP_SIZE_X];
	__local vector bufferPosY[WORK_GROUP_SIZE_X];
	__local vector bufferPosZ[WORK_GROUP_SIZE_X];
	__local vector bufferMass[WORK_GROUP_SIZE_X];
	
    // each thread holds a position/mass of the body it represents
    float bodyPos[3];
    float bodyVel[3];
	vector bodyAcc[3];
	float bodyMass;

	// clear acceleration
	bodyAcc[0] = bodyAcc[1] = bodyAcc[2] = (vector)0.f;
	
	// load data
	loadThreadData(oldBodyInfo, (MEMORY_TYPE_SOA float*)oldPosX, (MEMORY_TYPE_SOA float*)oldPosY, (MEMORY_TYPE_SOA float*)oldPosZ, (MEMORY_TYPE_SOA float*)mass,
		oldVel, (MEMORY_TYPE_SOA float*)oldVelX, (MEMORY_TYPE_SOA float*)oldVelY, (MEMORY_TYPE_SOA float*)oldVelZ, // velocity
		bodyPos, bodyVel, &bodyMass, // values to be filled
		get_group_id(0) * WORK_GROUP_SIZE_X, // start index
		min(WORK_GROUP_SIZE_X * ((int)get_group_id(0) + 1) - 1, n - 1)); // end index
	
	int blocks = n / (WORK_GROUP_SIZE_X * VECTOR_TYPE); // each calculates effect of WORK_GROUP_SIZE_X atoms to currect, i.e. thread's, one
	// start the calculation, process whole blocks
	for (int i = 0; i < blocks; i++) {
		processCompleteBlock(
			oldBodyInfo, oldPosX, oldPosY, oldPosZ, mass,
			bufferPosX, bufferPosY, bufferPosZ, bufferMass,
			bodyAcc, bodyPos, 
			softeningSqr, 
			i * WORK_GROUP_SIZE_X); // start index is the first body in the block being processed
	}
	
	// at the end, do the final block which is shorter than WORK_GROUP_SIZE_X
	processFinalBlock(
		oldBodyInfo, oldPosX, oldPosY, oldPosZ, mass,
		bufferPosX, bufferPosY, bufferPosZ, bufferMass,
		bodyAcc, bodyPos, 
		softeningSqr,
		blocks * WORK_GROUP_SIZE_X, (n / VECTOR_TYPE) - 1);
	
	// sum elements of acceleration vector, if any
	float resAccX, resAccY, resAccZ;
	resAccX = resAccY = resAccZ = 0.f;
	for (int i = 0; i < VECTOR_TYPE; i++) 
	{
		resAccX += ((float*)&bodyAcc[0])[i];
		resAccY += ((float*)&bodyAcc[1])[i];
		resAccZ += ((float*)&bodyAcc[2])[i];
	}
		
	// 'export' result
	if (gtid < n) {
		// calculate resulting position 	
		float resPosX = bodyPos[0] + timeDelta * bodyVel[0] + damping * timeDelta * timeDelta * resAccX;
		float resPosY = bodyPos[1] + timeDelta * bodyVel[1] + damping * timeDelta * timeDelta * resAccY;
		float resPosZ = bodyPos[2] + timeDelta * bodyVel[2] + damping * timeDelta * timeDelta * resAccZ;
		newBodyInfo[gtid] = (float4)(resPosX, resPosY, resPosZ, bodyMass);
		// calculate resulting velocity	
		float resVelX = bodyVel[0] + timeDelta * resAccX;
		float resVelY = bodyVel[1] + timeDelta * resAccY;
		float resVelZ = bodyVel[2] + timeDelta * resAccZ;
		newVel[gtid] = (float4)(resVelX, resVelY, resVelZ, 0.f);
	}
}