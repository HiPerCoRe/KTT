#if USE_CONSTANT_MEMORY == 0
    #define MEMORY_TYPE_AOS __global const
    #define MEMORY_TYPE_SOA __global const
#elif USE_CONSTANT_MEMORY == 1
    #if USE_SOA == 0
        #define MEMORY_TYPE_AOS __constant
        #define MEMORY_TYPE_SOA __global const
    #elif USE_SOA > 0
        #define MEMORY_TYPE_AOS __global const
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
		float invDist = half_rsqrt(distSqr);
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
	   
		vector invDist = half_rsqrt(distanceX * distanceX + distanceY * distanceY + distanceZ * distanceZ + softeningSqr);
		vector f = bufferMass * invDist * invDist * invDist;

		bodyAcc[0] += distanceX * f;
		bodyAcc[1] += distanceY * f;
		bodyAcc[2] += distanceZ * f;
	}
	#endif // USE_SOA == 0
}

// method to calculate acceleration caused by body J
void updateAccGM(vector bodyAcc[3],
	float bodyPos[3], // position of body I
 	MEMORY_TYPE_AOS float4* oldBodyInfo, // global data; [X,Y,Z,mass]
	MEMORY_TYPE_SOA vector* oldPosX,
	MEMORY_TYPE_SOA vector* oldPosY,
	MEMORY_TYPE_SOA vector* oldPosZ,
	MEMORY_TYPE_SOA vector* mass, 
	int index,
	float softeningSqr) // to avoid infinities and zero division
{
	#if USE_SOA == 0
	{
		updateAcc(bodyAcc, bodyPos, 
			oldBodyInfo[index].x,
			oldBodyInfo[index].y,
			oldBodyInfo[index].z,
			oldBodyInfo[index].w,
			softeningSqr);
	}
	#else // USE_SOA != 0
	{
		updateAcc(bodyAcc, bodyPos,
			oldPosX[index],
			oldPosY[index],
			oldPosZ[index],
			mass[index],
			softeningSqr);
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
	float bodyPos[OUTER_UNROLL_FACTOR][3], float bodyVel[OUTER_UNROLL_FACTOR][3], float bodyMass[OUTER_UNROLL_FACTOR]) // thread data
{
	int index = get_global_id(0) * OUTER_UNROLL_FACTOR;
#if INNER_UNROLL_FACTOR2 > 0
	# pragma unroll INNER_UNROLL_FACTOR2
#endif
	for (int j = 0; j < OUTER_UNROLL_FACTOR; ++j) {
		#if USE_SOA == 0
		{
			// store 'thread specific' body info to registers
			bodyPos[j][0] = oldBodyInfo[index + j].x;
			bodyPos[j][1] = oldBodyInfo[index + j].y;
			bodyPos[j][2] = oldBodyInfo[index + j].z;
			
			bodyVel[j][0] = oldVel[index + j].x;
			bodyVel[j][1] = oldVel[index + j].y;
			bodyVel[j][2] = oldVel[index + j].z;
			
			bodyMass[j] = oldBodyInfo[index + j].w;
		}
		#else // USE_SOA != 0
		{
			// store 'thread specific' body info to registers
			bodyPos[j][0] = oldPosX[index + j];
			bodyPos[j][1] = oldPosY[index + j];
			bodyPos[j][2] = oldPosZ[index + j];
			
			bodyVel[j][0] = oldVelX[index + j];
			bodyVel[j][1] = oldVelY[index + j];
			bodyVel[j][2] = oldVelZ[index + j];
			
			bodyMass[j] = mass[index + j];
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
	__local vector bufferPosX[WORK_GROUP_SIZE_X], // buffers
	__local vector bufferPosY[WORK_GROUP_SIZE_X],
	__local vector bufferPosZ[WORK_GROUP_SIZE_X],
	__local vector bufferMass[WORK_GROUP_SIZE_X],
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

	
// kernel calculating new position and velocity for n-bodies
#if VECTOR_SIZE > 1
__kernel __attribute__((vec_type_hint(vector)))
#endif
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
	float softeningSqr,
	int n)
{
	// buffers for bodies info processed by the work group
	__local vector bufferPosX[WORK_GROUP_SIZE_X];
	__local vector bufferPosY[WORK_GROUP_SIZE_X];
	__local vector bufferPosZ[WORK_GROUP_SIZE_X];
	__local vector bufferMass[WORK_GROUP_SIZE_X];
	
    // each thread holds a position/mass of the body it represents
    float bodyPos[OUTER_UNROLL_FACTOR][3];
    float bodyVel[OUTER_UNROLL_FACTOR][3];
	vector bodyAcc[OUTER_UNROLL_FACTOR][3];
	float bodyMass[OUTER_UNROLL_FACTOR];

	// clear acceleration
#if INNER_UNROLL_FACTOR2 > 0
	# pragma unroll INNER_UNROLL_FACTOR2
#endif
	for (int j = 0; j < OUTER_UNROLL_FACTOR; ++j) {
		bodyAcc[j][0] = bodyAcc[j][1] = bodyAcc[j][2] = (vector)0.f;
	}
	
	// load data
	loadThreadData(oldBodyInfo, (MEMORY_TYPE_SOA float*)oldPosX, (MEMORY_TYPE_SOA float*)oldPosY, (MEMORY_TYPE_SOA float*)oldPosZ, (MEMORY_TYPE_SOA float*)mass,
		oldVel, (MEMORY_TYPE_SOA float*)oldVelX, (MEMORY_TYPE_SOA float*)oldVelY, (MEMORY_TYPE_SOA float*)oldVelZ, // velocity
		bodyPos, bodyVel, &bodyMass); // values to be filled
	
	int blocks = n / (WORK_GROUP_SIZE_X * VECTOR_TYPE); // each calculates effect of WORK_GROUP_SIZE_X atoms to currect, i.e. thread's, one
	// start the calculation, process whole blocks
	for (int i = 0; i < blocks; i++) {
		#if LOCAL_MEM == 1 
			// load new values to buffer.
			// We know that all threads can be used now, so no condition is necessary
			fillBuffers(oldBodyInfo, oldPosX, oldPosY, oldPosZ, mass, bufferPosX, bufferPosY, bufferPosZ, bufferMass, i * WORK_GROUP_SIZE_X);
			barrier(CLK_LOCAL_MEM_FENCE);
		#endif // LOCAL_MEM == 1 
			// calculate the acceleration between the thread body and each other body loaded to buffer
        #if INNER_UNROLL_FACTOR1 > 0
		# pragma unroll INNER_UNROLL_FACTOR1
        #endif
		for(int index =  0; index < WORK_GROUP_SIZE_X; index++) {
            #if INNER_UNROLL_FACTOR2 > 0
			# pragma unroll INNER_UNROLL_FACTOR2
            #endif
			for (int j = 0; j < OUTER_UNROLL_FACTOR; ++j) {
				#if LOCAL_MEM == 1
					updateAcc(bodyAcc[j], bodyPos[j],
						bufferPosX[index], bufferPosY[index], bufferPosZ[index], bufferMass[index],
						softeningSqr);
				#else // LOCAL_MEM != 1
					updateAccGM(bodyAcc[j], bodyPos[j],
						oldBodyInfo, oldPosX, oldPosY, oldPosZ, mass,
						i * WORK_GROUP_SIZE_X + index,
						softeningSqr);
				#endif // LOCAL_MEM == 1
			}
		}
        #if  LOCAL_MEM == 1
		barrier(CLK_LOCAL_MEM_FENCE); // sync threads
        #endif

	}
	
	// sum elements of acceleration vector, if any
	float resAccX, resAccY, resAccZ;
	int index = get_global_id(0) * OUTER_UNROLL_FACTOR;
    #if INNER_UNROLL_FACTOR2 > 0
	# pragma unroll INNER_UNROLL_FACTOR2
    #endif
	for (int j = 0; j < OUTER_UNROLL_FACTOR; ++j) {
		resAccX = resAccY = resAccZ = 0.f;
		for (int i = 0; i < VECTOR_TYPE; i++) 
		{
			resAccX += ((float*)&bodyAcc[j][0])[i];
			resAccY += ((float*)&bodyAcc[j][1])[i];
			resAccZ += ((float*)&bodyAcc[j][2])[i];
		}
			
		// 'export' result
		// calculate resulting position 	
		float resPosX = bodyPos[j][0] + timeDelta * bodyVel[j][0] + damping * timeDelta * timeDelta * resAccX;
		float resPosY = bodyPos[j][1] + timeDelta * bodyVel[j][1] + damping * timeDelta * timeDelta * resAccY;
		float resPosZ = bodyPos[j][2] + timeDelta * bodyVel[j][2] + damping * timeDelta * timeDelta * resAccZ;
		newBodyInfo[index + j] = (float4)(resPosX, resPosY, resPosZ, bodyMass[j]);
		// calculate resulting velocity	
		float resVelX = bodyVel[j][0] + timeDelta * resAccX;
		float resVelY = bodyVel[j][1] + timeDelta * resAccY;
		float resVelZ = bodyVel[j][2] + timeDelta * resAccZ;
		newVel[index + j] = (float4)(resVelX, resVelY, resVelZ, 0.f);
	}
}
