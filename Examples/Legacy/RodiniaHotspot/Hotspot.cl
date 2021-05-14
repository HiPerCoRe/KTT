#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))

__kernel void hotspot(  int iteration,  //number of iteration
    global float *power,   //power input
    global float *temp_src,    //temperature input/output
    global float *temp_dst,    //temperature input/output
    int grid_cols,  //Col of grid
    int grid_rows,  //Row of grid
    int border_cols,  // border offset 
    int border_rows,  // border offset
    float Cap,      //Capacitance
    float Rx, 
    float Ry, 
    float Rz, 
    float step) {

  local float temp_on_cuda[BLOCK_SIZE_ROWS][BLOCK_SIZE_COLS];
#if LOCAL_MEMORY
  local float power_on_cuda[BLOCK_SIZE_ROWS][BLOCK_SIZE_COLS];
#endif
  local float temp_t[BLOCK_SIZE_ROWS][BLOCK_SIZE_COLS]; // saving temporary temperature result

  float amb_temp = 80.0f;
  float step_div_Cap;
  float Rx_1,Ry_1,Rz_1;

  int bx = get_group_id(0);
  int by = get_group_id(1);

  int tx = get_local_id(0);
  int ty = get_local_id(1);

  step_div_Cap=step/Cap;

  Rx_1=1.0f/Rx;
  Ry_1=1.0f/Ry;
  Rz_1=1.0f/Rz;

  // each block finally computes result for a small block
  // after N iterations. 
  // it is the non-overlapping small blocks that cover 
  // all the input data

  // calculate the small block size
  int small_block_rows = BLOCK_SIZE_ROWS-iteration*2;//EXPAND_RATE
  int small_block_cols = BLOCK_SIZE_COLS-iteration*2;//EXPAND_RATE

  // calculate the boundary for the block according to 
  // the boundary of its small block
  int blkY = small_block_rows*by-border_rows;
  int blkX = small_block_cols*bx-border_cols;
  int blkYmax = blkY+BLOCK_SIZE_ROWS-1;
  int blkXmax = blkX+BLOCK_SIZE_COLS-1;

  // calculate the global thread coordination
  int yidx = blkY+ty;
  int xidx = blkX+tx;

  // load data if it is within the valid input range
  int loadYidx=yidx, loadXidx=xidx;
  int index = grid_cols*loadYidx+loadXidx;


  for (int j = 0; j < BLOCK_SIZE_ROWS/WORK_GROUP_Y; j++)
  {
    // calculate the global thread coordination
    yidx = blkY+ty;
    xidx = blkX+tx;

    // load data if it is within the valid input range
    loadYidx=yidx;
    loadXidx=xidx;
    index = grid_cols*loadYidx+loadXidx;

    if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
      temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
#if LOCAL_MEMORY
      power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
#endif
    }
    ty += WORK_GROUP_Y;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  ty = get_local_id(1);
  // effective range within this block that falls within 
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validYmin = (blkY < 0) ? -blkY : 0;
  int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE_ROWS-1-(blkYmax-grid_rows+1) : BLOCK_SIZE_ROWS-1;
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE_COLS-1-(blkXmax-grid_cols+1) : BLOCK_SIZE_COLS-1;
  int N, S, E, W;

  bool computed[BLOCK_SIZE_ROWS/WORK_GROUP_Y];
#if LOOP_UNROLL
  if (iteration == 1)
  {
    int i = 0;
    ty = get_local_id(1);
    N = ty-1;
    S = ty+1;
    W = tx-1;
    E = tx+1;

    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    for (int j = 0; j < BLOCK_SIZE_ROWS/WORK_GROUP_Y; j++)
    {
      computed[j] = false;
      if( IN_RANGE(tx, i+1, BLOCK_SIZE_COLS-i-2) &&  \
          IN_RANGE(ty, i+1, BLOCK_SIZE_ROWS-i-2) &&  \
          IN_RANGE(tx, validXmin, validXmax) && \
          IN_RANGE(ty, validYmin, validYmax) ) {
        computed[j] = true;
#if LOCAL_MEMORY
        temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
            (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 +
            (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 +
            (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
#else
        yidx = blkY+ty;
        index = grid_cols*yidx+xidx;
        temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power[index] +
            (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 +
            (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 +
            (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
#endif /* LOCAL_MEMORY */
      }
      ty += WORK_GROUP_Y;
      N = ty-1;
      S = ty+1;
      N = (N < validYmin) ? validYmin : N;
      S = (S > validYmax) ? validYmax : S;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ty = get_local_id(1);
    for (int j = 0; j < BLOCK_SIZE_ROWS/WORK_GROUP_Y; j++)
    {
      if(j == BLOCK_SIZE_ROWS/WORK_GROUP_Y -1)
        break;
      if(computed[j])	 //Assign the computation range
      {
        temp_on_cuda[ty][tx]= temp_t[ty][tx];
      }
      ty += WORK_GROUP_Y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  else if (iteration == 2)
  {
    int i = 0;
    ty = get_local_id(1);
    N = ty-1;
    S = ty+1;
    W = tx-1;
    E = tx+1;

    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    for (int j = 0; j < BLOCK_SIZE_ROWS/WORK_GROUP_Y; j++)
    {
      computed[j] = false;
      if( IN_RANGE(tx, i+1, BLOCK_SIZE_COLS-i-2) &&  \
          IN_RANGE(ty, i+1, BLOCK_SIZE_ROWS-i-2) &&  \
          IN_RANGE(tx, validXmin, validXmax) && \
          IN_RANGE(ty, validYmin, validYmax) ) {
        computed[j] = true;
#if LOCAL_MEMORY
        temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
            (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 +
            (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 +
            (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
#else
        yidx = blkY+ty;
        index = grid_cols*yidx+xidx;
        temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power[index] +
            (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 +
            (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 +
            (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
#endif /* LOCAL_MEMORY */
      }
      ty += WORK_GROUP_Y;
      N = ty-1;
      S = ty+1;
      N = (N < validYmin) ? validYmin : N;
      S = (S > validYmax) ? validYmax : S;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ty = get_local_id(1);
    for (int j = 0; j < BLOCK_SIZE_ROWS/WORK_GROUP_Y; j++)
    {
      if(computed[j])	 //Assign the computation range
      {
        temp_on_cuda[ty][tx]= temp_t[ty][tx];
      }
      ty += WORK_GROUP_Y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    i = 1;
    ty = get_local_id(1);
    N = ty-1;
    S = ty+1;
    W = tx-1;
    E = tx+1;

    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    for (int j = 0; j < BLOCK_SIZE_ROWS/WORK_GROUP_Y; j++)
    {
      computed[j] = false;
      if( IN_RANGE(tx, i+1, BLOCK_SIZE_COLS-i-2) &&  \
          IN_RANGE(ty, i+1, BLOCK_SIZE_ROWS-i-2) &&  \
          IN_RANGE(tx, validXmin, validXmax) && \
          IN_RANGE(ty, validYmin, validYmax) ) {
        computed[j] = true;
#if LOCAL_MEMORY
        temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
            (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 +
            (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 +
            (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
#else
        yidx = blkY+ty;
        index = grid_cols*yidx+xidx;
        temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power[index] +
            (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 +
            (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 +
            (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
#endif /* LOCAL_MEMORY */
      }
      ty += WORK_GROUP_Y;
      N = ty-1;
      S = ty+1;
      N = (N < validYmin) ? validYmin : N;
      S = (S > validYmax) ? validYmax : S;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ty = get_local_id(1);
    for (int j = 0; j < BLOCK_SIZE_ROWS/WORK_GROUP_Y; j++)
    {
      if(j == BLOCK_SIZE_ROWS/WORK_GROUP_Y -1)
        break;
      if(computed[j])	 //Assign the computation range
      {
        temp_on_cuda[ty][tx]= temp_t[ty][tx];
      }
      ty += WORK_GROUP_Y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  else {
#endif /* LOOP_UNROLL */
    for (int i=0; i<iteration ; i++){
      ty = get_local_id(1);
      int N = ty-1;
      int S = ty+1;
      int W = tx-1;
      int E = tx+1;

      N = (N < validYmin) ? validYmin : N;
      S = (S > validYmax) ? validYmax : S;
      W = (W < validXmin) ? validXmin : W;
      E = (E > validXmax) ? validXmax : E;
      for (int j = 0; j < BLOCK_SIZE_ROWS/WORK_GROUP_Y; j++)
      {
        computed[j] = false;
        if( IN_RANGE(tx, i+1, BLOCK_SIZE_COLS-i-2) &&  \
            IN_RANGE(ty, i+1, BLOCK_SIZE_ROWS-i-2) &&  \
            IN_RANGE(tx, validXmin, validXmax) && \
            IN_RANGE(ty, validYmin, validYmax) ) {
          computed[j] = true;
#if LOCAL_MEMORY
          temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
              (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 + 
              (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 + 
              (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
#else
          yidx = blkY+ty;
          index = grid_cols*yidx+xidx;
          temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power[index] + 
              (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 + 
              (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 + 
              (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
#endif /* LOCAL_MEMORY */
        }
	ty += WORK_GROUP_Y;
        N = ty-1;
        S = ty+1;
        N = (N < validYmin) ? validYmin : N;
        S = (S > validYmax) ? validYmax : S;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      ty = get_local_id(1);
      for (int j = 0; j < BLOCK_SIZE_ROWS/WORK_GROUP_Y; j++)
      {
        if(i==iteration-1 && j == BLOCK_SIZE_ROWS/WORK_GROUP_Y -1)
          break;
        if(computed[j])	 //Assign the computation range
        {
          temp_on_cuda[ty][tx]= temp_t[ty][tx];
        }
        ty += WORK_GROUP_Y;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
#if LOOP_UNROLL
  }
#endif
  // update the global memory
  // after the last iteration, only threads coordinated within the 
  // small block perform the calculation and switch on ``computed''
  ty = get_local_id(1);
  xidx = blkX+tx;
  for (int j = 0; j < BLOCK_SIZE_ROWS/WORK_GROUP_Y; j++)
  {
    // calculate the global thread coordination
    yidx = blkY+ty;

    // load data if it is within the valid input range
    index = grid_cols*yidx+xidx;
    if (computed[j]){
      temp_dst[index]= temp_t[ty][tx];		
    }
    ty += WORK_GROUP_Y;
  }
}
