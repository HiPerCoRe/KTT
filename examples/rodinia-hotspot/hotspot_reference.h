#pragma once

#include "tuner_api.h"
#include "hotspot.h"
#include "hotspot_tunable.h"
#define BLOCK_SIZE_C BLOCK_SIZE_REF
#define BLOCK_SIZE_R BLOCK_SIZE_REF

class referenceHotspot : public ktt::ReferenceClass {
  public:

    //Constructor creates internal structures and setups the tuning environment
    // it takes arguments passed from command line arguments and initializes all data variables

    referenceHotspot(size_t resultId, int grid_rows, int grid_cols, int pyramid_height, int total_iterations, char* tfile, char*pfile, int borderCols, int borderRows, int smallBlockCol, int smallBlockRow, int blockCols, int blockRows, float grid_height, float grid_width, float Cap, float Rx, float Ry, float Rz, float max_slope, float step) : ReferenceClass() {

      // Initialize data variables
      // we need some of them calculated to set the ndRange for example
      this->resultArgumentId = resultId;
      this->grid_rows = grid_rows;
      this->grid_cols = grid_cols;
      this->pyramid_height = pyramid_height;
      this->total_iterations = total_iterations;
      size = grid_rows*grid_cols;
      this->blockCols = blockCols;
      this->blockRows = blockRows;
      this->borderCols = borderCols;
      this->borderRows = borderRows; 
      this->grid_height = grid_height;
      this->grid_width = grid_width;
      this->Cap = Cap;
      this->Rx = Rx;
      this->Ry = Ry;
      this->Rz = Rz;
      this->max_slope = max_slope;
      this->step = step;

      tempSrc = (float*) malloc(size* sizeof(float));
      tempDst = (float*) malloc(size* sizeof(float));
      power = (float*) malloc(size* sizeof(float));

      // Read input data from disk
      readinput(tempSrc, grid_rows, grid_cols, tfile);
      readinput(power, grid_rows, grid_cols, pfile);
      srcPosition = true;
    }

    //launchComputation is responsible for actual execution of tuned kernel */
    void computeResult() override {

      int t;
      for (t = 0; t < total_iterations; t += pyramid_height) {

        // Specify kernel arguments
        int iter = MIN(pyramid_height, total_iterations - t);
        // Run the kernel, order of tempSrc and tempDst swaping in each iteration
        compute_tran_temp(tempDst, iter, tempSrc, power, grid_rows, grid_cols);

        // Swap the source and destination temperatures
        if (t + pyramid_height < total_iterations)
        {
          float* tmp = tempSrc;
          tempSrc = tempDst;
          tempDst = tmp;
          srcPosition = !srcPosition;
        }
      }

      // Write final output to output file
      writeoutput(tempDst, grid_rows, grid_cols, ofile);

    }
    void* getData(const size_t argumentId) override {
      if (argumentId == resultArgumentId) {
        if (srcPosition)
          return (void*)tempDst;
        else
          return (void*)tempSrc;
      }
      throw std::runtime_error("No result available for specified argument id");
    }

    size_t getNumberOfElements(const size_t argumentId) const override {
      return size;
    }

  private:

    size_t resultArgumentId;
    int size;
    int grid_rows;
    int grid_cols;
    float* tempSrc; 
    float* tempDst; 
    float* power; 
    int pyramid_height; //number of iterations
    int total_iterations;


    int blockRows, blockCols;
    int smallBlockRow, smallBlockCol;
    int borderRows, borderCols;
    float grid_height, grid_width;

    float Cap;
    float Rx, Ry, Rz;
    float max_slope;
    float step;
    bool srcPosition;

    
    char* ofile = (char*) "reference_output.txt"; //the name of output file for temperatures

    /* Transient solver driver routine: simply converts the heat 
     * transfer differential equations to difference equations 
     * and solves the difference equations by iterating
     */
    void compute_tran_temp(float* result, int num_iterations, float* temp, float* power, int row, int col) 
    {
#ifdef VERBOSE
      int i = 0;
#endif

      float Rx_1=1.f/Rx;
      float Ry_1=1.f/Ry;
      float Rz_1=1.f/Rz;
      float Cap_1 = step/Cap;
#ifdef VERBOSE
      fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
      fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
#endif

          for (int i = 0; i < num_iterations ; i++)
          {
#ifdef VERBOSE
            fprintf(stdout, "iteration %d\n", i);
#endif
            single_iteration(result, temp, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1);
          }	
    }

    void single_iteration(float* result, float* temp, float* power, int row, int col,
        float Cap_1, float Rx_1, float Ry_1, float Rz_1)
    {
      float delta;
      int r, c;
      int chunk;
      int num_chunk = row*col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
      int chunks_in_row = col/BLOCK_SIZE_C;
      int chunks_in_col = row/BLOCK_SIZE_R;

      for ( chunk = 0; chunk < num_chunk; ++chunk )
      {
        int r_start = BLOCK_SIZE_R*(chunk/chunks_in_col);
        int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row); 
        int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
        int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;

        if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col )
        {
          for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
            for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
              /* Corner 1 */
              if ( (r == 0) && (c == 0) ) {
                delta = (Cap_1) * (power[0] +
                    (temp[1] - temp[0]) * Rx_1 +
                    (temp[col] - temp[0]) * Ry_1 +
                    (amb_temp - temp[0]) * Rz_1);
              }	/* Corner 2 */
              else if ((r == 0) && (c == col-1)) {
                delta = (Cap_1) * (power[c] +
                    (temp[c-1] - temp[c]) * Rx_1 +
                    (temp[c+col] - temp[c]) * Ry_1 +
                    (   amb_temp - temp[c]) * Rz_1);
              }	/* Corner 3 */
              else if ((r == row-1) && (c == col-1)) {
                delta = (Cap_1) * (power[r*col+c] + 
                    (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 + 
                    (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 + 
                    (   amb_temp - temp[r*col+c]) * Rz_1);					
              }	/* Corner 4	*/
              else if ((r == row-1) && (c == 0)) {
                delta = (Cap_1) * (power[r*col] + 
                    (temp[r*col+1] - temp[r*col]) * Rx_1 + 
                    (temp[(r-1)*col] - temp[r*col]) * Ry_1 + 
                    (amb_temp - temp[r*col]) * Rz_1);
              }	/* Edge 1 */
              else if (r == 0) {
                delta = (Cap_1) * (power[c] + 
                    (temp[c+1] + temp[c-1] - 2.0f*temp[c]) * Rx_1 + 
                    (temp[col+c] - temp[c]) * Ry_1 + 
                    (amb_temp - temp[c]) * Rz_1);
              }	/* Edge 2 */
              else if (c == col-1) {
                delta = (Cap_1) * (power[r*col+c] + 
                    (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0f*temp[r*col+c]) * Ry_1 + 
                    (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 + 
                    (amb_temp - temp[r*col+c]) * Rz_1);
              }	/* Edge 3 */
              else if (r == row-1) {
                delta = (Cap_1) * (power[r*col+c] + 
                    (temp[r*col+c+1] + temp[r*col+c-1] - 2.0f*temp[r*col+c]) * Rx_1 + 
                    (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 + 
                    (amb_temp - temp[r*col+c]) * Rz_1);
              }	/* Edge 4 */
              else if (c == 0) {
                delta = (Cap_1) * (power[r*col] + 
                    (temp[(r+1)*col] + temp[(r-1)*col] - 2.0f*temp[r*col]) * Ry_1 + 
                    (temp[r*col+1] - temp[r*col]) * Rx_1 + 
                    (amb_temp - temp[r*col]) * Rz_1);
              }
              result[r*col+c] =temp[r*col+c]+ delta;
            }
          }
          continue;
        }

        for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
          for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
            /* Update Temperatures */
            result[r*col+c] =temp[r*col+c]+ 
              ( Cap_1 * (power[r*col+c] + 
                         (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.f*temp[r*col+c]) * Ry_1 + 
                         (temp[r*col+c+1] + temp[r*col+c-1] - 2.f*temp[r*col+c]) * Rx_1 + 
                         (amb_temp - temp[r*col+c]) * Rz_1));
          }
        }
      }
    }



};
