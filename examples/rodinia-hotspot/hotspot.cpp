#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tuner_api.h"
#include "hotspot.h"
#include "hotspot_tunable.h"
#include "hotspot_reference.h"

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/rodinia-hotspot/hotspot_kernel.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../examples/rodinia-hotspot/hotspot_reference_kernel.cl"
#else
    #define KTT_KERNEL_FILE "../../examples/rodinia-hotspot/hotspot_kernel.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../../examples/rodinia-hotspot/hotspot_reference_kernel.cl"
#endif

int main(int argc, char** argv)
{
  // Initialize platform and device index
  ktt::PlatformIndex platformIndex = 0;
  ktt::DeviceIndex deviceIndex = 0;
  std::string kernelFile = KTT_KERNEL_FILE;
  std::string referenceKernelFile = KTT_REFERENCE_KERNEL_FILE;

  if (argc >= 2)
  {
    platformIndex = std::stoul(std::string{ argv[1] });
    if (argc >= 3)
    {
      deviceIndex = std::stoul(std::string{ argv[2] });
      if (argc >= 4)
      {
        kernelFile = std::string{ argv[3] };
      }
    }
  }


  // Declare data variables necessary for command line arguments parsing
  // all other (and these also) are in hotspotTunable class as member variables
  int grid_rows,grid_cols = 1024;
  char *tfile, *pfile, *ofile;

  int pyramid_height = 2; //number of iterations
  int total_iterations = 60;

  if (argc < 9)
    usage(argc, argv);
  if((grid_rows = atoi(argv[4]))<=0||
      (grid_cols = atoi(argv[4]))<=0||
      //    (pyramid_height = atoi(argv[5]))<=0||
      (total_iterations = atoi(argv[5]))<=0)
    usage(argc, argv);

  //input file of temperatures
  tfile=argv[6];
  //input file of power
  pfile=argv[7];
  //output file for temperatures
  ofile=argv[8];
  char refofile[] = "reference_output.txt";

  // Create tuner object for chosen platform and device
  ktt::Tuner tuner(platformIndex, deviceIndex);
  tuner.setCompilerOptions("-I./");
  tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

  std::vector<float> tempSrc; 
  std::vector<float> tempDst; 
  std::vector<float> power; 

  int borderRows, borderCols;
  float grid_height, grid_width;
  long size = grid_rows*grid_cols;

  float Cap;
  float Rx, Ry, Rz;
  float max_slope;
  float step;

  // --------------- pyramid parameters --------------- 
  grid_height = chip_height / grid_rows;
  grid_width = chip_width / grid_cols;

  Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  Rx = grid_width / (2.0f * K_SI * t_chip * grid_height);
  Ry = grid_height / (2.0f * K_SI * t_chip * grid_width);
  Rz = t_chip / (K_SI * grid_height * grid_width);

  max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  step = PRECISION / max_slope;

  tempSrc = std::vector<float>(size, 0.0);
  tempDst = std::vector<float>(size, 0.0);
  power = std::vector<float>(size, 0.0);

  // Read input data from disk
  readinput(tempSrc, grid_rows, grid_cols, tfile);
  readinput(power, grid_rows, grid_cols, pfile);

  // Declare kernel parameters
  ktt::KernelId kernelId;
  ktt::KernelId referenceKernelId;
  // Total NDRange size matches number of grid points
  const ktt::DimensionVector ndRangeDimensions(1, 1, 1);
  const ktt::DimensionVector workGroupDimensions(1, 1, 1);
  // Add two kernels to tuner, one of the kernels acts as reference kernel
  // 
  kernelId = tuner.addKernelFromFile(kernelFile, std::string("hotspot"), ndRangeDimensions, workGroupDimensions);
  referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, std::string("hotspot"), ndRangeDimensions, workGroupDimensions);
  // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
  tuner.addParameter(kernelId, "BLOCK_SIZE_ROWS", {8, 16, 32, 64});
  tuner.addParameter(kernelId, "BLOCK_SIZE_COLS", {8, 16, 32, 64});
  tuner.addParameter(kernelId, "PYRAMID_HEIGHT", {1, 2, 4, 8}); //, 2, 4});
  tuner.addParameter(kernelId, "WORK_GROUP_Y", {4, 8, 16, 32, 64});
  tuner.addParameter(kernelId, "LOCAL_MEMORY", {0, 1});
  tuner.addParameter(kernelId, "LOOP_UNROLL", {0,1});
  // Add conditions
  auto enoughToCompute = [](const std::vector<size_t>& vector) {return vector.at(0)/(vector.at(2)*2) > 1 && vector.at(1)/(vector.at(2)*2) > 1;};
  tuner.addConstraint(kernelId, {"BLOCK_SIZE_COLS", "WORK_GROUP_Y", "PYRAMID_HEIGHT"}, enoughToCompute);
  auto workGroupSmaller = [](const std::vector<size_t>& vector) {return vector.at(0)<=vector.at(1);};
  auto workGroupDividable = [](const std::vector<size_t>& vector) {return vector.at(1)%vector.at(0) == 0;};
  tuner.addConstraint(kernelId, {"WORK_GROUP_Y", "BLOCK_SIZE_ROWS"}, workGroupSmaller);
  tuner.addConstraint(kernelId, {"WORK_GROUP_Y", "BLOCK_SIZE_ROWS"}, workGroupDividable);
  // Add all arguments utilized by kernels
  ktt::ArgumentId iterationId = tuner.addArgumentScalar(0);
  ktt::ArgumentId powerId = tuner.addArgumentVector(power, ktt::ArgumentAccessType::ReadOnly);
  ktt::ArgumentId tempSrcId = tuner.addArgumentVector(std::vector<float>(tempSrc), ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId tempDstId = tuner.addArgumentVector(std::vector<float>(tempDst), ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId grid_colsId = tuner.addArgumentScalar(grid_cols);
  ktt::ArgumentId grid_rowsId = tuner.addArgumentScalar(grid_rows);

  ktt::ArgumentId borderColsId = tuner.addArgumentScalar(borderCols);
  ktt::ArgumentId borderRowsId = tuner.addArgumentScalar(borderRows);
  ktt::ArgumentId CapId = tuner.addArgumentScalar(Cap);
  ktt::ArgumentId RxId = tuner.addArgumentScalar(Rx);
  ktt::ArgumentId RyId = tuner.addArgumentScalar(Ry);
  ktt::ArgumentId RzId = tuner.addArgumentScalar(Rz);
  ktt::ArgumentId stepId = tuner.addArgumentScalar(step);


  // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
  tuner.setKernelArguments(kernelId,
      std::vector<ktt::ArgumentId>{ iterationId, 
      powerId, tempSrcId, tempDstId,
      grid_colsId, grid_rowsId, borderColsId, borderRowsId,
      CapId, RxId, RyId, RzId, stepId });
  tuner.setKernelArguments(referenceKernelId,
      std::vector<ktt::ArgumentId>{ iterationId, 
      powerId, tempSrcId, tempDstId,
      grid_colsId, grid_rowsId, borderColsId, borderRowsId,
      CapId, RxId, RyId, RzId, stepId });
  
  tunableHotspot* hotspot = new tunableHotspot(&tuner, kernelId, grid_rows, grid_cols, total_iterations, size, ofile, tempSrcId, tempDstId, iterationId, borderRowsId, borderColsId);

  referenceManipulator* referenceHotspot = new referenceManipulator(&tuner, referenceKernelId, grid_rows, grid_cols, total_iterations, size, refofile, tempSrcId, tempDstId, iterationId, borderRowsId, borderColsId);

  tuner.setTuningManipulator(hotspot->getKernelId(), std::unique_ptr<tunableHotspot>(hotspot));
  tuner.setTuningManipulator(referenceHotspot->getKernelId(), std::unique_ptr<referenceManipulator>(referenceHotspot));

  // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
  tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01f);

  // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
  tuner.setReferenceKernel(kernelId, referenceKernelId, {}, std::vector<ktt::ArgumentId>{tempDstId});

  hotspot->tune();

  tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
  tuner.printResult(kernelId, "hotspot_output.csv", ktt::PrintFormat::CSV);

  return 0;
}



void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <platform index> <device index> <kernel file> <grid_rows/grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
  fprintf(stderr, "\t <platform index> - the index of the platform, starting from 0\n");
  fprintf(stderr, "\t <device index> - the index of the device, starting from 0\n");
  fprintf(stderr, "\t <kernel file> - the path to kernel file\n");
  fprintf(stderr, "\t <grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t <sim_time>   - number of iterations\n");
  fprintf(stderr, "\t <temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t <power_file> - name of the file containing the dissipated power values of each cell\n");
  fprintf(stderr, "\t <output_file> - name of the output file\n");
  exit(1);
}

void readinput(std::vector<float> & vect, int grid_rows, int grid_cols, char *file) {

  int i,j;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if( (fp  = fopen(file, "r" )) ==0 )
  {
    fprintf(stderr, "The file %s was not opened\n", file);
    exit(1);
  }

  for (i=0; i <= grid_rows-1; i++) 
    for (j=0; j <= grid_cols-1; j++)
    {
      if (fgets(str, STR_SIZE, fp) == NULL) 
      {
        fprintf(stderr, "Error reading file %s\n", file);
        exit(1);
      }
      if (feof(fp))
      {
        fprintf(stderr, "Not enough lines in file %s\n", file);
        exit(1);
      }
      //if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
      if ((sscanf(str, "%f", &val) != 1))
      {
        fprintf(stderr, "Invalid file format in file %s\n", file);
        exit(1);
      }
      vect[i*grid_cols+j] = val;
    }

  fclose(fp);	

}

void writeoutput(std::vector<float> & vect, int grid_rows, int grid_cols, char *file) {

  int i,j, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 )
    printf( "The file %s was not opened\n", file);


  for (i=0; i < grid_rows; i++) 
    for (j=0; j < grid_cols; j++)
    {

      sprintf(str, "%d\t%.4f\n", index, vect[i*grid_cols+j]);
      fputs(str,fp);
      index++;
    }

  fclose(fp);	
}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {

  int i,j, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 )
    printf( "The file %s was not opened\n", file);


  for (i=0; i < grid_rows; i++) 
    for (j=0; j < grid_cols; j++)
    {

      sprintf(str, "%d\t%.4f\n", index, vect[i*grid_cols+j]);
      fputs(str,fp);
      index++;
    }

  fclose(fp);	
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  int i,j;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if( (fp  = fopen(file, "r" )) ==0 )
  {
    fprintf(stderr, "The file %s was not opened\n", file);
    exit(1);
  }


  for (i=0; i <= grid_rows-1; i++) 
    for (j=0; j <= grid_cols-1; j++)
    {
      if (fgets(str, STR_SIZE, fp) == NULL) {fprintf(stderr, "Error reading file %s\n", file); exit(1);}
      if (feof(fp))
      {
        fprintf(stderr, "Not enough lines in file %s\n", file);
        exit(1);
      }
      //if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
      if ((sscanf(str, "%f", &val) != 1))
      {
        fprintf(stderr, "Invalid file format in file %s\n", file);
        exit(1);
      }
      vect[i*grid_cols+j] = val;
    }

  fclose(fp);	

}
