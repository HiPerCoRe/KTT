#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tuner_api.h"
#include "hotspot.h"
#include "hotspot_tunable.h"

int main(int argc, char** argv)
{
  // Initialize platform and device index
  size_t platformIndex = 0;
  size_t deviceIndex = 0;
  auto kernelFile = std::string("../../examples/rodinia-hotspot/hotspot_kernel.cl");

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

  int total_iterations = 60;

  if (argc >=4 && argc < 9)
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

  // Create tuner object for chosen platform and device
  ktt::Tuner tuner(platformIndex, deviceIndex);
  tuner.setCompilerOptions("-I./");

  tunableHotspot* hotspot = new tunableHotspot(&tuner, kernelFile, grid_rows, grid_cols, total_iterations, tfile, pfile, ofile);
  tuner.setTuningManipulator(hotspot->getKernelId(), std::unique_ptr<tunableHotspot>(hotspot));
  hotspot->tune();
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
