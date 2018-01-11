#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <stdlib.h>
#include <limits.h>

#include "tuner_api.h"
#include "sort.h"
#include "sort_reference.h"
#include "sort_tunable.h"

#define RAND_MAX UINT_MAX

int main(int argc, char** argv)
{
  //Initialize platform and device index
  size_t platformIndex = 0;
  size_t deviceIndex = 0;
  auto kernelFile = std::string("../../examples/shoc-sort/sort.cl");
  auto referenceKernelFile = std::string("../../examples/shoc-sort/reference_sort.cl");

  if (argc >= 2)
  {
    platformIndex = std::stoul(std::string{argv[1]});
    if (argc >= 3)
    {
      deviceIndex = std::stoul(std::string{argv[2]});
      if (argc >= 4)
      {
        kernelFile = std::string{argv[3]};
      }
    }
  }

  // Declare data variables necessary for command line arguments parsing
  int problemSize = 1; //in MiB

  if (argc >= 5)
  {
    problemSize = atoi(argv[4]);
  }
  
  int size = problemSize * 1024 * 1024;

  // Create input and output vectors and initialize with random numbers
  std::vector<unsigned int> in = std::vector<unsigned int>(size);
  std::vector<unsigned int> out = std::vector<unsigned int>(size);

  srand((unsigned int)time(NULL));
  for (int i = 0; i < size; i++)
  {
    in[i] = rand();
    out[i] = -1;
  }

  // Create tuner object for chosen platform and device
  ktt::Tuner tuner(platformIndex, deviceIndex);

  //Declare kernels
  size_t kernelId; //id for composite kernel
  std::vector<size_t> kernelIds = std::vector<size_t>(3); //ids for specific kernels
  size_t referenceKernelId;
  std::vector<size_t> referenceKernelIds = std::vector<size_t>(3); //ids for reference insider kernels
  // we will tune also global and local size so just initialize here
  const ktt::DimensionVector ndRangeDimensions(1, 1, 1);
  const ktt::DimensionVector workGroupDimensions(1, 1, 1);

  kernelIds[0] = tuner.addKernelFromFile(kernelFile, std::string("reduce"), ndRangeDimensions, workGroupDimensions);
  kernelIds[1] = tuner.addKernelFromFile(kernelFile, std::string("top_scan"), workGroupDimensions, workGroupDimensions);
  kernelIds[2] = tuner.addKernelFromFile(kernelFile, std::string("bottom_scan"), ndRangeDimensions, workGroupDimensions);

  referenceKernelIds[0] = tuner.addKernelFromFile(referenceKernelFile, std::string("reduce"), ndRangeDimensions, workGroupDimensions);
  referenceKernelIds[1] = tuner.addKernelFromFile(referenceKernelFile, std::string("top_scan"), workGroupDimensions, workGroupDimensions);
  referenceKernelIds[2] = tuner.addKernelFromFile(referenceKernelFile, std::string("bottom_scan"), ndRangeDimensions, workGroupDimensions);

  //Add arguments for kernels
  size_t inId = tuner.addArgumentVector(std::vector<unsigned int>(in), ktt::ArgumentAccessType::ReadWrite);
  size_t outId = tuner.addArgumentVector(std::vector<unsigned int>(out), ktt::ArgumentAccessType::ReadWrite);
  size_t isumsId = tuner.addArgumentVector(std::vector<unsigned int>(0), ktt::ArgumentAccessType::ReadWrite); //vector, readwrite, must be added after global and local size are determined, as its size depends on the number of groups
  size_t sizeId = tuner.addArgumentScalar(size);
  size_t workGroupSizeId = tuner.addArgumentScalar(workGroupDimensions.getSizeX());

  size_t localMem1Id = tuner.addArgumentLocal(8); //local, workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  size_t localMem2Id = tuner.addArgumentLocal(8); //local, 2*workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  int shift = 0;
  size_t shiftId = tuner.addArgumentScalar(shift); //will be updated as the kernel execution is iterative

  tuner.setKernelArguments(kernelIds[0], std::vector<size_t>{inId, isumsId, sizeId, localMem1Id, shiftId});
  tuner.setKernelArguments(kernelIds[1], std::vector<size_t>{isumsId, workGroupSizeId, localMem2Id});
  tuner.setKernelArguments(kernelIds[2], std::vector<size_t>{inId, isumsId, outId, sizeId, localMem2Id, shiftId});
  tunableSort * sort = new tunableSort(&tuner, kernelIds, size, inId, outId, isumsId, localMem1Id, localMem2Id, workGroupSizeId, shiftId);
  kernelId = tuner.addComposition("sort", kernelIds, std::unique_ptr<tunableSort>(sort));
  sort->setKernelId(kernelId);
  tuner.setTuningManipulator(kernelId, std::unique_ptr<tunableSort>(sort));

  tuner.setKernelArguments(referenceKernelIds[0], std::vector<size_t>{inId, isumsId, sizeId, localMem1Id, shiftId});
  tuner.setKernelArguments(referenceKernelIds[1], std::vector<size_t>{isumsId, workGroupSizeId, localMem2Id});
  tuner.setKernelArguments(referenceKernelIds[2], std::vector<size_t>{inId, isumsId, outId, sizeId, localMem2Id, shiftId});
  referenceSort* refSort = new referenceSort(&tuner, referenceKernelIds, size, inId, outId, isumsId, localMem1Id, localMem2Id, workGroupSizeId, shiftId);
  referenceKernelId = tuner.addComposition("ref", referenceKernelIds, std::unique_ptr<referenceSort>(refSort));
  tuner.setTuningManipulator(referenceKernelId, std::unique_ptr<referenceSort>(refSort));

  tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.9f);

  tuner.setReferenceKernel(kernelId, referenceKernelId, {}, std::vector<size_t>{outId});

  sort->tune();
  return 0;
}
