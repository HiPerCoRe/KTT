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

#ifndef RAND_MAX
#define RAND_MAX UINT_MAX
#endif

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/shoc-sort/sort.cl"
#else
    #define KTT_KERNEL_FILE "../../examples/shoc-sort/sort.cl"
#endif

int main(int argc, char** argv)
{
  //Initialize platform and device index
  size_t platformIndex = 0;
  size_t deviceIndex = 0;
  std::string kernelFile = KTT_KERNEL_FILE;

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
  
  int size = problemSize * 256 * 256;

  // Create input and output vectors and initialize with random numbers
  std::vector<unsigned int> in(size);

  srand((unsigned int)time(NULL));
  for (int i = 0; i < size; i++)
  {
    in[i] = rand();
  }

  // Create tuner object for chosen platform and device
  ktt::Tuner tuner(platformIndex, deviceIndex);

  //Declare kernels
  ktt::KernelId compositionId = -1; //id for composite kernel
  std::vector<ktt::KernelId> kernelIds(3);//ids for specific kernels
  // we will tune also global and local size so just initialize here
  const ktt::DimensionVector ndRangeDimensions;
  const ktt::DimensionVector workGroupDimensions;

  kernelIds[0] = tuner.addKernelFromFile(kernelFile, std::string("reduce"), ndRangeDimensions, workGroupDimensions);
  kernelIds[1] = tuner.addKernelFromFile(kernelFile, std::string("top_scan"), workGroupDimensions, workGroupDimensions);
  kernelIds[2] = tuner.addKernelFromFile(kernelFile, std::string("bottom_scan"), ndRangeDimensions, workGroupDimensions);

  //Add arguments for kernels
  ktt::ArgumentId inId = tuner.addArgumentVector(in, ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId outId = tuner.addArgumentVector(std::vector<unsigned int>(size), ktt::ArgumentAccessType::ReadWrite);
  int isumsSize = 64*16;
  ktt::ArgumentId isumsId = tuner.addArgumentVector(std::vector<unsigned int>(isumsSize), ktt::ArgumentAccessType::ReadWrite); //vector, readwrite, must be added after global and local size are determined, as its size depends on the number of groups
  ktt::ArgumentId sizeId = tuner.addArgumentScalar(size);
  ktt::ArgumentId workGroupSizeId = tuner.addArgumentScalar(256);

  ktt::ArgumentId localMem1Id = tuner.addArgumentLocal<unsigned int>(256); //local, workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  ktt::ArgumentId localMem2Id = tuner.addArgumentLocal<unsigned int>(512); //local, 2*workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  ktt::ArgumentId localMem3Id = tuner.addArgumentLocal<unsigned int>(512); //local, 2*workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  int shift = 0;
  ktt::ArgumentId shiftId = tuner.addArgumentScalar(shift); //will be updated as the kernel execution is iterative

  tunableSort * sort = new tunableSort(&tuner, kernelIds, size, inId, outId, isumsId, sizeId, localMem1Id, localMem2Id, workGroupSizeId, shiftId);
  compositionId = tuner.addComposition("sort", kernelIds, std::unique_ptr<tunableSort>(sort));
  sort->setKernelId(compositionId);
  tuner.setCompositionKernelArguments(compositionId, kernelIds[0], std::vector<size_t>{inId, isumsId, sizeId, localMem1Id, shiftId});
  tuner.setCompositionKernelArguments(compositionId, kernelIds[1], std::vector<size_t>{isumsId, workGroupSizeId, localMem2Id});
  tuner.setCompositionKernelArguments(compositionId, kernelIds[2], std::vector<size_t>{inId, isumsId, outId, sizeId, localMem3Id, shiftId});

  tuner.addParameter(compositionId, "BLOCK_SIZE_ROWS", { 8, 16, 32, 64 });

  tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.9f);

  tuner.setReferenceClass(compositionId, std::make_unique<referenceSort>(in), std::vector<ktt::ArgumentId>{outId});

  sort->tune();
  return 0;
}
