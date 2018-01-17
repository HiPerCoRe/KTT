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
  std::vector<unsigned int> in = std::vector<unsigned int>(size);

  srand((unsigned int)time(NULL));
  for (int i = 0; i < size; i++)
  {
    in[i] = rand();
  }

  // Create tuner object for chosen platform and device
  ktt::Tuner tuner(platformIndex, deviceIndex);

  //Declare kernels
  size_t kernelId = -1; //id for composite kernel
  std::vector<size_t> kernelIds = std::vector<size_t>(3); //ids for specific kernels
  // we will tune also global and local size so just initialize here
  const ktt::DimensionVector ndRangeDimensions(1, 1, 1);
  const ktt::DimensionVector workGroupDimensions(1, 1, 1);

  kernelIds[0] = tuner.addKernelFromFile(kernelFile, std::string("reduce"), ndRangeDimensions, workGroupDimensions);
  kernelIds[1] = tuner.addKernelFromFile(kernelFile, std::string("top_scan"), workGroupDimensions, workGroupDimensions);
  kernelIds[2] = tuner.addKernelFromFile(kernelFile, std::string("bottom_scan"), ndRangeDimensions, workGroupDimensions);

  printf("ids kernel %u %u %u\n", kernelIds[0], kernelIds[1], kernelIds[2]);

  //Add arguments for kernels
  size_t inId = tuner.addArgumentVector(std::vector<unsigned int>(in), ktt::ArgumentAccessType::ReadWrite);
  size_t outId = tuner.addArgumentVector(std::vector<unsigned int>(size), ktt::ArgumentAccessType::ReadWrite);
  int isumsSize = 64*16;
  size_t isumsId = tuner.addArgumentVector(std::vector<unsigned int>(isumsSize), ktt::ArgumentAccessType::ReadWrite); //vector, readwrite, must be added after global and local size are determined, as its size depends on the number of groups
  size_t sizeId = tuner.addArgumentScalar(size);
  size_t workGroupSizeId = tuner.addArgumentScalar(256);

  size_t localMem1Id = tuner.addArgumentLocal<unsigned int>(256); //local, workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  size_t localMem2Id = tuner.addArgumentLocal<unsigned int>(512); //local, 2*workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  size_t localMem3Id = tuner.addArgumentLocal<unsigned int>(512); //local, 2*workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  int shift = 0;
  size_t shiftId = tuner.addArgumentScalar(shift); //will be updated as the kernel execution is iterative

  tunableSort * sort = new tunableSort(&tuner, kernelIds, size, inId, outId, isumsId, sizeId, localMem1Id, localMem2Id, workGroupSizeId, shiftId);
  kernelId = tuner.addComposition("sort", kernelIds, std::unique_ptr<tunableSort>(sort));
  sort->setKernelId(kernelId);
  tuner.setCompositionKernelArguments(kernelId, kernelIds[0], std::vector<size_t>{inId, isumsId, sizeId, localMem1Id, shiftId});
  tuner.setCompositionKernelArguments(kernelId, kernelIds[1], std::vector<size_t>{isumsId, workGroupSizeId, localMem2Id});
  tuner.setCompositionKernelArguments(kernelId, kernelIds[2], std::vector<size_t>{inId, isumsId, outId, sizeId, localMem3Id, shiftId});
  printf("kernelId %d\n", kernelId);

  tuner.addCompositionKernelParameter(kernelId, kernelIds[0], "BLOCK_SIZE_ROWS", {8, 16, 32, 64}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add, ktt::Dimension(0));
  tuner.addCompositionKernelParameter(kernelId, kernelIds[1], "BLOCK_SIZE_ROWS", {8, 16, 32, 64}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add, ktt::Dimension(0));
  tuner.addCompositionKernelParameter(kernelId, kernelIds[2], "BLOCK_SIZE_ROWS", {8, 16, 32, 64}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add, ktt::Dimension(0));

  tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.9f);

  tuner.setReferenceClass(kernelId, std::make_unique<referenceSort>(&tuner, &in), std::vector<ktt::ArgumentId>{outId});

  setbuf(stdout, NULL);
  sort->tune();
  return 0;
}
