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

  // Create input and output vectors and initialize with pseudorandom numbers
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
  const ktt::DimensionVector ndRangeDimensions(0, 1, 1);
  const ktt::DimensionVector workGroupDimensions(0, 1, 1);

  kernelIds[0] = tuner.addKernelFromFile(kernelFile, std::string("reduce"), ndRangeDimensions, workGroupDimensions);
  kernelIds[1] = tuner.addKernelFromFile(kernelFile, std::string("top_scan"), workGroupDimensions, workGroupDimensions);
  kernelIds[2] = tuner.addKernelFromFile(kernelFile, std::string("bottom_scan"), ndRangeDimensions, workGroupDimensions);


  //Add arguments for kernels
  size_t inId = tuner.addArgumentVector(std::vector<unsigned int>(in), ktt::ArgumentAccessType::ReadWrite);
  size_t outId = tuner.addArgumentVector(std::vector<unsigned int>(size), ktt::ArgumentAccessType::ReadWrite);
  int numberOfGroups = 1;
  int isumsSize = numberOfGroups*16*sizeof(unsigned int);
  size_t isumsId = tuner.addArgumentVector(std::vector<unsigned int>(isumsSize), ktt::ArgumentAccessType::ReadWrite); //vector, readwrite, must be added after global and local size are determined, as its size depends on the number of groups
  size_t sizeId = tuner.addArgumentScalar(size);
  size_t numberOfGroupsId = tuner.addArgumentScalar(1);

  int localSize = 1;
  size_t localMem1Id = tuner.addArgumentLocal<unsigned int>(localSize); //local, workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  size_t localMem2Id = tuner.addArgumentLocal<unsigned int>(2*localSize); //local, 2*workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  size_t localMem3Id = tuner.addArgumentLocal<unsigned int>(2*localSize); //local, 2*workgroupsize*sizeof(unsigned int), must be added after local size is determined, as its size depends on that
  int shift = 0;
  size_t shiftId = tuner.addArgumentScalar(shift); //will be updated as the kernel execution is iterative

  tunableSort * sort = new tunableSort(&tuner, kernelIds, size, inId, outId, isumsId, sizeId, localMem1Id, localMem2Id, localMem3Id, numberOfGroupsId, shiftId);
  kernelId = tuner.addComposition("sort", kernelIds, std::unique_ptr<tunableSort>(sort));
  sort->setKernelId(kernelId);
  tuner.setCompositionKernelArguments(kernelId, kernelIds[0], std::vector<size_t>{inId, isumsId, sizeId, localMem1Id, shiftId});
  tuner.setCompositionKernelArguments(kernelId, kernelIds[1], std::vector<size_t>{isumsId, numberOfGroupsId, localMem2Id});
  tuner.setCompositionKernelArguments(kernelId, kernelIds[2], std::vector<size_t>{inId, isumsId, outId, sizeId, localMem3Id, shiftId});

  tuner.addCompositionKernelParameter(kernelId, kernelIds[0], "LOCAL_SIZE", {128, 256, 512}, ktt::ThreadModifierType::Local, ktt::ThreadModifierAction::Add, ktt::Dimension(0));
  //local size below 128, i.e. 64 or 32, does not work correctly, not even with the benchmark code
  tuner.addCompositionKernelParameter(kernelId, kernelIds[0], "GLOBAL_SIZE", {512, 1024, 2048, 4096, 8192, 16384, 32768}, ktt::ThreadModifierType::Global, ktt::ThreadModifierAction::Add, ktt::Dimension(0));
  //auto workGroupSmaller = [](std::vector<size_t> vector) {return vector.at(0)<=vector.at(1);};
  //tuner.addConstraint(kernelId, workGroupSmaller, {"LOCAL_SIZE", "GLOBAL_SIZE"});
  tuner.addCompositionKernelParameter(kernelId, kernelIds[1], "LOCAL_SIZE", {128, 256, 512}, ktt::ThreadModifierType::Local, ktt::ThreadModifierAction::Add, ktt::Dimension(0));
  tuner.addCompositionKernelParameter(kernelId, kernelIds[2], "LOCAL_SIZE", {128, 256, 512}, ktt::ThreadModifierType::Local, ktt::ThreadModifierAction::Add, ktt::Dimension(0));
  tuner.addCompositionKernelParameter(kernelId, kernelIds[2], "GLOBAL_SIZE", {512, 1024, 2048, 4096, 8192, 16384, 32768}, ktt::ThreadModifierType::Global, ktt::ThreadModifierAction::Add, ktt::Dimension(0));

  //parameter for the length of OpenCl vector data types used in the kernels
  tuner.addCompositionKernelParameter(kernelId, kernelIds[0], "FPVECTNUM", {4, 8, 16}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add, ktt::Dimension(0));
  tuner.addCompositionKernelParameter(kernelId, kernelIds[2], "FPVECTNUM", {4, 8, 16}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add, ktt::Dimension(0));

  tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.9f);

  tuner.setReferenceClass(kernelId, std::make_unique<referenceSort>(&tuner, &in), std::vector<ktt::ArgumentId>{outId});

  sort->tune();
  return 0;
}
