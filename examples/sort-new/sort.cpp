#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <stdlib.h>
#include <limits.h>

#include "tuner_api.h"
#include "sort_reference.h"
#include "sort_tunable.h"

#ifndef RAND_MAX
#define RAND_MAX UINT_MAX
#endif

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/sort-new/sort_kernel.cu"
#else
    #define KTT_KERNEL_FILE "../../examples/sort-new/sort_kernel.cu"
#endif

int main(int argc, char** argv)
{
  // Initialize platform and device index
  size_t platformIndex = 0;
  size_t deviceIndex = 0;
  std::string kernelFile = KTT_KERNEL_FILE;
  int problemSize = 1; // In MiB

  if (argc >= 2)
  {
    platformIndex = std::stoul(std::string{argv[1]});
    if (argc >= 3)
    {
      deviceIndex = std::stoul(std::string{argv[2]});
      if (argc >= 4)
      {
        problemSize = atoi(argv[3]);
      }
    }
  }


  if (argc >= 5)
  {
    kernelFile = std::string{argv[4]};
  }
  
  int size = problemSize * 1024 * 1024 / sizeof(unsigned int);

  // Create input and output vectors and initialize with pseudorandom numbers
  std::vector<unsigned int> keysIn(size);
  std::vector<unsigned int> keysOut(size);
  std::vector<unsigned int> valuesIn(size);
  std::vector<unsigned int> valuesOut(size);
  uint bytes = size * sizeof(uint);

  //srand((unsigned int)time(NULL));
  srand(123);
  for (int i = 0; i < size; i++)
  {
    valuesIn[i] = keysIn[i] = i%1024;//rand();
  }

  // Create tuner object for chosen platform and device
  ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);
  tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);
  ktt::DeviceInfo device = tuner.getCurrentDeviceInfo();
  tuner.setCompilerOptions("-G");
  tuner.setLoggingLevel(ktt::LoggingLevel::Debug); 

  // Declare kernels and their dimensions
  std::vector<ktt::KernelId> kernelIds(5);
  const ktt::DimensionVector ndRangeDimensions;
  const ktt::DimensionVector ndRangeDimensions4(size/4, 1, 1);
  const ktt::DimensionVector ndRangeDimensions2(size/2, 1, 1);
  const ktt::DimensionVector workGroupDimensions;

  kernelIds[0] = tuner.addKernelFromFile(kernelFile, std::string("radixSortBlocks"), ndRangeDimensions4, workGroupDimensions);
  kernelIds[1] = tuner.addKernelFromFile(kernelFile, std::string("findRadixOffsets"), ndRangeDimensions2, workGroupDimensions);
  kernelIds[2] = tuner.addKernelFromFile(kernelFile, std::string("reorderData"), ndRangeDimensions2, workGroupDimensions);
  kernelIds[3] = tuner.addKernelFromFile(kernelFile, std::string("vectorAddUniform4"), ndRangeDimensions, workGroupDimensions);
  kernelIds[4] = tuner.addKernelFromFile(kernelFile, std::string("scan"), ndRangeDimensions, workGroupDimensions);

  // Add arguments for kernels
  ktt::ArgumentId nbitsId = tuner.addArgumentScalar(nbits);
  ktt::ArgumentId startBitId = tuner.addArgumentScalar(0);
  
  ktt::ArgumentId keysOutId = tuner.addArgumentVector(keysOut, ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId valuesOutId = tuner.addArgumentVector(valuesOut, ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId keysInId = tuner.addArgumentVector(keysIn, ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId valuesInId = tuner.addArgumentVector(valuesIn, ktt::ArgumentAccessType::ReadWrite);

  ktt::ArgumentId countersId = tuner.addArgumentVector(std::vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId counterSumsId = tuner.addArgumentVector(std::vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId blockOffsetsId = tuner.addArgumentVector(std::vector<unsigned int>(1), ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId scanBlocksSumId = tuner.addArgumentVector(std::vector<unsigned int*>(1), ktt::ArgumentAccessType::ReadWrite);
  
  ktt::ArgumentId sizeId = tuner.addArgumentScalar(size);
  ktt::ArgumentId sortBlockSizeId = tuner.addArgumentScalar(1);
  ktt::ArgumentId scanBlockSizeId = tuner.addArgumentScalar(1);
  ktt::ArgumentId sortNumBlocksId = tuner.addArgumentScalar(1); //will be updated
  ktt::ArgumentId scanNumBlocksId = tuner.addArgumentScalar(1); //will be updated
  ktt::ArgumentId numElementsId = tuner.addArgumentScalar(1);

  ktt::ArgumentId scanOutDataId = tuner.addArgumentVector(std::vector<uint>(1), ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId scanInDataId = tuner.addArgumentVector(std::vector<uint>(1), ktt::ArgumentAccessType::ReadOnly);
  ktt::ArgumentId scanOneBlockSumId = tuner.addArgumentVector(std::vector<uint>(1), ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId fullBlockId = tuner.addArgumentScalar(0); //will be updated
  ktt::ArgumentId storeSumId = tuner.addArgumentScalar(0); //will be updated
  ktt::ArgumentId vectorOutDataId = tuner.addArgumentVector(std::vector<uint>(1), ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId vectorBlockSumId = tuner.addArgumentVector(std::vector<uint>(1), ktt::ArgumentAccessType::ReadOnly);
  ktt::ArgumentId vectorNumElementsId = tuner.addArgumentScalar(1);

  
  ktt::KernelId compositionId = tuner.addComposition("sort", kernelIds, std::make_unique<TunableSort>(kernelIds, device, size, keysOutId, valuesOutId, keysInId, valuesInId, keysIn, valuesIn, sortNumBlocksId, sortBlockSizeId, scanNumBlocksId, scanBlockSizeId, countersId, counterSumsId, blockOffsetsId, scanBlocksSumId, startBitId, scanOutDataId, scanInDataId, scanOneBlockSumId, numElementsId, fullBlockId, storeSumId, vectorOutDataId, vectorBlockSumId, vectorNumElementsId));
  tuner.setCompositionKernelArguments(compositionId, kernelIds[0], std::vector<size_t>{nbitsId, startBitId, keysOutId, valuesOutId, keysInId, valuesInId});
  tuner.setCompositionKernelArguments(compositionId, kernelIds[1], std::vector<size_t>{keysOutId, countersId, blockOffsetsId, startBitId, sizeId, scanNumBlocksId});
  tuner.setCompositionKernelArguments(compositionId, kernelIds[2], std::vector<size_t>{startBitId, keysOutId, valuesOutId, keysInId, valuesInId, blockOffsetsId, counterSumsId, countersId, scanNumBlocksId});
  tuner.setCompositionKernelArguments(compositionId, kernelIds[3], std::vector<size_t>{scanOutDataId, scanOneBlockSumId, numElementsId});
  tuner.setCompositionKernelArguments(compositionId, kernelIds[4], std::vector<size_t>{scanOutDataId, scanInDataId, scanOneBlockSumId, numElementsId, fullBlockId, storeSumId});

  // Parameter for the length of OpenCL vector data types used in the kernels
  //tuner.addParameter(compositionId, "FPVECTNUM", {4, 8, 16});
  // Local size below 128 does not work correctly, not even with the benchmark code
  tuner.addParameter(compositionId, "SORT_BLOCK_SIZE", {32, 64, 128, 256, 512, 1024});
  tuner.addParameter(compositionId, "SCAN_BLOCK_SIZE", {32, 64, 128, 256, 512, 1024});
  auto workGroupConstraint = [](const std::vector<size_t>& vector) {return vector.at(1) <= vector.at(0)*2;};
  tuner.addConstraint(compositionId, workGroupConstraint, {"SORT_BLOCK_SIZE", "SCAN_BLOCK_SIZE"});

  tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.9);
  tuner.setReferenceClass(compositionId, std::make_unique<ReferenceSort>(valuesIn), std::vector<ktt::ArgumentId>{valuesOutId});

  tuner.tuneKernel(compositionId);
  tuner.printResult(compositionId, std::cout, ktt::PrintFormat::Verbose);
  tuner.printResult(compositionId, std::string("sort_result.csv"), ktt::PrintFormat::CSV);
  return 0;
}
