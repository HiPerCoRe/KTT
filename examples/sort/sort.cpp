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

#define USE_CUDA 0

#if USE_CUDA == 0
    #if defined(_MSC_VER)
        #define KTT_KERNEL_FILE "../examples/sort/sort.cl"
    #else
        #define KTT_KERNEL_FILE "../../examples/sort/sort.cl"
    #endif
#else
    #if defined(_MSC_VER)
        #define KTT_KERNEL_FILE "../examples/sort/sort.cu"
    #else
        #define KTT_KERNEL_FILE "../../examples/sort/sort.cu"
    #endif
#endif

int main(int argc, char** argv)
{
  // Initialize platform and device index
  ktt::PlatformIndex platformIndex = 0;
  ktt::DeviceIndex deviceIndex = 0;
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

  int problemSize = 32; // In MiB

  if (argc >= 5)
  {
    problemSize = atoi(argv[4]);
  }
  
  int size = problemSize * 1024 * 1024 / sizeof(unsigned int);

  // Create input and output vectors and initialize with pseudorandom numbers
  std::vector<unsigned int> in(size);

  srand((unsigned int)time(NULL));
  for (int i = 0; i < size; i++)
  {
    in[i] = rand();
  }

  // Create tuner object for chosen platform and device
#if USE_CUDA == 0
  ktt::Tuner tuner(platformIndex, deviceIndex);
#else
    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);
    tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);
#endif
  tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

  // Declare kernels and their dimensions
  std::vector<ktt::KernelId> kernelIds(3);
  const ktt::DimensionVector ndRangeDimensions;
  const ktt::DimensionVector workGroupDimensions;

  kernelIds[0] = tuner.addKernelFromFile(kernelFile, std::string("reduce"), ndRangeDimensions, workGroupDimensions);
  kernelIds[1] = tuner.addKernelFromFile(kernelFile, std::string("top_scan"), workGroupDimensions, workGroupDimensions);
  kernelIds[2] = tuner.addKernelFromFile(kernelFile, std::string("bottom_scan"), ndRangeDimensions, workGroupDimensions);

  // Add arguments for kernels
  ktt::ArgumentId inId = tuner.addArgumentVector(in, ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId outId = tuner.addArgumentVector(std::vector<unsigned int>(size), ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId sizeId = tuner.addArgumentScalar(size);
  ktt::ArgumentId numberOfGroupsId = tuner.addArgumentScalar(1);
  int shift = 0;
  ktt::ArgumentId shiftId = tuner.addArgumentScalar(shift); // Will be updated as the kernel execution is iterative
  
  int numberOfGroups = 1;
  int isumsSize = 16 * numberOfGroups;
  // Vector argument will be updated in tuning manipulator as its size depends on the number of work-groups
  ktt::ArgumentId isumsId = tuner.addArgumentVector(std::vector<unsigned int>(isumsSize), ktt::ArgumentAccessType::ReadWrite);

  // Local memory arguments will be updated in tuning manipulator as their size depends on work-group size
  int localSize = 1;  
  
  ktt::KernelId compositionId = tuner.addComposition("sort", kernelIds, std::make_unique<TunableSort>(kernelIds, size, inId, outId, isumsId, sizeId,
      numberOfGroupsId, shiftId));
  tuner.setCompositionKernelArguments(compositionId, kernelIds[0], std::vector<ktt::ArgumentId>{inId, isumsId, sizeId, shiftId});
  tuner.setCompositionKernelArguments(compositionId, kernelIds[1], std::vector<ktt::ArgumentId>{isumsId, numberOfGroupsId});
  tuner.setCompositionKernelArguments(compositionId, kernelIds[2], std::vector<ktt::ArgumentId>{inId, isumsId, outId, sizeId, shiftId});

  // Parameter for the length of OpenCL vector data types used in the kernels
#if USE_CUDA == 0
  tuner.addParameter(compositionId, "FPVECTNUM", {4, 8, 16});
#else
  tuner.addParameter(compositionId, "FPVECTNUM", {4});
#endif

  // Local size below 128 does not work correctly, not even with the benchmark code
  tuner.addParameter(compositionId, "LOCAL_SIZE", {128, 256, 512});
  tuner.setThreadModifier(compositionId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "LOCAL_SIZE", ktt::ModifierAction::Multiply);

  // Second kernel global size is always equal to local size
  tuner.addParameter(compositionId, "GLOBAL_SIZE", {512, 1024, 2048, 4096, 8192, 16384, 32768});
  tuner.setCompositionKernelThreadModifier(compositionId, kernelIds[0], ktt::ModifierType::Global, ktt::ModifierDimension::X, "GLOBAL_SIZE",
      ktt::ModifierAction::Multiply);
  tuner.setCompositionKernelThreadModifier(compositionId, kernelIds[1], ktt::ModifierType::Global, ktt::ModifierDimension::X, "LOCAL_SIZE",
      ktt::ModifierAction::Multiply);
  tuner.setCompositionKernelThreadModifier(compositionId, kernelIds[2], ktt::ModifierType::Global, ktt::ModifierDimension::X, "GLOBAL_SIZE",
      ktt::ModifierAction::Multiply);

  auto workGroupConstraint = [](const std::vector<size_t>& vector) {return vector.at(0) != 128 || vector.at(1) != 32768;};
  tuner.addConstraint(compositionId, {"LOCAL_SIZE", "GLOBAL_SIZE"}, workGroupConstraint);

  tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.9);
  tuner.setReferenceClass(compositionId, std::make_unique<ReferenceSort>(in), std::vector<ktt::ArgumentId>{outId});

  tuner.tuneKernel(compositionId);
  tuner.printResult(compositionId, std::cout, ktt::PrintFormat::Verbose);
  tuner.printResult(compositionId, std::string("sort_result.csv"), ktt::PrintFormat::CSV);

  /*std::vector<unsigned int> firstMatrix(10);
  ktt::OutputDescriptor output(outId, (void*)firstMatrix.data(), 10*sizeof(unsigned int));
   ktt::ComputationResult bestConf = tuner.getBestComputationResult(compositionId);
   tuner.runKernel(compositionId, bestConf.getConfiguration(), {output});*/

  return 0;
}

