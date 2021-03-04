#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

#ifndef RAND_MAX
#define RAND_MAX UINT_MAX
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Sort/Sort.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Sort/Sort.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = defaultKernelFile;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));

        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));

            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);
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

    for (int i = 0; i < size; ++i)
    {
        in[i] = rand();
    }

    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // Declare kernels and their dimensions
    std::vector<ktt::KernelDefinitionId> definitionIds(3);
    const ktt::DimensionVector ndRangeDimensions;
    const ktt::DimensionVector workGroupDimensions;

    definitionIds[0] = tuner.AddKernelDefinitionFromFile("reduce", kernelFile, ndRangeDimensions, workGroupDimensions);
    definitionIds[1] = tuner.AddKernelDefinitionFromFile("top_scan", kernelFile, workGroupDimensions, workGroupDimensions);
    definitionIds[2] = tuner.AddKernelDefinitionFromFile("bottom_scan", kernelFile, ndRangeDimensions, workGroupDimensions);

    // Add arguments for kernels
    const ktt::ArgumentId inId = tuner.AddArgumentVector(in, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId outId = tuner.AddArgumentVector(std::vector<unsigned int>(size), ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId sizeId = tuner.AddArgumentScalar(size);
    const ktt::ArgumentId numberOfGroupsId = tuner.AddArgumentScalar(1);
    int shift = 0;
    const ktt::ArgumentId shiftId = tuner.AddArgumentScalar(shift); // Will be updated as the kernel execution is iterative
  
    int numberOfGroups = 1;
    int isumsSize = 16 * numberOfGroups;
    // Vector argument will be updated in tuning manipulator as its size depends on the number of work-groups
    const ktt::ArgumentId isumsId = tuner.AddArgumentVector(std::vector<unsigned int>(isumsSize), ktt::ArgumentAccessType::ReadWrite);

    const ktt::KernelId kernel = tuner.CreateCompositeKernel("Sort", definitionIds,
        [&definitionIds, size, numberOfGroupsId, isumsId, shiftId, inId, outId, sizeId](ktt::ComputeInterface& interface)
    {
        const int radix_width = 4;
        const std::vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();
        uint64_t localSize = ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "LOCAL_SIZE");
        uint64_t globalSize = ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "GLOBAL_SIZE");

        int numberOfGroups = static_cast<int>(globalSize / localSize);
        interface.UpdateScalarArgument(numberOfGroupsId, &numberOfGroups);
        int isumsSize = 16 * numberOfGroups;

        // Vector, read-write, must be added after global and local size are determined, as its size depends on the number of groups
        interface.ResizeBuffer(isumsId, isumsSize * sizeof(unsigned int), false);

        bool inOutSwapped = false;

        for (int shift = 0; shift < sizeof(unsigned int) * 8; shift += radix_width)
        {
            // Like scan, we use a reduce-then-scan approach

            // But before proceeding, update the shift appropriately for each kernel. This is how many bits to shift to the right used in binning.
            interface.UpdateScalarArgument(shiftId, &shift);

            // Also, the sort is not in place, so swap the input and output buffers on each pass.
            const bool even = ((shift / radix_width) % 2 == 0) ? true : false;

            if (even)
            {
                interface.ChangeArguments(definitionIds[0], {inId, isumsId, sizeId, shiftId});
            }
            else
            {
                interface.ChangeArguments(definitionIds[0], {outId, isumsId, sizeId, shiftId});
            }

            // Each thread block gets an equal portion of the input array, and computes occurrences of each digit.
            interface.RunKernel(definitionIds[0]);

            // Next, a top-level exclusive scan is performed on the per block histograms. This is done by a single work group
            // (note global size here is the same as local).
            interface.RunKernel(definitionIds[1]);

            // Finally, a bottom-level scan is performed by each block that is seeded with the scanned histograms which rebins,
            // locally scans, then scatters keys to global memory
            interface.RunKernel(definitionIds[2]);

            // Also, the sort is not in place, so swap the input and output buffers on each pass.
            interface.SwapArguments(definitionIds[2], inId, outId);

            if (shift + radix_width < sizeof(unsigned int) * 8) // Not the last iteration
            {
                inOutSwapped = !inOutSwapped;
            }
        }

        if (inOutSwapped)
        {
            // Copy contents of in to out, since they are swapped
            interface.CopyBuffer(outId, inId);
        }
    });

    tuner.SetArguments(definitionIds[0], {inId, isumsId, sizeId, shiftId});
    tuner.SetArguments(definitionIds[1], {isumsId, numberOfGroupsId});
    tuner.SetArguments(definitionIds[2], {inId, isumsId, outId, sizeId, shiftId});

    // Parameter for the length of OpenCL vector data types used in the kernels
    if constexpr (computeApi == ktt::ComputeApi::OpenCL)
    {
        tuner.AddParameter(kernel, "FPVECTNUM", std::vector<uint64_t>{4, 8, 16});
    }
    else
    {
        tuner.AddParameter(kernel, "FPVECTNUM", std::vector<uint64_t>{4});
    }

    // Local size below 128 does not work correctly, not even with the benchmark code
    tuner.AddParameter(kernel, "LOCAL_SIZE", std::vector<uint64_t>{128, 256, 512});
    tuner.AddThreadModifier(kernel, definitionIds, ktt::ModifierType::Local, ktt::ModifierDimension::X, "LOCAL_SIZE",
        ktt::ModifierAction::Multiply);

    // Second kernel global size is always equal to local size
    tuner.AddParameter(kernel, "GLOBAL_SIZE", std::vector<uint64_t>{512, 1024, 2048, 4096, 8192, 16384, 32768});
    tuner.AddThreadModifier(kernel, {definitionIds[0], definitionIds[2]}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
        "GLOBAL_SIZE", ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definitionIds[1]}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "LOCAL_SIZE",
        ktt::ModifierAction::Multiply);

    auto workGroupConstraint = [](const std::vector<uint64_t>& vector) {return vector.at(0) != 128 || vector.at(1) != 32768;};
    tuner.AddConstraint(kernel, {"LOCAL_SIZE", "GLOBAL_SIZE"}, workGroupConstraint);

    tuner.SetReferenceComputation(outId, [&in](void* buffer)
    {
        std::memcpy(buffer, in.data(), in.size() * sizeof(unsigned int));
        unsigned int* intArray = static_cast<unsigned int*>(buffer);
        std::sort(intArray, intArray + in.size());
    });

    const auto results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "SortOutput", ktt::OutputFormat::JSON);

    return 0;
}
