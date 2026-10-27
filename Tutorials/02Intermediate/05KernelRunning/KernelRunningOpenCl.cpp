// KTT tutorial demonstrating kernel setup and execution via KTT
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

int main(int argc, char** argv)
{
    // Initialize platform index, device index and path to kernel.
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + KTT_TUTORIAL_KERNEL_FILE;

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

    const size_t numberOfElements = 1024 * 1024;
    const ktt::DimensionVector ndRangeDimensions(numberOfElements);
    const ktt::DimensionVector workGroupDimensions(256);
    
    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);
    float scalar= 3.0f;

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);

    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, ndRangeDimensions,
        workGroupDimensions);
    // By default, KTT makes a copy of the input data once and then doesn't update that copy. The intended solution is to update
    // input data explicitly via Tuner methods, as using the ComputeInterface methods in the launcher is counted towards
    // kernel duration. 
    // For vector types, do not forget to set ArgumentManagementType to User and referenceUserData to true to ensure
    // consistent behavior.
    // 
    // For scalars, it is the only solution. For vectors, another solution is to
    // disable caching of read-only arguments, so they are pulled from user data every (tuning) run.
    // This requires management type to be Framework and referenceUserData to be true. KTT's reference will,
    // however, be invalidated whenever vector size changes.

    // Note the explicit setting of management type and reference flag.
    const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::User, true);
    const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::User, true);

    // Even write-only buffers must be user-managed if we intend to use Tuner::DownloadBuffer. Otherwise, buffers
    // are cleared after each kernel run.
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::User, true);
    const ktt::ArgumentId scalarId = tuner.AddArgumentScalar(scalar);
    tuner.SetArguments(definition, {aId, bId, resultId, scalarId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // The buffer must be uploaded manually before running the kernel for the first time if management type is User. 
    // Note that these method calls do not count towards DataMovementOverhead, unlike the default framework management.
    // This does not affect which configuration is selected as best, as it is not counted in the kernel's total duration 
    // (only calls made inside the kernel launcher are). If the user wishes to measure the data overhead regardless, it has 
    // to be done manually.
    tuner.UpdateBuffer(aId);
    tuner.UpdateBuffer(bId);
    tuner.UpdateBuffer(resultId);

    // Run the specified kernel. The second argument would contain the configuration that the kernel should run with, 
    // if there were any. The third argument is used to retrieve the kernel output. For each kernel argument that
    // is retrieved, one BufferOutputDescriptor must be specified. Each of these descriptors contains id of the retrieved argument
    // and memory location where the argument data will be stored. Optionally, it can also include number of bytes to be retrieved,
    // if only a part of the argument is needed. Here, the data is stored back into result buffer which was created earlier. Note
    // that the memory location size needs to be equal or greater than the retrieved argument size.
    tuner.Run(kernel, {}, {ktt::BufferOutputDescriptor(resultId, result.data())});

    // Print first ten elements from the result to check they were computed correctly.
    std::cout << "\nPrinting the first 10 elements from result: \n";
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << "\n";

    // It is also possible to download the data after running the kernel. Again, destination size must be greater or
    // equal to buffer size.
    std::vector<float>result_copy(numberOfElements);
    tuner.DownloadBuffer(resultId, result_copy.data());
    std::cout << "\nPrinting the first 10 elements from result_copy: \n";
    for (size_t i = 0; i < 10; ++i) 
    {
        std:: cout << result_copy[i] << " ";
    }
    std::cout << "\n\n";
    
    std::fill(a.begin(), a.end(), 0);
    b.clear();
    b.resize(numberOfElements, 0);

    // b was resized, so we must provide an explicit data pointer to it. On the other hand, a wasn't.
    tuner.UpdateBuffer(aId);
    tuner.UpdateBuffer(bId, b.data()/*, b.size() * sizeof(float)*/); // Would also need to give an explicit size if
                                                                     // it was different in the end.

    tuner.Run(kernel, {}, {ktt::BufferOutputDescriptor(resultId, result.data())});
    // Print first ten elements from the result to check they were computed correctly.
    std::cout << "\nPrinting the first 10 elements from result after zeroing input buffers but leaving scalar = 3.0: \n";
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << "\n\n";

    scalar = 0;
    // Type of the input must have the same size as what the kernel argument was initialized with.
    tuner.UpdateScalarArgument(scalarId, &scalar); 

    tuner.Run(kernel, {}, {ktt::BufferOutputDescriptor(resultId, result.data())});
    // Print first ten elements from the result to check they were computed correctly.
    std::cout << "\nPrinting the first 10 elements from result after zeroing scalar as well: \n";
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
