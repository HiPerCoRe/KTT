// KTT tutorial demonstrating kernel setup and execution via KTT
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

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

    const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId scalarId = tuner.AddArgumentScalar(scalar);
    tuner.SetArguments(definition, {aId, bId, resultId, scalarId});

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    // By default KTT makes a copy of the input data once and then doesn't update that copy. The intended solution is to update
    // input data explicitly in the kernel launcher. For scalars, it is the only solution. For vectors, another solution is to
    // set the argument to be referenced from user data, without making a KTT-managed copy (see overloads of AddArgumentVector),
    // and then disable caching of read-only arguments, so they are pulled from user data every (tuning) run.
    //
    // Do note that any operations inside the kernel launcher will be included in the total duration measurement, and so could
    // skew the best configuration selection, or the searcher behavior. In this case the effect should not be significant,
    // because all configurations transfer the same amount of data, so they should be slowed roughly equally.
    tuner.SetLauncher(kernel, [&a, &aId, &b, &bId, &scalar, &scalarId, &definition](ktt::ComputeInterface &interface){
        interface.UpdateBuffer(aId, a.data());
        interface.UpdateBuffer(bId, b.data());
        interface.UpdateScalarArgument(scalarId, &scalar);
        // Do not forget to also run the kernel definition.
        interface.RunKernel(definition);
    });

    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

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
    std::cout << "\n\n";

    a.clear();
    b.clear();
    a.resize(numberOfElements, 0);
    b.resize(numberOfElements, 0);

    tuner.Run(kernel, {}, {ktt::BufferOutputDescriptor(resultId, result.data())});
    // Print first ten elements from the result to check they were computed correctly.
    std::cout << "\nPrinting the first 10 elements from result after zeroing input buffers but leaving scalar = 3.0: \n";
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << "\n\n";

    scalar = 0;

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
