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

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/KernelTunerPnpoly/KernelTunerPnpoly.cu";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/KernelTunerPnpoly/KernelTunerPnpolyReference.cu";
    const std::string defaultMlModel = kernelPrefix + "../Examples/KernelTunerPnpoly/Models/3090-KTPnpolyOutput_DT.sav";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

// Toggle rapid test (e.g., disable output validation).
const bool rapidTest = true;

// Toggle kernel profiling.
const bool useProfiling = false;

// Toggle usage of profile-based searcher
const bool useProfileSearcher = true;

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b)
{
    return ((a / b) * b == a) ? true : false;
}

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = defaultKernelFile;
    std::string referenceKernelFile = defaultReferenceKernelFile;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));

        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));

            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);

                if (argc >= 5)
                {
                    referenceKernelFile = std::string(argv[4]);
                }
            }
        }
    }

    // Declare data variables
    uint32_t dataSize; 
    uint32_t vertSize;

    if constexpr (!useProfiling)
    {
        dataSize = 20000000;
        vertSize = 600;
    }
    else
    {
        dataSize = 20000000;
        vertSize = 600;
    }

    const ktt::DimensionVector ndRangeDimensions(dataSize);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceNdRangeDimensions(dataSize/256);
    const ktt::DimensionVector referenceWorkGroupDimensions(256);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    std::vector<int> bitmap(dataSize, 0);
    std::vector<float> points(dataSize*2, 0.0f);
    std::vector<float> vertices(vertSize*2, 1.0);

    // Populates input data structure by padded data
    for (size_t i = 0; i < dataSize*2; ++i)
    {
        points[i] = distribution(engine);
    }
    for (size_t i = 0; i < vertSize*2; ++i)
    {
        vertices[i] = distribution(engine);
    }


    // Create tuner object for chosen platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    //tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    if constexpr (useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        tuner.SetProfiling(true);
    }

    // Add two kernels to tuner, one of the kernels acts as reference kernel
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("Pnpoly", kernelFile, ndRangeDimensions, workGroupDimensions);
    const ktt::KernelDefinitionId referenceDefinition = tuner.AddKernelDefinitionFromFile("PnpolyReference", referenceKernelFile, referenceNdRangeDimensions, referenceWorkGroupDimensions);

    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Pnpoly", definition);
    const ktt::KernelId referenceKernel = tuner.CreateSimpleKernel("PnpolyReference", referenceDefinition);

    // fake tuning parameters, encoding input
    tuner.AddParameter(kernel, "VERTICES", std::vector<uint64_t>{vertSize});

    // tuning parameters
    tuner.AddParameter(kernel, "BLOCK_SIZE_X", std::vector<uint64_t>{32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992});
    tuner.AddParameter(kernel, "TILE_SIZE", std::vector<uint64_t>{1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
    tuner.AddParameter(kernel, "BETWEEN_METHOD", std::vector<uint64_t>{0, 1, 2, 3});
    tuner.AddParameter(kernel, "USE_METHOD", std::vector<uint64_t>{0, 1, 2});

    // Add kernel dimension modifiers based on added tuning parameters
    auto globalModifier = [](const uint64_t size, const std::vector<uint64_t>& vector)
    {
        return (((size+vector.at(0)-1) / vector.at(0))+vector.at(1)-1) / vector.at(1);
    };

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"BLOCK_SIZE_X", "TILE_SIZE"},
        globalModifier);

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE_X", ktt::ModifierAction::Multiply);

    // Add all arguments utilized by kernels
    const ktt::ArgumentId pointsId = tuner.AddArgumentVector(points, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId verticesId = tuner.AddArgumentVector(vertices, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId bitmapId = tuner.AddArgumentVector(bitmap, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId vertSizeId = tuner.AddArgumentScalar(vertSize);
    const ktt::ArgumentId dataSizeId = tuner.AddArgumentScalar(dataSize);

    tuner.SetArguments(definition, {bitmapId, pointsId, verticesId, dataSizeId}); 
    tuner.SetArguments(referenceDefinition, {bitmapId, pointsId, verticesId, dataSizeId, vertSizeId}); 

    if constexpr (!rapidTest)
    {
        //tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);
        tuner.SetReferenceKernel(bitmapId, referenceKernel, ktt::KernelConfiguration());
    }

    // Launch kernel tuning
    //tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
    if constexpr (useProfileSearcher)
    {
        tuner.SetProfileBasedSearcher(kernel, defaultMlModel, false);
    }
    //else
    //    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
    const auto results = tuner.Tune(kernel, std::make_unique<ktt::ConfigurationCount>(600));
    //const auto results = tuner.Tune(kernel);
    tuner.SaveResults(results, "KTPnpolyOutput", ktt::OutputFormat::XML);
    tuner.SaveResults(results, "KTPnpolyOutput", ktt::OutputFormat::JSON);

    return 0;
};
