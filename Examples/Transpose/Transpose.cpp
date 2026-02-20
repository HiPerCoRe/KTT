#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../Example.h"

class Transpose: public Example {
public:
    Transpose(int argc, char** argv): Example(argc, argv, 16, "Examples/Transpose", "Transpose", "TransposeReference"),
        m_width(1024*static_cast<int>(std::sqrtf(m_problemSize))),
        m_height(m_width)
    {
        
    }

protected:
    const int m_width;
    const int m_height;
    std::vector<float> m_dst;
    std::vector<float> m_src;

    ktt::KernelDefinitionId m_referenceDefinition;
    ktt::KernelId m_referenceKernel;
    
    ktt::ArgumentId m_srcId;
    ktt::ArgumentId m_dstId;
    ktt::ArgumentId m_widthId;
    ktt::ArgumentId m_heightId;
    

    void InitData() override 
    {
        // Declare data variables
        m_dst.resize(m_width * m_height);
        m_src.resize(m_width * m_height);

        // Initialize data
        std::random_device device;
        std::default_random_engine engine(device());
        std::uniform_real_distribution<float> distribution(0.0f, 10.0f);

        for (int i = 0; i < m_width * m_height; ++i)
        {
            m_src[i] = distribution(engine);
        }
    }

    void InitKernels() override
    {
        const ktt::DimensionVector ndRangeDimensions(m_width, m_height);
        const ktt::DimensionVector ndRangeDimensionsReference(m_width / 16, m_height / 16);
        const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);

        // Create m_kernel and configure input/output
        m_definition = m_tuner.AddKernelDefinitionFromFile("mtran", m_kernelFile, ndRangeDimensions,
            ktt::DimensionVector(1, 1));
        m_referenceDefinition = m_tuner.AddKernelDefinitionFromFile("mtranReference", m_referenceKernelFile,
            ndRangeDimensionsReference, referenceWorkGroupDimensions);
        
        m_kernel = m_tuner.CreateSimpleKernel("Transposition", m_definition);
        m_referenceKernel = m_tuner.CreateSimpleKernel("TranspositionReference", m_referenceDefinition);

    }

    void InitReference() override 
    {
        if (!m_rapidTest)
        {
            m_tuner.SetReferenceKernel(m_dstId, m_referenceKernel, ktt::KernelConfiguration());
            m_tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.0001);
        }
    }
    
    void InitKernelArguments() override 
    {
        m_srcId = m_tuner.AddArgumentVector(m_src, ktt::ArgumentAccessType::ReadOnly);
        m_dstId = m_tuner.AddArgumentVector(m_dst, ktt::ArgumentAccessType::WriteOnly);
        m_widthId = m_tuner.AddArgumentScalar(m_width);
        m_heightId = m_tuner.AddArgumentScalar(m_height);
        
        m_tuner.SetArguments(m_definition, {m_dstId, m_srcId, m_widthId, m_heightId});
        m_tuner.SetArguments(m_referenceDefinition, {m_dstId, m_srcId, m_widthId, m_heightId});
    }
    
    void InitTuningParameters() override 
    {
        // Create tuning space
        if (m_computeApi == ktt::ComputeApi::OpenCL)
        {
            m_tuner.AddParameter(m_kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4, 8});
            m_tuner.AddParameter(m_kernel, "PREFETCH", std::vector<uint64_t>{0, 1, 2});
        }
        else
        {
            m_tuner.AddParameter(m_kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4});
        }

        m_tuner.AddParameter(m_kernel, "CR", std::vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "LOCAL_MEM", std::vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "PADD_LOCAL", std::vector<uint64_t>{0, 1});

        const std::vector<uint64_t> sizeRanges = {1, 2, 4, 8, 16, 32, 64};

        m_tuner.AddParameter(m_kernel, "WORK_GROUP_SIZE_X", sizeRanges);
        m_tuner.AddParameter(m_kernel, "WORK_GROUP_SIZE_Y", sizeRanges);
        m_tuner.AddParameter(m_kernel, "TILE_SIZE_X", sizeRanges);
        m_tuner.AddParameter(m_kernel, "TILE_SIZE_Y", sizeRanges);
        m_tuner.AddParameter(m_kernel, "DIAGONAL_MAP", std::vector<uint64_t>{0, 1});
        
        // Constrain tuning space
        auto xConstraint = [] (const std::vector<uint64_t>& v) { return (v[0] == v[1]); };
        auto yConstraint = [] (const std::vector<uint64_t>& v) { return ((v[1] <= v[0]) && (v[0] % v[1] == 0)); };
        auto tConstraint = [] (const std::vector<uint64_t>& v) { return (!v[0] || (v[1] <= v[2]*v[3])); };
        auto pConstraint = [] (const std::vector<uint64_t>& v) { return (v[0] || !v[1]); };
        auto vlConstraint = [] (const std::vector<uint64_t>& v) { return (!v[0] || v[1] == 1); };

        uint64_t maxMult = 64;
        auto vConstraint = [maxMult] (const std::vector<uint64_t>& v) { return (v[0]*v[1] <= maxMult); };
        
        m_tuner.AddConstraint(m_kernel, {"TILE_SIZE_X", "WORK_GROUP_SIZE_X"}, xConstraint);
        m_tuner.AddConstraint(m_kernel, {"TILE_SIZE_Y", "WORK_GROUP_SIZE_Y"}, yConstraint);
        m_tuner.AddConstraint(m_kernel, {"LOCAL_MEM", "TILE_SIZE_Y", "WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, tConstraint);
        m_tuner.AddConstraint(m_kernel, {"LOCAL_MEM", "PADD_LOCAL"}, pConstraint);
        m_tuner.AddConstraint(m_kernel, {"LOCAL_MEM", "VECTOR_TYPE"}, vlConstraint);
        m_tuner.AddConstraint(m_kernel, {"TILE_SIZE_X", "VECTOR_TYPE"}, vConstraint);

        // Configure parallelism
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
            ktt::ModifierAction::Multiply);
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
            ktt::ModifierAction::Multiply);
        auto xGlobalModifier = [](const uint64_t size, const std::vector<uint64_t>& vector) {return size / vector.at(0) / vector.at(1);};
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
            {"TILE_SIZE_X", "VECTOR_TYPE"}, xGlobalModifier);
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "TILE_SIZE_Y",
            ktt::ModifierAction::Divide);

        auto wgSize = [](const std::vector<uint64_t>& v) {return v[0]*v[1] >= 32;};
        m_tuner.AddConstraint(m_kernel, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, wgSize);
    }
};

int main(int argc, char **argv)
{
    Transpose transpose(argc, argv);
    /*
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

    // Declare and initialize data
    int m_width;
    int m_height;

    if constexpr (!useProfiling)
    {
        m_width = 8192;
        m_height = 8192;
    }
    else
    {
        m_width = 4096*2;
        m_height = 4096*2;
    }

    const ktt::DimensionVector ndRangeDimensions(m_width, m_height);
    const ktt::DimensionVector ndRangeDimensionsReference(m_width / 16, m_height / 16);
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);

    // Declare data variables
    std::vector<float> m_dst(m_width * m_height);
    std::vector<float> m_src(m_width * m_height);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 10.0f);

    for (int i = 0; i < m_width * m_height; ++i)
    {
        m_src[i] = distribution(engine);
    }

    // Create m_tuner
    ktt::m_tuner tuner(platformIndex, deviceIndex, computeApi);
    m_tuner.SetGlobalSizeType(ktt::GlobalSizeType::CUDA);
    m_tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
    //m_tuner.SetLoggingLevel(ktt::LoggingLevel::Debug);

    if constexpr (useProfiling)
    {
        printf("Executing with profiling switched ON.\n");
        m_tuner.SetProfiling(true);
    }

    // Create m_kernel and configure input/output
    const ktt::KernelDefinitionId m_definition = m_tuner.AddKernelDefinitionFromFile("mtran", kernelFile, ndRangeDimensions,
        ktt::DimensionVector(1, 1));
    const ktt::KernelId referenceDefinition = m_tuner.AddKernelDefinitionFromFile("mtranReference", referenceKernelFile,
        ndRangeDimensionsReference, referenceWorkGroupDimensions);
    
    const ktt::KernelId m_kernel = m_tuner.CreateSimpleKernel("Transposition", m_definition);
    const ktt::KernelId referenceKernel = m_tuner.CreateSimpleKernel("TranspositionReference", referenceDefinition);

    const ktt::ArgumentId srcId = m_tuner.AddArgumentVector(m_src, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId dstId = m_tuner.AddArgumentVector(m_dst, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId widthId = m_tuner.AddArgumentScalar(m_width);
    const ktt::ArgumentId heightId = m_tuner.AddArgumentScalar(m_height);
    
    m_tuner.SetArguments(m_definition, {dstId, srcId, widthId, heightId});
    m_tuner.SetArguments(referenceDefinition, {dstId, srcId, widthId, heightId});

    // Create tuning space
    if constexpr (computeApi == ktt::ComputeApi::OpenCL)
    {
        m_tuner.AddParameter(m_kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4, 8});
        m_tuner.AddParameter(m_kernel, "PREFETCH", std::vector<uint64_t>{0, 1, 2});
    }
    else
    {
        m_tuner.AddParameter(m_kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4});
    }

    m_tuner.AddParameter(m_kernel, "CR", std::vector<uint64_t>{0, 1});
    m_tuner.AddParameter(m_kernel, "LOCAL_MEM", std::vector<uint64_t>{0, 1});
    m_tuner.AddParameter(m_kernel, "PADD_LOCAL", std::vector<uint64_t>{0, 1});

    std::vector<uint64_t> sizeRanges;

    if constexpr (!useWideParameters && !useDenseParameters)
    {
        sizeRanges = {1, 2, 4, 8, 16, 32, 64};
    }
    else if constexpr (!useWideParameters)
    {
        sizeRanges = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64};
    }
    else
    {
        sizeRanges = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128};
    }

    m_tuner.AddParameter(m_kernel, "WORK_GROUP_SIZE_X", sizeRanges);
    m_tuner.AddParameter(m_kernel, "WORK_GROUP_SIZE_Y", sizeRanges);
    m_tuner.AddParameter(m_kernel, "TILE_SIZE_X", sizeRanges);
    m_tuner.AddParameter(m_kernel, "TILE_SIZE_Y", sizeRanges);
    m_tuner.AddParameter(m_kernel, "DIAGONAL_MAP", std::vector<uint64_t>{0, 1});
    
    // Constraint tuning space
    auto xConstraint = [] (const std::vector<uint64_t>& v) { return (v[0] == v[1]); };
    auto yConstraint = [] (const std::vector<uint64_t>& v) { return ((v[1] <= v[0]) && (v[0] % v[1] == 0)); };
    auto tConstraint = [] (const std::vector<uint64_t>& v) { return (!v[0] || (v[1] <= v[2]*v[3])); };
    auto pConstraint = [] (const std::vector<uint64_t>& v) { return (v[0] || !v[1]); };
    auto vlConstraint = [] (const std::vector<uint64_t>& v) { return (!v[0] || v[1] == 1); };

    uint64_t maxMult = 64;

    if constexpr (useWideParameters)
    {
        maxMult = 128;
    }

    auto vConstraint = [maxMult] (const std::vector<uint64_t>& v) { return (v[0]*v[1] <= maxMult); };
    
    m_tuner.AddConstraint(m_kernel, {"TILE_SIZE_X", "WORK_GROUP_SIZE_X"}, xConstraint);
    m_tuner.AddConstraint(m_kernel, {"TILE_SIZE_Y", "WORK_GROUP_SIZE_Y"}, yConstraint);
    m_tuner.AddConstraint(m_kernel, {"LOCAL_MEM", "TILE_SIZE_Y", "WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, tConstraint);
    m_tuner.AddConstraint(m_kernel, {"LOCAL_MEM", "PADD_LOCAL"}, pConstraint);
    m_tuner.AddConstraint(m_kernel, {"LOCAL_MEM", "VECTOR_TYPE"}, vlConstraint);
    m_tuner.AddConstraint(m_kernel, {"TILE_SIZE_X", "VECTOR_TYPE"}, vConstraint);

    // Configure parallelism
    m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);
    m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
        ktt::ModifierAction::Multiply);
    auto xGlobalModifier = [](const uint64_t size, const std::vector<uint64_t>& vector) {return size / vector.at(0) / vector.at(1);};
    m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
        {"TILE_SIZE_X", "VECTOR_TYPE"}, xGlobalModifier);
    m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "TILE_SIZE_Y",
        ktt::ModifierAction::Divide);

    auto wgSize = [](const std::vector<uint64_t>& v) {return v[0]*v[1] >= 32;};
    m_tuner.AddConstraint(m_kernel, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, wgSize);

    if constexpr (!rapidTest)
    {
        m_tuner.SetReferenceKernel(dstId, referenceKernel, ktt::KernelConfiguration());
        m_tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.0001);
    }

    // Perform tuning
    const auto results = m_tuner.Tune(m_kernel/*, std::make_unique<ktt::ConfigurationCount>(1)*);
    m_tuner.SaveResults(results, "TranspositionOutput", ktt::OutputFormat::XML);
    m_tuner.SaveResults(results, "TranspositionOutput", ktt::OutputFormat::JSON);
    */
    return 0;
}
