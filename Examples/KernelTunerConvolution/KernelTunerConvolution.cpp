#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../ExampleReferenceKernel.h"
#include "Api/Configuration/DimensionVector.h"

using namespace std;

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b)
{
    return a % b == 0;
}

class KernelTunerConvolution: public ExampleReferenceKernel {
protected:
    KernelTunerConvolution(std::shared_ptr<ExampleRefKernelConfiguration> config, int defaultProblemSize,
              string exampleFolderPath, string defaultKernelFileBaseName,
              string defaultReferenceKernelFileBaseName):
        ExampleReferenceKernel(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName,
                defaultReferenceKernelFileBaseName)
    {
        m_kSizeX = 1024*static_cast<int>(sqrtf(m_problemSize));
        m_kSizeY = m_kSizeX;

        ndRangeDimensions = ktt::DimensionVector(m_kSizeX, m_kSizeY);
    }

    friend ExampleReferenceKernel;

    // Declare data variables
    uint32_t m_kSizeX; // Matrix dimension X
    uint32_t m_kSizeY; // Matrix dimension Y
    
    ktt::DimensionVector ndRangeDimensions;

    const unsigned HFS = 7; // Half m_filter size
    const unsigned FS = (HFS + HFS + 1); // Filter size

    std::vector<float> m_input;
    std::vector<float> m_output;
    std::vector<float> m_filter;

    ktt::ArgumentId m_inputId;
    ktt::ArgumentId m_filterId; 
    ktt::ArgumentId m_outputId;
    ktt::ArgumentId m_sizeXId;
    ktt::ArgumentId m_sizeYId;
    ktt::ArgumentId m_filterXId;
    ktt::ArgumentId m_filterYId;
       

    void InitData() override 
    {
        m_input.resize((m_kSizeX + 2 * HFS) * (m_kSizeY + 2 * HFS), 0.0f);
        m_output.resize(m_kSizeX * m_kSizeY, 0.0f);
        m_filter.resize((2*HFS+1)*(2*HFS+1));

        FillBuffers<float>({&m_input, &m_filter}, -2.f, 2.f);
    }

    void InitKernel() override 
    {
        // Add all arguments utilized by kernels
        m_outputId = m_tuner->AddArgumentVector(m_output, ktt::ArgumentAccessType::WriteOnly);
        m_inputId = m_tuner->AddArgumentVector(m_input, ktt::ArgumentAccessType::ReadOnly);
        m_filterId = m_tuner->AddArgumentVector(m_filter, ktt::ArgumentAccessType::ReadOnly);
        cout << m_outputId << endl;
        m_sizeXId = m_tuner->AddArgumentScalar(m_kSizeX);
        m_sizeYId = m_tuner->AddArgumentScalar(m_kSizeY);
        m_filterXId = m_tuner->AddArgumentScalar(HFS*2+1);
        m_filterYId = m_tuner->AddArgumentScalar(HFS*2+1);

        InitKernelDefault("Convolution", "Convolution", ndRangeDimensions, 
            {m_outputId, m_inputId, m_filterId});
    }

    void InitReference() override 
    {
        const ktt::DimensionVector referenceWorkGroupDimensions(8, 8);

        InitReferenceKernelDefault("ConvolutionReference", ndRangeDimensions, referenceWorkGroupDimensions,
            {m_outputId, m_inputId, m_filterId, m_sizeXId, m_sizeYId, m_filterXId, m_filterYId}, {m_outputId});
    }

    void InitTuningSpace() override
    {
        // fake tuning parameters, encoding m_input
        m_tuner->AddParameter(m_kernel, "IMAGE_WIDTH", std::vector<uint64_t>{m_kSizeX});
        m_tuner->AddParameter(m_kernel, "IMAGE_HEIGHT", std::vector<uint64_t>{m_kSizeY});
        m_tuner->AddParameter(m_kernel, "HFS", std::vector<uint64_t>{HFS});

        // tuning parameters
        m_tuner->AddParameter(m_kernel, "BLOCK_SIZE_X", std::vector<uint64_t>{1, 2, 4, 8, 16, 32, 48, 64, 96, 112, 128});
        m_tuner->AddParameter(m_kernel, "BLOCK_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
        m_tuner->AddParameter(m_kernel, "TILE_SIZE_X", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8});
        m_tuner->AddParameter(m_kernel, "TILE_SIZE_Y", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8});
        m_tuner->AddParameter(m_kernel, "PADDING", std::vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "READ_ONLY", std::vector<uint64_t>{0, 1});

        // Add kernel dimension modifiers based on added tuning parameters
        auto globalModifier = [](const uint64_t size, const std::vector<uint64_t>& vector)
        {
            return (((size+vector.at(0)-1) / vector.at(0))+vector.at(1)-1) / vector.at(1);
        };

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"BLOCK_SIZE_X", "TILE_SIZE_X"},
            globalModifier);
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"BLOCK_SIZE_Y", "TILE_SIZE_Y"},
            globalModifier);

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE_X", ktt::ModifierAction::Multiply);
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "BLOCK_SIZE_Y", ktt::ModifierAction::Multiply);

        // Add constraints
        auto minWGConstraint = [](const std::vector<uint64_t>& v) {return v[0] * v[1] >= 64;};
        m_tuner->AddConstraint(m_kernel, {"BLOCK_SIZE_X", "BLOCK_SIZE_Y"}, minWGConstraint);
        auto maxWGConstraint = [](const std::vector<uint64_t>& v) {return v[0] * v[1] <= 1024;};
        m_tuner->AddConstraint(m_kernel, {"BLOCK_SIZE_X", "BLOCK_SIZE_Y"}, maxWGConstraint);
        auto tileConstraint = [](const std::vector<uint64_t>& v) {return v[0] *v[1] < 30;};
        m_tuner->AddConstraint(m_kernel, {"TILE_SIZE_X", "TILE_SIZE_Y"}, tileConstraint);
    }
};

int main(int argc, char** argv)
{
    unique_ptr<KernelTunerConvolution> ktConvolution = KernelTunerConvolution::Create<KernelTunerConvolution> (
        argc, argv, 16, "Examples/KernelTunerConvolution", "KernelTunerConvolution", "KernelTunerConvolutionReference"
    );
    ktConvolution->Run();

    return 0;
};
