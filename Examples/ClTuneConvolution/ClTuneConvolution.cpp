#include "../ExampleReferenceKernel.h"
#include "Api/Configuration/DimensionVector.h"
#include <memory>
#include <cmath>

using namespace std;

class ClTuneConvolution : public ExampleReferenceKernel {
public:
    ClTuneConvolution(int argc, char **argv, string exampleFolderPath,
                      string defaultKernelFileBaseName, string defaultRefKernelFileBaseName) :
        ExampleReferenceKernel(argc, argv, exampleFolderPath, defaultKernelFileBaseName, defaultRefKernelFileBaseName),
        kSizeX(4096),
        kSizeY(4096),
        m_problemSize(kSizeX, kSizeY)
    {
    }

    friend ExampleReferenceKernel;

    const unsigned HFS = 3; // Half filter size
    const unsigned FS = (HFS + HFS + 1); // Filter size

    uint32_t kSizeX;
    uint32_t kSizeY;
    ktt::DimensionVector m_problemSize;

    vector<float> mat_a;
    vector<float> mat_b;
    vector<float> coeff;

    ktt::ArgumentId kSizeXId;
    ktt::ArgumentId kSizeYId;
    ktt::ArgumentId matAId;
    ktt::ArgumentId coeffId;
    ktt::ArgumentId matBId;

    ktt::KernelDefinitionId m_definition;

    // Helper function to perform an integer division + ceiling (round-up)
    static size_t CeilDiv(size_t a, size_t b) {
        return (a + b - 1) / b;
    }

    // Helper function to determine whether or not 'a' is a multiple of 'b'
    static bool IsMultiple(size_t a, size_t b) {
        return a % b == 0;
    }

protected:
    void InitCLI() override
    {
        ExampleBase::InitCLI();

        UseInputSizeOption(2, m_problemSize);
    }

    void InitData() override
    {
        kSizeX = m_problemSize.GetSizeX();
        kSizeY = m_problemSize.GetSizeY();
        mat_a.resize((kSizeX + 2 * HFS) * (kSizeY + 2 * HFS), 0.0f);
        mat_b.resize(kSizeX * kSizeY, 0.0f);
        coeff.resize(FS * FS);

        FillBuffers<float>({&mat_a}, -2.0f, 2.0f);

        float sigma = 1.0f;
        float mean = FS / 2.0f;
        float sum = 0.0f;

        for (size_t x = 0; x < FS; ++x) {
            for (size_t y = 0; y < FS; ++y) {
                const float exponent = -0.5f * (pow((x - mean) / sigma, 2.0f) + pow((y - mean) / sigma, 2.0f));
                coeff[y * FS + x] = static_cast<float>(exp(exponent) / (2.0f * 3.14159265f * sigma * sigma));
                sum += coeff[y * FS + x];
            }
        }

        for (auto& item : coeff) {
            item = item / sum;
        }
    }

    void InitKernel() override
    {
        m_tuner->SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
        kSizeXId = m_tuner->AddArgumentScalar(kSizeX);
        kSizeYId = m_tuner->AddArgumentScalar(kSizeY);
        matAId = m_tuner->AddArgumentVector(mat_a, ktt::ArgumentAccessType::ReadOnly);
        coeffId = m_tuner->AddArgumentVector(coeff, ktt::ArgumentAccessType::ReadOnly);
        matBId = m_tuner->AddArgumentVector(mat_b, ktt::ArgumentAccessType::WriteOnly);

        const ktt::DimensionVector ndRangeDimensions(kSizeX, kSizeY);
        const ktt::DimensionVector workGroupDimensions;

        m_definition = m_tuner->AddKernelDefinitionFromFile("conv", m_kernelFile, ndRangeDimensions, workGroupDimensions);

        m_kernel = m_tuner->CreateSimpleKernel("Convolution", m_definition);

        m_tuner->SetArguments(m_definition, {kSizeXId, kSizeYId, matAId, coeffId, matBId});
    }

    void InitTuningSpace() override
    {
        vector<uint64_t> blockRange = {8, 16, 32, 64};
        vector<uint64_t> wptRange = {1, 2, 4, 8};

        m_tuner->AddParameter(m_kernel, "TBX", blockRange);
        m_tuner->AddParameter(m_kernel, "TBY", blockRange);
        m_tuner->AddParameter(m_kernel, "LOCAL", vector<uint64_t>{0, 1, 2});
        m_tuner->AddParameter(m_kernel, "WPTX", wptRange);
        m_tuner->AddParameter(m_kernel, "WPTY", wptRange);
        m_tuner->AddParameter(m_kernel, "VECTOR", vector<uint64_t>{1, 2, 4});
        m_tuner->AddParameter(m_kernel, "UNROLL_FACTOR", vector<uint64_t>{1, static_cast<uint64_t>(FS)});
        m_tuner->AddParameter(m_kernel, "PADDING", vector<uint64_t>{0, 1});

        vector<uint64_t> integers{8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74};

        // Introduces a helper parameter to compute the proper number of threads for the LOCAL == 2 case.
        // In this case, the workgroup size (TBX by TBY) is extra large (TBX_XL by TBY_XL) because it uses
        // extra threads to compute the halo threads. How many extra threads are needed is dependend on
        // the filter size. Here we support a the TBX and TBY size plus up to 10 extra threads.
        m_tuner->AddParameter(m_kernel, "TBX_XL", integers);
        m_tuner->AddParameter(m_kernel, "TBY_XL", integers);

        // Add kernel dimension modifiers based on added tuning parameters
        auto globalModifier = [](const uint64_t size, const vector<uint64_t>& v) {
            return (size * v[0]) / (v[1] * v[2]);
        };

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"TBX_XL", "TBX", "WPTX"}, globalModifier);
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"TBY_XL", "TBY", "WPTY"}, globalModifier);

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "TBX_XL", ktt::ModifierAction::Multiply);
        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "TBY_XL", ktt::ModifierAction::Multiply);

        // Add constraints
        auto haloThreads = [this](const vector<uint64_t>& v) {
            if (v[0] == 2) { return (v[1] == v[2] + CeilDiv(2 * HFS, v[3])); } // With halo threads
            else { return (v[1] == v[2]); } // Without halo threads
        };

        m_tuner->AddConstraint(m_kernel, {"LOCAL", "TBX_XL", "TBX", "WPTX"}, haloThreads);
        m_tuner->AddConstraint(m_kernel, {"LOCAL", "TBY_XL", "TBY", "WPTY"}, haloThreads);

        // Sets the constrains on the vector size
        auto vectorConstraint = [this](const vector<uint64_t>& v) {
            if (v[0] == 2) { return IsMultiple(v[2], v[1]) && IsMultiple(2 * HFS, v[1]); }
            else { return IsMultiple(v[2], v[1]); }
        };

        m_tuner->AddConstraint(m_kernel, {"LOCAL", "VECTOR", "WPTX"}, vectorConstraint);

        // Sets padding to zero in case local memory is not used
        auto paddingConstraint = [](const vector<uint64_t>& v) { return (v[1] == 0 || v[0] != 0); };
        m_tuner->AddConstraint(m_kernel, {"LOCAL", "PADDING"}, paddingConstraint);

        // Ensure divisibility
        auto divConstraint = [](const vector<uint64_t>& v) { return v[0] % v[1] == 0; };
        m_tuner->AddConstraint(m_kernel, {"TBX", "WPTX"}, divConstraint);
        m_tuner->AddConstraint(m_kernel, {"TBY", "WPTY"}, divConstraint);
    }

    void InitReference() override
    {
        const ktt::DimensionVector ndRangeDimensions(kSizeX, kSizeY);
        const ktt::DimensionVector referenceWorkGroupDimensions(8, 8);

        InitReferenceKernelDefault("conv_reference", ndRangeDimensions, referenceWorkGroupDimensions,
            {kSizeXId, kSizeYId, matAId, coeffId, matBId}, {matBId}, 0.001f);
    }
};

int main(int argc, char** argv)
{
    unique_ptr<ClTuneConvolution> convolution = ClTuneConvolution::Create<ClTuneConvolution>(argc, argv, "Examples/ClTuneConvolution", "ClTuneConvolution", "ClTuneConvolutionReference");
    convolution->Run();

    return 0;
}
