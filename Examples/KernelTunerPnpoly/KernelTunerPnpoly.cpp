#include "../ExampleReferenceKernel.h"
#include <memory>

using namespace std;


class KernelTunerPnpoly: public ExampleReferenceKernel {
protected:
    KernelTunerPnpoly(std::shared_ptr<ExampleRefKernelConfiguration> config, int defaultProblemSize,
              string exampleFolderPath, string defaultKernelFileBaseName,
              string defaultReferenceKernelFileBaseName):
        ExampleReferenceKernel(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName,
                defaultReferenceKernelFileBaseName)
    {
        m_dataSize = m_problemSize * 1024 * 1024;
        m_vertSize = 600;
    }

    friend ExampleReferenceKernel;

    uint64_t m_dataSize, m_vertSize;

    vector<int> m_bitmap;
    vector<float> m_points;
    vector<float> m_vertices;

    ktt::ArgumentId m_bitmapId;
    ktt::ArgumentId m_pointsId;
    ktt::ArgumentId m_verticesId;
    ktt::ArgumentId m_dataSizeId;
    ktt::ArgumentId m_vertSizeId;

    void InitData() override
    {
        // Declare data variables
        m_bitmap.resize(m_dataSize, 0);
        m_points.resize(m_dataSize*2, 0.0f);
        m_vertices.resize(m_vertSize*2, 1.0);

        // Populates input data structure by padded data
        FillBuffers<float>({&m_points, &m_vertices}, 0.0f, 1.0f);
    }

    void InitKernel() override
    {
        const ktt::DimensionVector ndRangeDimensions(m_dataSize);
        const ktt::DimensionVector workGroupDimensions;

        m_pointsId = m_tuner->AddArgumentVector(m_points, ktt::ArgumentAccessType::ReadOnly);
        m_verticesId = m_tuner->AddArgumentVector(m_vertices, ktt::ArgumentAccessType::ReadOnly);
        m_bitmapId = m_tuner->AddArgumentVector(m_bitmap, ktt::ArgumentAccessType::WriteOnly);
        m_dataSizeId = m_tuner->AddArgumentScalar(m_dataSize);

        InitKernelDefault("Pnpoly", "Pnpoly", ndRangeDimensions, {m_bitmapId, m_pointsId, m_verticesId, m_dataSizeId});
    }

    void InitReference() override
    {
        const ktt::DimensionVector referenceNdRangeDimensions(m_dataSize/256);
        const ktt::DimensionVector referenceWorkGroupDimensions(256);

        m_vertSizeId = m_tuner->AddArgumentScalar(m_vertSize);

        InitReferenceKernelDefault("PnpolyReference", referenceNdRangeDimensions, referenceWorkGroupDimensions,
                                   {m_bitmapId, m_pointsId, m_verticesId, m_dataSizeId, m_vertSizeId},
                                   {m_bitmapId});
    }

    void InitTuningSpace() override
    {
        // fake tuning parameters, encoding input
        m_tuner->AddParameter(m_kernel, "VERTICES", vector<uint64_t>{m_vertSize});

        // tuning parameters
        m_tuner->AddParameter(m_kernel, "BLOCK_SIZE_X", vector<uint64_t>{32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992});
        m_tuner->AddParameter(m_kernel, "TILE_SIZE", vector<uint64_t>{1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
        m_tuner->AddParameter(m_kernel, "BETWEEN_METHOD", vector<uint64_t>{0, 1, 2, 3});
        m_tuner->AddParameter(m_kernel, "USE_METHOD", vector<uint64_t>{0, 1, 2});

        // Add kernel dimension modifiers based on added tuning parameters
        auto globalModifier = [](const uint64_t size, const vector<uint64_t>& vector)
        {
            return (((size+vector.at(0)-1) / vector.at(0))+vector.at(1)-1) / vector.at(1);
        };

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"BLOCK_SIZE_X", "TILE_SIZE"},
            globalModifier);

        m_tuner->AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE_X", ktt::ModifierAction::Multiply);
    }
};

int main(int argc, char **argv)
{
    unique_ptr<KernelTunerPnpoly> knpnpoly = KernelTunerPnpoly::Create<KernelTunerPnpoly>(
        argc, argv, 20, "Examples/KernelTunerPnpoly", "KernelTunerPnpoly", "KernelTunerPnpolyReference"
    );
    knpnpoly->Run();

    return 0;
}
