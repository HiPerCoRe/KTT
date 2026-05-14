#include "../ExampleReferenceKernel.h"
#include <memory>

using namespace std;


class Transpose: public ExampleReferenceKernel {
protected:
    Transpose(int argc, char** argv, int defaultProblemSize, string exampleFolderPath, string defaultKernelFileBaseName, 
              string defaultReferenceKernelFileBaseName, bool rapidTest, bool useProfiling): 
        ExampleReferenceKernel(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName,
                defaultReferenceKernelFileBaseName, rapidTest, useProfiling)
    {
        m_width = 1024*static_cast<int>(sqrtf(m_problemSize));
        m_height = m_width;
    }

    friend ExampleReferenceKernel;

    int m_width, m_height;

    vector<float> m_dst;
    vector<float> m_src;

    ktt::ArgumentId m_srcId;
    ktt::ArgumentId m_dstId;
    ktt::ArgumentId m_widthId;
    ktt::ArgumentId m_heightId;
    

    void InitData() override 
    {
        // Declare data variables
        m_dst.resize(m_width * m_height);
        m_src.resize(m_width * m_height);

        FillBuffers<float>({&m_src});
    }

    void InitKernel() override
    {
        m_srcId = m_tuner.AddArgumentVector(m_src, ktt::ArgumentAccessType::ReadOnly);
        m_dstId = m_tuner.AddArgumentVector(m_dst, ktt::ArgumentAccessType::WriteOnly);
        m_widthId = m_tuner.AddArgumentScalar(m_width);
        m_heightId = m_tuner.AddArgumentScalar(m_height);
        
        InitKernelDefault("mtran", "Transposition", ktt::DimensionVector(m_width, m_height),
                          {m_dstId, m_srcId, m_widthId, m_heightId});
    }

    void InitReference() override 
    {
        const int tileSize = 16;
        InitReferenceKernelDefault("mtranReference", ktt::DimensionVector(m_width/tileSize, m_height/tileSize), 
                                   ktt::DimensionVector(tileSize, tileSize), {m_dstId, m_srcId, m_widthId, m_heightId}, 
                                   {m_dstId});
    }
    
    void InitTuningSpace() override 
    {
        // Create tuning space
        if (m_computeApi == ktt::ComputeApi::OpenCL)
        {
            m_tuner.AddParameter(m_kernel, "VECTOR_TYPE", vector<uint64_t>{1, 2, 4, 8});
            m_tuner.AddParameter(m_kernel, "PREFETCH", vector<uint64_t>{0, 1, 2});
        }
        else
        {
            m_tuner.AddParameter(m_kernel, "VECTOR_TYPE", vector<uint64_t>{1, 2, 4});
        }

        m_tuner.AddParameter(m_kernel, "CR", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "LOCAL_MEM", vector<uint64_t>{0, 1});
        m_tuner.AddParameter(m_kernel, "PADD_LOCAL", vector<uint64_t>{0, 1});

        const vector<uint64_t> sizeRanges = {1, 2, 4, 8, 16, 32, 64};

        m_tuner.AddParameter(m_kernel, "WORK_GROUP_SIZE_X", sizeRanges);
        m_tuner.AddParameter(m_kernel, "WORK_GROUP_SIZE_Y", sizeRanges);
        m_tuner.AddParameter(m_kernel, "TILE_SIZE_X", sizeRanges);
        m_tuner.AddParameter(m_kernel, "TILE_SIZE_Y", sizeRanges);
        m_tuner.AddParameter(m_kernel, "DIAGONAL_MAP", vector<uint64_t>{0, 1});
        
        // Constrain tuning space
        auto xConstraint = [] (const vector<uint64_t>& v) { return (v[0] == v[1]); };
        auto yConstraint = [] (const vector<uint64_t>& v) { return ((v[1] <= v[0]) && (v[0] % v[1] == 0)); };
        auto tConstraint = [] (const vector<uint64_t>& v) { return (!v[0] || (v[1] <= v[2]*v[3])); };
        auto pConstraint = [] (const vector<uint64_t>& v) { return (v[0] || !v[1]); };
        auto vlConstraint = [] (const vector<uint64_t>& v) { return (!v[0] || v[1] == 1); };

        uint64_t maxMult = 64;
        auto vConstraint = [maxMult] (const vector<uint64_t>& v) { return (v[0]*v[1] <= maxMult); };
        
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
        auto xGlobalModifier = [](const uint64_t size, const vector<uint64_t>& vector) {return size / vector.at(0) / vector.at(1);};
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
            {"TILE_SIZE_X", "VECTOR_TYPE"}, xGlobalModifier);
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "TILE_SIZE_Y",
            ktt::ModifierAction::Divide);

        auto wgSize = [](const vector<uint64_t>& v) {return v[0]*v[1] >= 32;};
        m_tuner.AddConstraint(m_kernel, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, wgSize);
    }
};

int main(int argc, char **argv)
{
    unique_ptr<Transpose> transpose = Transpose::Create<Transpose>(
        argc, argv, 64, "Examples/Transpose", "Transpose", "TransposeReference"
    );
    transpose->Run();
}
