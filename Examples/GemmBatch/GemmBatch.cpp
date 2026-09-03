#include <iostream>
#include <string>
#include <vector>
#include "../ExampleReferenceComputation.h"

using namespace std;

class GemmBatch : public ExampleReferenceComputation {
protected:
    GemmBatch(int argc, char **argv, int defaultProblemSize,
              std::string exampleFolderPath, std::string defaultKernelFileBaseName) :
        ExampleReferenceComputation(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName)
    {
        m_batch = m_problemSize * 16 * 1024;
    }

    friend ExampleBase;

    // Declare and initialize data (m, n > 1)
    const unsigned a = 4;
    const unsigned b = 4;
    const unsigned c = 4;
    int m_batch;

    std::vector<float> m_srcA;
    std::vector<float> m_srcB;
    std::vector<float> m_dst;

    ktt::ArgumentId m_srcAId;
    ktt::ArgumentId m_srcBId;
    ktt::ArgumentId m_dstId;
    ktt::ArgumentId m_nId;

    void InitData() override
    {
        std::cout << "Computing C = AB using " << m_batch << " matrices of sizes"
            << std::endl
            << "A: " << a << " x " << b << std::endl
            << "B: " << c << " x " << a << std::endl
            << "C: " << c << " x " << b << std::endl;

        m_srcA.resize(a * b * m_batch);
        m_srcB.resize(c * a * m_batch);
        m_dst.resize(c * b * m_batch);

        FillBuffers({&m_srcA, &m_srcB}, 0.f, 10.f);
    }

    void InitKernel() override 
    {
        ktt::DimensionVector ndRangeDimensions(m_batch);
        ktt::DimensionVector workGroupDimensions;

        // create input/output
        m_srcAId = m_tuner->AddArgumentVector(m_srcA, ktt::ArgumentAccessType::ReadOnly);
        m_srcBId = m_tuner->AddArgumentVector(m_srcB, ktt::ArgumentAccessType::ReadOnly);
        m_dstId = m_tuner->AddArgumentVector(m_dst, ktt::ArgumentAccessType::WriteOnly);
        m_nId = m_tuner->AddArgumentScalar(m_batch);

        InitKernelDefault("gemm_batch", "GemmBatch", ndRangeDimensions, {m_srcAId, m_srcBId, m_dstId, m_nId});

        m_tuner->SetLauncher(m_kernel, [this](ktt::ComputeInterface& interface) {
            const std::vector<ktt::ParameterPair>& pairs = interface.GetCurrentConfiguration().GetPairs();

            size_t padd_c = ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "PADD_C");
            size_t y = ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "GROUP_SIZE_Y");
            size_t z = ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "GROUP_SIZE_Z");

            ktt::DimensionVector myGlobalSize(1, 1, 1);
            ktt::DimensionVector myLocalSize(1, 1, 1);

            // if (m_computeApi == ktt::ComputeApi::OpenCL)
            // {
            //     myGlobalSize.SetSizeX(m_batch * (c + padd_c) / z);
            //     myGlobalSize.SetSizeY(y);
            //     myGlobalSize.SetSizeZ(z);
            // }
            // else
            // {
                myGlobalSize.SetSizeX(m_batch / z);
            // }

            myLocalSize.SetSizeX(c + padd_c);
            myLocalSize.SetSizeY(y);
            myLocalSize.SetSizeZ(z);

            interface.RunKernel(m_definition, myGlobalSize, myLocalSize);
        });
    }

    void InitTuningSpace() override 
    {
        m_tuner->AddParameter(m_kernel, "SIZE_A", vector<uint64_t>{(size_t)a});
        m_tuner->AddParameter(m_kernel, "SIZE_B", vector<uint64_t>{(size_t)b});
        m_tuner->AddParameter(m_kernel, "SIZE_C", vector<uint64_t>{(size_t)c});
        m_tuner->AddParameter(m_kernel, "GROUP_SIZE_Y", vector<uint64_t>{1, 2, 4, 8, 16, 32});
        m_tuner->AddParameter(m_kernel, "GROUP_SIZE_Z", vector<uint64_t>{1, 2, 4, 8, 16, 32, 64});
        m_tuner->AddParameter(m_kernel, "CACHING_STRATEGY", vector<uint64_t>{0, 1, 2}); /* 0 = implicit caching, 1 = local memory, 2 = private memory */
        m_tuner->AddParameter(m_kernel, "PADD_AA", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "PADD_AB", vector<uint64_t>{0, 1});
        if (c % 4 == 0)
            m_tuner->AddParameter(m_kernel, "PADD_C", vector<uint64_t>{0});
        else
            m_tuner->AddParameter(m_kernel, "PADD_C", vector<uint64_t>{0, c % 4});
        m_tuner->AddParameter(m_kernel, "DIRECT_WRITE", vector<uint64_t>{0, 1});
        m_tuner->AddParameter(m_kernel, "UNROLL_K", vector<uint64_t>{0, 1});

        auto parallelismConstraint = [](const std::vector<size_t>& v) {return v[0] <= v[1];};
        m_tuner->AddConstraint(m_kernel, {"GROUP_SIZE_Y", "SIZE_B"}, parallelismConstraint);
        auto paddConstraint = [](const std::vector<size_t>& v) {return (v[0] == 0 && v[1] == 0 && v[2] == 0) || (v[3] > 0);};
        m_tuner->AddConstraint(m_kernel, {"PADD_AA", "PADD_AB", "PADD_C", "CACHING_STRATEGY"}, paddConstraint);
        auto dwConstraint = [](const std::vector<size_t>& v) {return (v[0] == 1) || (v[1] > 0);};
        m_tuner->AddConstraint(m_kernel, {"DIRECT_WRITE", "CACHING_STRATEGY"}, dwConstraint);
        auto unrollkConstraint = [](const std::vector<size_t>& v) {return (v[0] == 0) || (v[1] == 2);};
        m_tuner->AddConstraint(m_kernel, {"UNROLL_K", "CACHING_STRATEGY"}, unrollkConstraint);
    #define SHARED_PER_BLOCK (49152/4)
        auto memConstraint = [](const std::vector<size_t>& v) {size_t a = v[1]; size_t b = v[2]; size_t c = v[3]; return (v[0] == 1 && ((a+v[7])*(b+v[8])+c*a+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK) || (v[0] == 2 && v[5] == 1 && ((a+v[7])*(b+v[8])+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK) || (v[0] == 2 && ((a+v[7])*(b+v[8])+c*a+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK);};
        m_tuner->AddConstraint(m_kernel, {"CACHING_STRATEGY", "SIZE_A", "SIZE_B", "SIZE_C", "DIRECT_WRITE", "GROUP_SIZE_Y", "GROUP_SIZE_Z", "PADD_AA", "PADD_AB"}, memConstraint);
    #define MAX_BLOCK_SIZE 1024
        auto blockConstraint = [](const std::vector<size_t>&v) {return ((v[0]+v[2])*v[1]*v[3] < MAX_BLOCK_SIZE) && ((v[0]+v[2])*v[1]*v[3] >= 32);};
        m_tuner->AddConstraint(m_kernel, {"SIZE_C", "GROUP_SIZE_Y", "PADD_C", "GROUP_SIZE_Z"}, blockConstraint);
    }

    void InitReference() override
    {
        m_tuner->SetReferenceComputation(m_dstId, [this](void* buffer) {
            std::vector<float> res(m_batch * c * b);

            for (int i = 0; i < m_batch; i++) {
                for (unsigned j = 0; j < c; j++) {
                    for (unsigned k = 0; k < b; k++) {
                        float tmp = 0.0f;
                        for (unsigned l = 0; l < a; l++) {
                            tmp += m_srcA[i*a*b + k*a + l] * m_srcB[i*c*a + l*c + j];
                        }
                        res[i*c*b + k*c + j] = tmp;
                    }
                }
            }

            std::memcpy(buffer, res.data(), res.size() * sizeof(float));
        });
    }
};


int main(int argc, char **argv)
{
    auto gemmBatch = GemmBatch::Create<GemmBatch>(argc, argv, 4, "Examples/GemmBatch", "Gemm");
    gemmBatch->Run();
}
