#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../ExampleReferenceComputation.h"

using namespace std;

class Reduction : public ExampleReferenceComputation 
{
protected:
    Reduction(
        int argc,
        char** argv, 
        int defaultProblemSize, 
        string exampleFolderPath,
        string defaultKernelFileBaseName, 
        bool rapidTest = false,
        bool useProfiling = false
    ): ExampleReferenceComputation(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName, rapidTest, useProfiling)
    {
        m_size = m_problemSize * 1024 * 1024;
    }

    friend ExampleReferenceComputation;

    uint32_t m_size;
    vector<float> m_src, m_dest;
    ktt::ArgumentId m_dstId;

    void InitData() override 
    {
        const uint32_t sizeAlloc = ((m_size+16-1)/16)*16; // pad to the longest vector size
        m_src.resize(sizeAlloc);
        m_dest.resize(sizeAlloc);
        FillBuffers({&m_src}, 0, 1000);
    }

    void InitKernels() override 
    {
        const uint32_t nUp = ((m_size+512-1)/512)*512; // maximum WG size used in tuning parameters
        const ktt::ArgumentId srcId = m_tuner.AddArgumentVector(m_src, ktt::ArgumentAccessType::ReadWrite);
        m_dstId = m_tuner.AddArgumentVector(m_dest, ktt::ArgumentAccessType::ReadWrite);
        const ktt::ArgumentId nId = m_tuner.AddArgumentScalar(m_size);
        uint32_t offset = 0;
        const ktt::ArgumentId inOffsetId = m_tuner.AddArgumentScalar(offset);
        const ktt::ArgumentId outOffsetId = m_tuner.AddArgumentScalar(offset);

        InitKernelDefault("reduce", "Reduction", ktt::DimensionVector(nUp), {srcId, m_dstId, nId, inOffsetId, outOffsetId});

        m_tuner.SetLauncher(m_kernel, [this, srcId, nId, inOffsetId, outOffsetId](ktt::ComputeInterface& interface)
        {
            const ktt::DimensionVector& globalSize = interface.GetCurrentGlobalSize(m_definition);
            const ktt::DimensionVector& localSize = interface.GetCurrentLocalSize(m_definition);
            const vector<ktt::ParameterPair>& pairs = interface.GetCurrentConfiguration().GetPairs();
            ktt::DimensionVector myGlobalSize = globalSize;

            // change global size for constant numbers of work-groups
            // this may be done by thread modifier operators as well
            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "UNBOUNDED_WG") == 0)
            {
                myGlobalSize = ktt::DimensionVector(ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "WG_NUM"));
            }

            // execute reduction kernel
            interface.RunKernel(m_definition, myGlobalSize, localSize);

            // execute kernel log n times, when atomics are not used 
            if (ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "USE_ATOMICS") == 0)
            {
                uint32_t n = static_cast<uint32_t>(globalSize.GetSizeX());
                uint32_t inOffset = 0;
                uint32_t outOffset = n;
                uint32_t vectorSize = static_cast<uint32_t>(ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "VECTOR_SIZE"));
                uint32_t wgSize = static_cast<uint32_t>(localSize.GetSizeX());
                size_t iterations = 0; // make sure the end result is in the correct buffer

                while (n > 1 || iterations % 2 == 1)
                {
                    interface.SwapArguments(m_definition, srcId, m_dstId);
                    myGlobalSize.SetSizeX((n + vectorSize - 1) / vectorSize);
                    myGlobalSize.SetSizeX(((myGlobalSize.GetSizeX() - 1) / wgSize + 1));
                    
                    if (myGlobalSize == localSize)
                    {
                        outOffset = 0; // only one WG will be executed
                    }

                    interface.UpdateScalarArgument(nId, &n);
                    interface.UpdateScalarArgument(outOffsetId, &outOffset);
                    interface.UpdateScalarArgument(inOffsetId, &inOffset);

                    interface.RunKernel(m_definition, myGlobalSize, localSize);
                    n = (n + wgSize * vectorSize - 1) / (wgSize * vectorSize);
                    inOffset = outOffset / vectorSize; // input is vectorized, output is scalar
                    outOffset += n;
                    ++iterations;
                }
            }
        });
    }

    void InitTuningParameters() override 
    {
        // get number of compute units
        const ktt::DeviceInfo di = m_tuner.GetCurrentDeviceInfo();
        cout << "Number of compute units: " << di.GetMaxComputeUnits() << endl;
        size_t cus = di.GetMaxComputeUnits();

        m_tuner.AddParameter(m_kernel, "WORK_GROUP_SIZE_X", vector<uint64_t>{32, 64, 128, 256, 512});

        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
            ktt::ModifierAction::Multiply);
        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
            ktt::ModifierAction::Divide);
        m_tuner.AddParameter(m_kernel, "UNBOUNDED_WG", vector<uint64_t>{0, 1});

        m_tuner.AddParameter(m_kernel, "WG_NUM", vector<uint64_t>{0, cus, cus * 2, cus * 4, cus * 8, cus * 16});

        if (m_computeApi == ktt::ComputeApi::OpenCL)
        {
            m_tuner.AddParameter(m_kernel, "VECTOR_SIZE", vector<uint64_t>{1, 2, 4, 8, 16});
        }
        else
        {
            m_tuner.AddParameter(m_kernel, "VECTOR_SIZE", vector<uint64_t>{1, 2, 4});
        }

        m_tuner.AddThreadModifier(m_kernel, {m_definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "VECTOR_SIZE",
            ktt::ModifierAction::Divide);
        m_tuner.AddParameter(m_kernel, "USE_ATOMICS", vector<uint64_t>{0, 1});

        auto persistConstraint = [](const vector<uint64_t>& v) {return (v[0] && v[1] == 0) || (!v[0] && v[1] > 0);};
        m_tuner.AddConstraint(m_kernel, {"UNBOUNDED_WG", "WG_NUM"}, persistConstraint);
        auto persistentAtomic = [](const vector<uint64_t>& v) {return (v[0] == 1) || (v[0] == 0 && v[1] == 1);};
        m_tuner.AddConstraint(m_kernel, {"UNBOUNDED_WG", "USE_ATOMICS"}, persistentAtomic);
        auto unboundedWG = [](const vector<uint64_t>& v) {return (!v[0] || v[1] >= 32);};
        m_tuner.AddConstraint(m_kernel, {"UNBOUNDED_WG", "WORK_GROUP_SIZE_X"}, unboundedWG);
    }

    void InitReference() override 
    {
        m_tuner.SetReferenceComputation(m_dstId, [this](void* buffer)
        {
            float* result = static_cast<float*>(buffer);
            vector<double> resD(m_src.size());
            size_t resSize = m_src.size();

            for (size_t i = 0; i < resSize; ++i)
            {
                resD[i] = static_cast<double>(m_src[i]);
            }

            while (resSize > 1)
            {
                for (size_t i = 0; i < resSize / 2; ++i)
                {
                    resD[i] = resD[i * 2] + resD[i * 2 + 1];
                }

                if (resSize % 2 != 0)
                {
                    resD[resSize / 2 - 1] += resD[resSize - 1];
                }

                resSize = resSize / 2;
            }

            cout << "Reference in double: " << setprecision(10) << resD[0] << endl;
            result[0] = static_cast<float>(resD[0]);
        });

        m_tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, static_cast<double>(m_size) * 10'000.0 / 10'000'000.0);
        m_tuner.SetValidationRange(m_dstId, 1);
    }
};

int main(int argc, char** argv)
{
    unique_ptr<Reduction> reduction = Reduction::Create<Reduction>(argc, argv, 64, "Examples/Reduction", "Reduction");
    reduction->Run();
    
    return 0;
}
