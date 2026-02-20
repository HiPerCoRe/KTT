#include "ExampleReferenceKernel.h"

using namespace std;

ExampleReferenceKernel::ExampleReferenceKernel(int argc, char** argv, 
                 int defaultProblemSize, 
                 string exampleFolderPath,
                 string defaultKernelFileBaseName, 
                 string defaultRefKernelFileBaseName,
                 bool rapidTest,
                 bool useProfiling): ExampleBase(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName, rapidTest, useProfiling)
{
    m_refKernelFile = GetKernelFilePath(exampleFolderPath, defaultRefKernelFileBaseName);
    if (argc >= 6)
    {
        m_refKernelFile = string(argv[5]);
    }
}

void ExampleReferenceKernel::PostInitialize()
{
    ExampleBase::PostInitialize();
    InitReference();
}

void ExampleReferenceKernel::InitReferenceKernelDefault(
        const string &refKernelName,
        const ktt::DimensionVector &ndRangeDimensions,
        const ktt::DimensionVector &workGroupDimensions,
        const vector<ktt::ArgumentId> &arguments,
        const vector<ktt::ArgumentId> &outputArguments
    )
{
    m_refDefinition = m_tuner.AddKernelDefinitionFromFile(refKernelName, m_refKernelFile,
        ndRangeDimensions, workGroupDimensions);
    m_tuner.SetArguments(m_refDefinition, arguments);
    m_refKernel = m_tuner.CreateSimpleKernel(refKernelName, m_refDefinition);

    if (!m_rapidTest)
    {
        for (auto arg : outputArguments) {
            m_tuner.SetReferenceKernel(arg, m_refKernel, ktt::KernelConfiguration());
        }
        m_tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.0001);
    }
}
