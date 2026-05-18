#include "ExampleReferenceKernel.h"
#include <memory>

using namespace std;

ExampleReferenceKernel::ExampleReferenceKernel(
    shared_ptr<ExampleRefKernelConfiguration> config,
    int defaultProblemSize,
    string exampleFolderPath,
    string defaultKernelFileBaseName,
    string defaultRefKernelFileBaseName
) :
    ExampleBase(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName)
{
    if (config->refKernelFile.empty()) {
        m_refKernelFile = GetKernelFilePath(exampleFolderPath, defaultRefKernelFileBaseName);
    } else {
        m_refKernelFile = config->refKernelFile;
    }
}

void ExampleReferenceKernel::PostInitialize()
{
    ExampleBase::PostInitialize();
    if (!m_config->rapidTest) InitReference();
}

void ExampleReferenceKernel::InitReferenceKernelDefault(
        const string &refKernelName,
        const ktt::DimensionVector &ndRangeDimensions,
        const ktt::DimensionVector &workGroupDimensions,
        const vector<ktt::ArgumentId> &arguments,
        const vector<ktt::ArgumentId> &outputArguments,
        const float precision
    )
{
    m_refDefinition = m_tuner.AddKernelDefinitionFromFile(refKernelName, m_refKernelFile,
        ndRangeDimensions, workGroupDimensions);
    m_tuner.SetArguments(m_refDefinition, arguments);
    m_refKernel = m_tuner.CreateSimpleKernel(refKernelName, m_refDefinition);

    for (auto arg : outputArguments) {
        m_tuner.SetReferenceKernel(arg, m_refKernel, ktt::KernelConfiguration());
    }
    m_tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, precision);
}
