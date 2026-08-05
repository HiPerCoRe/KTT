#include "ExampleReferenceKernel.h"
#include <memory>

using namespace std;

ExampleReferenceKernel::ExampleReferenceKernel(
    int argc, char **argv,
    int defaultProblemSize,
    string exampleFolderPath,
    string defaultKernelFileBaseName,
    string defaultRefKernelFileBaseName
) :
    ExampleBase(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName)
{
        m_refKernelFile = GetKernelFilePath(exampleFolderPath, defaultRefKernelFileBaseName);
}

void ExampleReferenceKernel::InitCLI()
{
    ExampleBase::InitCLI();
    m_cli.AddOption({[this](const vector<string> &args) {
        m_refKernelFile = args[0];
    }, "--refKernelPath", "Reference kernel file path (expects string)", "<path>", 1});
}

void ExampleReferenceKernel::PostInitialize()
{
    ExampleBase::PostInitialize();
    if (m_rapidTest) InitReference();
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
    m_refDefinition = m_tuner->AddKernelDefinitionFromFile(refKernelName, m_refKernelFile,
        ndRangeDimensions, workGroupDimensions);
    m_tuner->SetArguments(m_refDefinition, arguments);
    m_refKernel = m_tuner->CreateSimpleKernel(refKernelName, m_refDefinition);

    for (auto arg : outputArguments) {
        m_tuner->SetReferenceKernel(arg, m_refKernel, ktt::KernelConfiguration());
    }
    m_tuner->SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, precision);
}
