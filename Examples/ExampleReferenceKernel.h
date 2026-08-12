#pragma once
#include "ExampleBase.h"

class ExampleReferenceKernel : public ExampleBase
{
public:
    ExampleReferenceKernel(
        int argc, char **argv,
        std::string exampleFolderPath,
        std::string defaultKernelFileBaseName,
        std::string defaultRefKernelFileBaseName);

    template <class T>
    static std::unique_ptr<T> Create(int argc, char** argv, std::string exampleFolderPath,
                                     std::string defaultKernelFileBaseName, std::string defaultRefKernelFileBaseName)
    {
        std::unique_ptr<T> ex(new T(argc, argv, exampleFolderPath,
                                    defaultKernelFileBaseName, defaultRefKernelFileBaseName));
        ex->PostInitialize();
        return ex;
    }

protected:
    
    std::string m_refKernelFile;

    ktt::KernelDefinitionId m_refDefinition;
    ktt::KernelId m_refKernel;

    void InitCLI() override;
    void PostInitialize() override;
    virtual void InitReference() = 0;

    void InitReferenceKernelDefault(
        const std::string &refKernelName,
        const ktt::DimensionVector &ndRangeDimensions,
        const ktt::DimensionVector &workGroupDimensions,
        const std::vector<ktt::ArgumentId> &arguments,
        const std::vector<ktt::ArgumentId> &outputArguments,
        const float precision = 0.0001
    );
};
