#include <Ktt.h>
#include "ExampleBase.h"

class ExampleReferenceKernel : public ExampleBase 
{
public:
    ExampleReferenceKernel(
        int argc,
        char** argv, 
        int defaultProblemSize, 
        std::string exampleFolderPath,
        std::string defaultKernelFileBaseName, 
        std::string defaultRefKernelFileBaseName,
        bool rapidTest = false,
        bool useProfiling = false);

    template <class T>
    static std::shared_ptr<T> Create(int argc, char** argv, int defaultProblemSize, std::string exampleFolderPath, 
                                     std::string defaultKernelFileBaseName, std::string defaultRefKernelFileBaseName,
                                     bool rapidTest = false, bool useProfiling = false)
    {
        std::shared_ptr<T> ex(new T(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName,
                                    defaultRefKernelFileBaseName, rapidTest, useProfiling));
        ex->PostInitialize();
        return ex;
    }

protected:
    
    std::string m_refKernelFile;

    ktt::KernelDefinitionId m_refDefinition;
    ktt::KernelId m_refKernel;

    void PostInitialize() override;
    virtual void InitReference() = 0;

    void InitReferenceKernelDefault(
        const std::string &refKernelName,
        const ktt::DimensionVector &ndRangeDimensions,
        const ktt::DimensionVector &workGroupDimensions,
        const std::vector<ktt::ArgumentId> &arguments,
        const std::vector<ktt::ArgumentId> &outputArguments
    );
    void InitReferenceOutputsDefault(const std::vector<ktt::ArgumentId> &outputArguments);
};
