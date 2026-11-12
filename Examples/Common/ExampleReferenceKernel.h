#pragma once
#include "ExampleBase.h"

/** @class Base class providing common functionality for Examples using a reference kernel. 
  * Defines its own Create method because of a different constructor signature.
  * See ExampleBase for more information.
  */
class ExampleReferenceKernel : public ExampleBase
{
public:
    /** @fn ExampleBase constructor extended by defaultRefKernelFileBasename. See parent constructor for other params.
      * @param defaultRefKernelFileBaseName Base name for the reference kernel file, without suffix.
      */
    ExampleReferenceKernel(
        int argc, char **argv,
        std::string exampleFolderPath,
        std::string defaultKernelFileBaseName,
        std::string defaultRefKernelFileBaseName);

    /** @fn Factory method for ExampleReferenceKernel. */
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

    /** @fn Abstract method. Intended to implement the reference kernel initialization.
      * Good place to use InitReferenceKernelDefault(...)
      */
    virtual void InitReference() = 0;

    /** @fn Helper method that initializes a simple reference kernel.
      * @param refKernelName Kernel function name that should be called by KTT when running the kernel.
      * @param ndRangeDimensions Global grid size.
      * @param workGroupDimensions Local group size.
      * @param arguments Argument IDs. Must be in the same order as what the kernel function expects.
      * @param outputArguments Which buffer(s) is this kernel intended to be a reference for.
      * @param precision Maximum absolute difference between reference and actual output to pass as valid.
      */
    void InitReferenceKernelDefault(
        const std::string &refKernelName,
        const ktt::DimensionVector &ndRangeDimensions,
        const ktt::DimensionVector &workGroupDimensions,
        const std::vector<ktt::ArgumentId> &arguments,
        const std::vector<ktt::ArgumentId> &outputArguments,
        const float precision = 0.0001
    );
};
