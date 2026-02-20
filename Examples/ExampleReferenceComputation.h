#pragma once
#include "ExampleBase.h"

class ExampleReferenceComputation : ExampleBase 
{
public:
    using ExampleBase::ExampleBase;

    template <class T>
    static std::shared_ptr<T> Create(int argc, char** argv, int defaultProblemSize, std::string exampleFolderPath, 
            std::string defaultKernelFileBaseName, bool rapidTest = false, bool useProfiling = false)
    {
        std::shared_ptr<T> ex(new T(argc, argv, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName,
                                    rapidTest, useProfiling));
        ex->PostInitialize();
        return ex;
    }

    void PostInitialize() override;

    virtual void InitReference() = 0;
};
