#pragma once
#include "ExampleBase.h"

class ExampleReferenceComputation : public ExampleBase 
{
public:
    using ExampleBase::ExampleBase;

    void PostInitialize() override;

    virtual void InitReference() = 0;
};
