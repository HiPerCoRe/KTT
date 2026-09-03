#pragma once
#include "ExampleBase.h"

/** @class Base class providing common functionality for Examples that use a reference computation.
  * See ExampleBase for more information.
  */
class ExampleReferenceComputation : public ExampleBase 
{
public:
    using ExampleBase::ExampleBase;

    void PostInitialize() override;

    /** @fn Abstract method. Intended to implement the reference computation initialization. */
    virtual void InitReference() = 0;
};
