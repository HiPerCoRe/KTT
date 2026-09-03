#include "ExampleReferenceComputation.h"

void ExampleReferenceComputation::PostInitialize() 
{
    ExampleBase::PostInitialize();
    if (!m_rapidTest) InitReference();
}
