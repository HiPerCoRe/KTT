#include "ExampleReferenceComputation.h"

void ExampleReferenceComputation::PostInitialize() 
{
    ExampleBase::PostInitialize();
    if (!m_config->rapidTest) InitReference();
}
