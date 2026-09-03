#include "ComputeEngine.h"

using namespace ktt;
using namespace std;

ComputeEngine::ComputeEngine(GlobalSizeType globalSizeType) :
    m_Configuration(globalSizeType)
{}

string ComputeEngine::GetCompilerOptions()
{
    return m_Configuration.GetStaticCompilerOptions();
}

void ComputeEngine::AddCompilerOptions(const std::string& options)
{
    std::string allOptions = GetCompilerOptions();
    allOptions += " ";
    allOptions += options;
    SetCompilerOptions(allOptions, true);
}