#ifdef KTT_API_CUDA

#include <algorithm>
#include <regex>
#include <vector>
#include <cuda.h>

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/CudaProgram.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/StringUtility.h>

namespace ktt
{

CudaProgram::CudaProgram(const std::string& name, const std::string& source) :
    m_Name(name),
    m_Source(source)
{
    CheckError(nvrtcCreateProgram(&m_Program, source.data(), nullptr, 0, nullptr, nullptr), "nvrtcCreateProgram");
}

CudaProgram::~CudaProgram()
{
    CheckError(nvrtcDestroyProgram(&m_Program), "nvrtcDestroyProgram");
}

void CudaProgram::Build(const std::string& compilerOptions) const
{
    CheckError(nvrtcAddNameExpression(m_Program, m_Name.c_str()), "nvrtcAddNameExpression");
    std::vector<std::string> individualOptions;
    std::vector<const char*> individualOptionsChar;

    if (!compilerOptions.empty())
    {
        std::regex separator(" ");
        std::sregex_token_iterator iterator(compilerOptions.begin(), compilerOptions.end(), separator, -1);
        std::copy(iterator, std::sregex_token_iterator(), std::back_inserter(individualOptions));

        std::transform(individualOptions.begin(), individualOptions.end(), std::back_inserter(individualOptionsChar),
            [](const std::string& sourceString)
        {
            return sourceString.data();
        });
    }

    const nvrtcResult result = nvrtcCompileProgram(m_Program, static_cast<int>(individualOptionsChar.size()),
        individualOptionsChar.data());

    const std::string buildInfo = GetBuildInfo();
    CheckError(result, "nvrtcCompileProgram", buildInfo);
}

const std::string& CudaProgram::GetSource() const
{
    return m_Source;
}

std::string CudaProgram::GetLoweredName() const
{
    const char* loweredName;
    CheckError(nvrtcGetLoweredName(m_Program, m_Name.c_str(), &loweredName), "nvrtcGetLoweredName");
    return std::string(loweredName);
}

nvrtcProgram CudaProgram::GetProgram() const
{
    return m_Program;
}

std::string CudaProgram::GetPtxSource() const
{
    size_t size;
    CheckError(nvrtcGetPTXSize(m_Program, &size), "nvrtcGetPTXSize");

    std::string result(size, '\0');
    CheckError(nvrtcGetPTX(m_Program, result.data()), "nvrtcGetPTX");

    RemoveTrailingZero(result);
    return result;
}

std::string CudaProgram::GetBuildInfo() const
{
    size_t infoSize;
    CheckError(nvrtcGetProgramLogSize(m_Program, &infoSize), "nvrtcGetProgramLogSize");

    std::string infoString(infoSize, '\0');
    CheckError(nvrtcGetProgramLog(m_Program, infoString.data()), "nvrtcGetProgramLog");

    RemoveTrailingZero(infoString);
    return infoString;
}

} // namespace ktt

#endif // KTT_API_CUDA
