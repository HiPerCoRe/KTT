#ifdef KTT_API_CUDA

#include <algorithm>
#include <cstring>
#include <regex>
#include <vector>
#include <cuda.h>

#include <ComputeEngine/Cuda/CudaProgram.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CudaProgram::CudaProgram(const std::string& source) :
    m_Source(source)
{
    CheckError(nvrtcCreateProgram(&m_Program, source.data(), nullptr, 0, nullptr, nullptr), "nvrtcCreateProgram");
}

CudaProgram::~CudaProgram()
{
    CheckError(nvrtcDestroyProgram(&m_Program), "nvrtcDestroyProgram");
}

void CudaProgram::Build() const
{
    std::vector<const char*> individualOptionsChar;

    if (!m_CompilerOptions.empty())
    {
        std::vector<std::string> individualOptions;
        std::regex separator(" ");
        std::sregex_token_iterator iterator(m_CompilerOptions.begin(), m_CompilerOptions.end(), separator, -1);
        std::copy(iterator, std::sregex_token_iterator(), std::back_inserter(individualOptions));

        std::transform(individualOptions.begin(), individualOptions.end(), std::back_inserter(individualOptionsChar),
            [](const std::string& sourceString)
        {
            // Making a copy is necessary, because NVRTC expects the options to be in continuous memory block
            char* result = new char[sourceString.size() + 1];
            std::memcpy(result, sourceString.c_str(), sourceString.size() + 1);
            return result; 
        });
    }

    const nvrtcResult result = nvrtcCompileProgram(m_Program, static_cast<int>(individualOptionsChar.size()),
        individualOptionsChar.data());

    for (size_t i = 0; i < individualOptionsChar.size(); ++i)
    {
        delete[] individualOptionsChar[i];
    }

    const std::string buildInfo = GetBuildInfo();
    CheckError(result, "nvrtcCompileProgram", buildInfo);
}

const std::string& CudaProgram::GetSource() const
{
    return m_Source;
}

nvrtcProgram CudaProgram::GetProgram() const
{
    return m_Program;
}

std::string CudaProgram::GetPtxSource() const
{
    size_t size;
    CheckError(nvrtcGetPTXSize(m_Program, &size), "nvrtcGetPTXSize");

    std::string result(size, ' ');
    CheckError(nvrtcGetPTX(m_Program, result.data()), "nvrtcGetPTX");

    return result;
}

void CudaProgram::SetCompilerOptions(const std::string& options)
{
    m_CompilerOptions = options;
}

std::string CudaProgram::GetBuildInfo() const
{
    size_t infoSize;
    CheckError(nvrtcGetProgramLogSize(m_Program, &infoSize), "nvrtcGetProgramLogSize");

    std::string infoString(infoSize, ' ');
    CheckError(nvrtcGetProgramLog(m_Program, infoString.data()), "nvrtcGetProgramLog");

    return infoString;
}

} // namespace ktt

#endif // KTT_API_CUDA
