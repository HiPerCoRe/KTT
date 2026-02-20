#ifdef KTT_API_CUDA

#include <algorithm>
#include <regex>
#include <vector>
#include <cuda.h>

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/CudaKernel.h>
#include <ComputeEngine/Cuda/CudaProgram.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <KernelArgument/KernelArgument.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StringUtility.h>

//#define KTT_EXPERIMENTAL_SLEEP_AFTER_COMPILATION

#ifdef KTT_EXPERIMENTAL_SLEEP_AFTER_COMPILATION
#include <chrono>
#include <thread>
#include <random>
#endif

namespace ktt
{

CudaProgram::CudaProgram(const std::string& name, const std::string& source, const std::vector<KernelArgument*>& symbolArguments) :
    m_Name(name),
    m_Source(source),
    m_SymbolArguments(symbolArguments)
{
    CheckError(nvrtcCreateProgram(&m_Program, source.data(), nullptr, 0, nullptr, nullptr), "nvrtcCreateProgram");
}

CudaProgram::~CudaProgram()
{
    CheckError(nvrtcDestroyProgram(&m_Program), "nvrtcDestroyProgram");
}

void CudaProgram::Build(const std::string& compilerOptions) const
{
    const std::string logCompilerOptions = compilerOptions.empty() ? "empty" : compilerOptions;
    Logger::LogDebug("Building CUDA program with name " + m_Name + " and options: " + logCompilerOptions);

    CheckError(nvrtcAddNameExpression(m_Program, m_Name.c_str()), "nvrtcAddNameExpression");
    
    for (const auto* argument : m_SymbolArguments)
    {
        CheckError(nvrtcAddNameExpression(m_Program, argument->GetSymbolName().c_str()), "nvrtcAddNameExpression");
    }
    
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
    CheckError(result, "nvrtcCompileProgram", buildInfo, ExceptionReason::CompilerError);

#ifdef KTT_EXPERIMENTAL_SLEEP_AFTER_COMPILATION
    // Random sleep after compilation
    #ifndef KTT_SLEEP_MIN_MS
    #define KTT_SLEEP_MIN_MS 0
    #endif
    #ifndef KTT_SLEEP_MAX_MS
    #define KTT_SLEEP_MAX_MS 1000
    #endif
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(KTT_SLEEP_MIN_MS, KTT_SLEEP_MAX_MS);
    int sleepMs = distribution(generator);
    Logger::LogDebug("Sleeping for " + std::to_string(sleepMs) + " ms after compilation of CUDA program " + m_Name);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
#endif
}

void CudaProgram::InitializeSymbolData(const CudaKernel& kernel) const
{
    for (const auto* argument : m_SymbolArguments)
    {
        const char* symbolName;
        CheckError(nvrtcGetLoweredName(m_Program, argument->GetSymbolName().c_str(), &symbolName), "nvrtcGetLoweredName");
        
        CUdeviceptr symbolAddress;
        CheckError(cuModuleGetGlobal(&symbolAddress, nullptr, kernel.GetModule(), symbolName), "cuModuleGetGlobal");
        CheckError(cuMemcpyHtoD(symbolAddress, argument->GetData(), argument->GetDataSize()), "cuMemcpyHtoD");
    }
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
