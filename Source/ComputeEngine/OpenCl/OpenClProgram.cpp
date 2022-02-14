#ifdef KTT_API_OPENCL

#include <cstddef>
#include <vector>

#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <ComputeEngine/OpenCl/OpenClProgram.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StringUtility.h>

namespace ktt
{

OpenClProgram::OpenClProgram(const OpenClContext& context, const std::string& source) :
    m_Source(source),
    m_Device(context.GetDevice())
{
    const size_t sourceLength = source.size();
    const char* sourcePointer = m_Source.data();
    cl_int result;
    m_Program = clCreateProgramWithSource(context.GetContext(), 1, &sourcePointer, &sourceLength, &result);
    CheckError(result, "clCreateProgramWithSource");
}

OpenClProgram::~OpenClProgram()
{
    CheckError(clReleaseProgram(m_Program), "clReleaseProgram");
}

void OpenClProgram::Build(const std::string& compilerOptions) const
{
    const std::string logCompilerOptions = compilerOptions.empty() ? "empty" : compilerOptions;
    Logger::LogDebug("Building OpenCL program with options: " + logCompilerOptions);

    std::vector<cl_device_id> devices{m_Device};
    cl_int result = clBuildProgram(m_Program, static_cast<cl_uint>(devices.size()), devices.data(), compilerOptions.data(),
        nullptr, nullptr);

    const std::string buildInfo = GetBuildLog();
    CheckError(result, "clBuildProgram", buildInfo, ExceptionReason::CompilerError);
}

const std::string& OpenClProgram::GetSource() const
{
    return m_Source;
}

cl_program OpenClProgram::GetProgram() const
{
    return m_Program;
}

cl_device_id OpenClProgram::GetDevice() const
{
    return m_Device;
}

std::string OpenClProgram::GetBuildLog() const
{
    size_t infoSize;
    CheckError(clGetProgramBuildInfo(m_Program, m_Device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &infoSize),
        "clGetProgramBuildInfo");
    
    std::string infoString(infoSize, '\0');
    CheckError(clGetProgramBuildInfo(m_Program, m_Device, CL_PROGRAM_BUILD_LOG, infoSize, infoString.data(), nullptr),
        "clGetProgramBuildInfo");

    RemoveTrailingZero(infoString);
    return infoString;
}

} // namespace ktt

#endif // KTT_API_OPENCL
