#ifdef KTT_API_OPENCL

#include <cstddef>
#include <vector>

#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <ComputeEngine/OpenCl/OpenClProgram.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>

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

void OpenClProgram::Build() const
{
    std::vector<cl_device_id> devices{m_Device};
    cl_int result = clBuildProgram(m_Program, static_cast<cl_uint>(devices.size()), devices.data(), m_CompilerOptions.data(),
        nullptr, nullptr);

    const std::string buildInfo = GetBuildLog();
    CheckError(result, "clBuildProgram", buildInfo);
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

void OpenClProgram::SetCompilerOptions(const std::string& options)
{
    m_CompilerOptions = options;
}

std::string OpenClProgram::GetBuildLog() const
{
    size_t infoSize;
    CheckError(clGetProgramBuildInfo(m_Program, m_Device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &infoSize),
        "clGetProgramBuildInfo");
    
    std::string infoString(infoSize, ' ');
    CheckError(clGetProgramBuildInfo(m_Program, m_Device, CL_PROGRAM_BUILD_LOG, infoSize, infoString.data(), nullptr),
        "clGetProgramBuildInfo");
    return infoString;
}

} // namespace ktt

#endif // KTT_API_OPENCL
