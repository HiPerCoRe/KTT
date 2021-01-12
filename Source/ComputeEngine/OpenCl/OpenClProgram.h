#pragma once

#ifdef KTT_API_OPENCL

#include <string>
#include <vector>
#include <CL/cl.h>

#include <ComputeEngine/OpenCl/OpenClContext.h>

namespace ktt
{

class OpenClProgram
{
public:
    explicit OpenClProgram(const OpenClContext& context, const std::string& source);
    ~OpenClProgram();

    void Build(const std::string& compilerOptions);

    const std::string& GetSource() const;
    cl_program GetProgram() const;
    cl_device_id GetDevice() const;

private:
    std::string m_Source;
    cl_program m_Program;
    cl_device_id m_Device;
    
    std::string GetBuildLog() const;
};

} // namespace ktt

#endif // KTT_API_OPENCL
