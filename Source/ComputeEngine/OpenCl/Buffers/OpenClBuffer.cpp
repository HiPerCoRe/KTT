#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/Buffers/OpenClBuffer.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

cl_mem_flags OpenClBuffer::GetMemoryFlags()
{
    switch (GetAccessType())
    {
    case ArgumentAccessType::ReadWrite:
        return CL_MEM_READ_WRITE;
    case ArgumentAccessType::Read:
        return CL_MEM_READ_ONLY;
    case ArgumentAccessType::Write:
        return CL_MEM_WRITE_ONLY;
    default:
        KttError("Unhandled argument access type value");
        return 0;
    }
}

} // namespace ktt

#endif // KTT_API_OPENCL
