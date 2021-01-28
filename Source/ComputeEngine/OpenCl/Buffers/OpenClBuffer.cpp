#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/Buffers/OpenClBuffer.h>
#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

OpenClBuffer::OpenClBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator, const OpenClContext& context) :
    m_Argument(argument),
    m_Generator(generator),
    m_Context(context.GetContext()),
    m_BufferSize(argument.GetDataSize()),
    m_MemoryFlags(GetMemoryFlags()),
    m_UserOwned(false)
{}

OpenClBuffer::OpenClBuffer(KernelArgument& argument, IdGenerator<TransferActionId>& generator) :
    m_Argument(argument),
    m_Generator(generator),
    m_Context(nullptr),
    m_BufferSize(argument.GetDataSize()),
    m_MemoryFlags(GetMemoryFlags()),
    m_UserOwned(true)
{}

ArgumentId OpenClBuffer::GetArgumentId() const
{
    return m_Argument.GetId();
}

ArgumentAccessType OpenClBuffer::GetAccessType() const
{
    return m_Argument.GetAccessType();
}

ArgumentMemoryLocation OpenClBuffer::GetMemoryLocation() const
{
    return m_Argument.GetMemoryLocation();
}

size_t OpenClBuffer::GetSize() const
{
    return m_BufferSize;
}

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
