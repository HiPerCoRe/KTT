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

const ArgumentId& OpenClBuffer::GetArgumentId() const
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
    case ArgumentAccessType::Undefined:
        KttError("Arguments with undefined access type cannot be buffers");
        return 0;
    case ArgumentAccessType::ReadOnly:
        return CL_MEM_READ_ONLY;
    case ArgumentAccessType::WriteOnly:
        return CL_MEM_WRITE_ONLY;
    case ArgumentAccessType::ReadWrite:
        return CL_MEM_READ_WRITE;
    default:
        KttError("Unhandled argument access type value");
        return 0;
    }
}

} // namespace ktt

#endif // KTT_API_OPENCL
