#include "cuda_core.h"
#include "utility/timer.h"

namespace ktt
{

#ifdef PLATFORM_CUDA

CudaCore::CudaCore(const size_t deviceIndex) :
    deviceIndex(deviceIndex),
    compilerOptions(std::string(""))
{
    checkCudaError(cuInit(0), "cuInit");

    auto devices = getCudaDevices();
    if (deviceIndex >= devices.size())
    {
        throw std::runtime_error(std::string("Invalid device index: ") + std::to_string(deviceIndex));
    }

    context = std::make_unique<CudaContext>(devices.at(deviceIndex).getDevice());
    stream = std::make_unique<CudaStream>(context->getContext(), devices.at(deviceIndex).getDevice());
}

void CudaCore::printComputeApiInfo(std::ostream& outputTarget) const
{
    outputTarget << "Platform 0: " << "NVIDIA CUDA" << std::endl;
    auto devices = getCudaDevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        outputTarget << "Device " << i << ": " << devices.at(i).getName() << std::endl;
    }
    outputTarget << std::endl;
}

std::vector<PlatformInfo> CudaCore::getPlatformInfo() const
{
    int driverVersion;
    checkCudaError(cuDriverGetVersion(&driverVersion), "cuDriverGetVersion");

    PlatformInfo cuda(0, "NVIDIA CUDA");
    cuda.setVendor("NVIDIA Corporation");
    cuda.setVersion(std::to_string(driverVersion));
    cuda.setExtensions("");
    return std::vector<PlatformInfo>{ cuda };
}

std::vector<DeviceInfo> CudaCore::getDeviceInfo(const size_t) const
{
    std::vector<DeviceInfo> result;
    auto devices = getCudaDevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        result.push_back(getCudaDeviceInfo(i));
    }

    return result;
}

DeviceInfo CudaCore::getCurrentDeviceInfo() const
{
    return getCudaDeviceInfo(deviceIndex);
}

void CudaCore::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void CudaCore::uploadArgument(const KernelArgument& kernelArgument)
{
    if (kernelArgument.getArgumentUploadType() != ArgumentUploadType::Vector)
    {
        return;
    }
    
    clearBuffer(kernelArgument.getId());

    std::unique_ptr<CudaBuffer> buffer = createBuffer(kernelArgument);
    buffer->uploadData(kernelArgument.getData(), kernelArgument.getDataSizeInBytes());
    buffers.insert(std::move(buffer)); // buffer data will be stolen
}

void CudaCore::updateArgument(const size_t argumentId, const void* data, const size_t dataSizeInBytes)
{
    for (const auto& buffer : buffers)
    {
        if (buffer->getKernelArgumentId() == argumentId)
        {
            buffer->uploadData(data, dataSizeInBytes);
            return;
        }
    }
    throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(argumentId));
}

KernelArgument CudaCore::downloadArgument(const size_t argumentId) const
{
    for (const auto& buffer : buffers)
    {
        if (buffer->getKernelArgumentId() != argumentId)
        {
            continue;
        }

        KernelArgument argument(buffer->getKernelArgumentId(), buffer->getBufferSize() / buffer->getElementSize(), buffer->getDataType(),
            buffer->getMemoryType(), ArgumentUploadType::Vector);
        buffer->downloadData(argument.getData(), argument.getDataSizeInBytes());
        return argument;
    }

    throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(argumentId));
}

void CudaCore::clearBuffer(const size_t argumentId)
{
    auto iterator = buffers.cbegin();

    while (iterator != buffers.cend())
    {
        if (iterator->get()->getKernelArgumentId() == argumentId)
        {
            buffers.erase(iterator);
            return;
        }
        else
        {
            ++iterator;
        }
    }
}

void CudaCore::clearBuffers()
{
    buffers.clear();
}

void CudaCore::clearBuffers(const ArgumentMemoryType& argumentMemoryType)
{
    auto iterator = buffers.cbegin();

    while (iterator != buffers.cend())
    {
        if (iterator->get()->getMemoryType() == argumentMemoryType)
        {
            iterator = buffers.erase(iterator);
        }
        else
        {
            ++iterator;
        }
    }
}

KernelRunResult CudaCore::runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
    const std::vector<size_t>& localSize, const std::vector<const KernelArgument*>& argumentPointers)
{
    std::unique_ptr<CudaProgram> program = createAndBuildProgram(source);
    std::unique_ptr<CudaKernel> kernel = createKernel(*program, kernelName);
    std::vector<CUdeviceptr*> kernelArguments = getKernelArguments(argumentPointers);

    Timer timer;
    timer.start();
    float duration = enqueueKernel(*kernel, globalSize, localSize, kernelArguments, getSharedMemorySizeInBytes(argumentPointers));

    timer.stop();
    uint64_t overhead = timer.getElapsedTime();
    return KernelRunResult(static_cast<uint64_t>(duration), overhead);
}

std::unique_ptr<CudaProgram> CudaCore::createAndBuildProgram(const std::string& source) const
{
    auto program = std::make_unique<CudaProgram>(source);
    program->build(compilerOptions);
    return program;
}

std::unique_ptr<CudaBuffer> CudaCore::createBuffer(const KernelArgument& argument) const
{
    auto buffer = std::make_unique<CudaBuffer>(argument.getId(), argument.getDataSizeInBytes(), argument.getElementSizeInBytes(),
        argument.getArgumentDataType(), argument.getArgumentMemoryType());
    return buffer;
}

std::unique_ptr<CudaEvent> CudaCore::createEvent() const
{
    auto event = std::make_unique<CudaEvent>();
    return event;
}

std::unique_ptr<CudaKernel> CudaCore::createKernel(const CudaProgram& program, const std::string& kernelName) const
{
    auto kernel = std::make_unique<CudaKernel>(program.getPtxSource(), kernelName);
    return kernel;
}

float CudaCore::enqueueKernel(CudaKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
    const std::vector<CUdeviceptr*>& kernelArguments, const size_t localMemorySize) const
{
    auto start = createEvent();
    auto end = createEvent();

    checkCudaError(cuEventRecord(start->getEvent(), stream->getStream()), std::string("cuEventRecord"));
    checkCudaError(cuLaunchKernel(kernel.getKernel(), static_cast<unsigned int>(globalSize.at(0)), static_cast<unsigned int>(globalSize.at(1)),
        static_cast<unsigned int>(globalSize.at(2)), static_cast<unsigned int>(localSize.at(0)), static_cast<unsigned int>(localSize.at(1)),
        static_cast<unsigned int>(localSize.at(2)), static_cast<unsigned int>(localMemorySize), stream->getStream(), (void**)kernelArguments.data(),
        nullptr), std::string("cuLaunchKernel"));
    checkCudaError(cuEventRecord(end->getEvent(), stream->getStream()), std::string("cuEventRecord"));

    // Wait for computation to finish
    checkCudaError(cuEventSynchronize(end->getEvent()), std::string("cuEventSynchronize"));

    return getKernelRunDuration(start->getEvent(), end->getEvent());
}

std::vector<CudaDevice> CudaCore::getCudaDevices() const
{
    int deviceCount;
    checkCudaError(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");

    std::vector<CUdevice> deviceIds(deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        checkCudaError(cuDeviceGet(&deviceIds.at(i), i), "cuDeviceGet");
    }

    std::vector<CudaDevice> devices;
    for (const auto deviceId : deviceIds)
    {
        std::string name(40, ' ');
        checkCudaError(cuDeviceGetName(&name[0], 40, deviceId), "cuDeviceGetName");
        devices.push_back(CudaDevice(deviceId, name));
    }

    return devices;
}

DeviceInfo CudaCore::getCudaDeviceInfo(const size_t deviceIndex) const
{
    auto devices = getCudaDevices();
    DeviceInfo result(deviceIndex, devices.at(deviceIndex).getName());

    CUdevice id = devices.at(deviceIndex).getDevice();
    result.setExtensions("");
    result.setVendor("NVIDIA Corporation");
    
    size_t globalMemory;
    checkCudaError(cuDeviceTotalMem(&globalMemory, id), "cuDeviceTotalMem");
    result.setGlobalMemorySize(globalMemory);

    int localMemory;
    checkCudaError(cuDeviceGetAttribute(&localMemory, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, id), "cuDeviceGetAttribute");
    result.setLocalMemorySize(localMemory);

    int constantMemory;
    checkCudaError(cuDeviceGetAttribute(&constantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, id), "cuDeviceGetAttribute");
    result.setMaxConstantBufferSize(constantMemory);

    int computeUnits;
    checkCudaError(cuDeviceGetAttribute(&computeUnits, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, id), "cuDeviceGetAttribute");
    result.setMaxComputeUnits(computeUnits);

    int workGroupSize;
    checkCudaError(cuDeviceGetAttribute(&workGroupSize, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, id), "cuDeviceGetAttribute");
    result.setMaxWorkGroupSize(workGroupSize);
    result.setDeviceType(DeviceType::GPU);

    return result;
}

std::vector<CUdeviceptr*> CudaCore::getKernelArguments(const std::vector<const KernelArgument*>& argumentPointers)
{
    std::vector<CUdeviceptr*> result;

    for (const auto argument : argumentPointers)
    {
        if (argument->getArgumentUploadType() == ArgumentUploadType::Local)
        {
            continue;
        }
        else if (argument->getArgumentUploadType() == ArgumentUploadType::Vector)
        {
            CUdeviceptr* cachedBuffer = loadBufferFromCache(argument->getId());
            if (cachedBuffer == nullptr)
            {
                uploadArgument(*argument);
                cachedBuffer = loadBufferFromCache(argument->getId());
            }

            result.push_back(cachedBuffer);
        }
        else if (argument->getArgumentUploadType() == ArgumentUploadType::Scalar)
        {
            result.push_back((CUdeviceptr*)argument->getData());
        }
    }

    return result;
}

size_t CudaCore::getSharedMemorySizeInBytes(const std::vector<const KernelArgument*>& argumentPointers) const
{
    size_t result = 0;

    for (const auto argument : argumentPointers)
    {
        if (argument->getArgumentUploadType() == ArgumentUploadType::Local)
        {
            result += argument->getDataSizeInBytes();
        }
    }

    return result;
}

CUdeviceptr* CudaCore::loadBufferFromCache(const size_t argumentId) const
{
    for (const auto& buffer : buffers)
    {
        if (buffer->getKernelArgumentId() == argumentId)
        {
            return buffer->getBuffer();
        }
    }
    return nullptr;
}

#else

CudaCore::CudaCore(const size_t)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::printComputeApiInfo(std::ostream&) const
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

std::vector<PlatformInfo> CudaCore::getPlatformInfo() const
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

std::vector<DeviceInfo> CudaCore::getDeviceInfo(const size_t) const
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

DeviceInfo CudaCore::getCurrentDeviceInfo() const
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::setCompilerOptions(const std::string&)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::uploadArgument(const KernelArgument&)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::updateArgument(const size_t, const void*, const size_t)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

KernelArgument CudaCore::downloadArgument(const size_t) const
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::clearBuffer(const size_t)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::clearBuffers()
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::clearBuffers(const ArgumentMemoryType&)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

KernelRunResult CudaCore::runKernel(const std::string&, const std::string&, const std::vector<size_t>&, const std::vector<size_t>&,
    const std::vector<const KernelArgument*>&)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

#endif // PLATFORM_CUDA

} // namespace ktt
