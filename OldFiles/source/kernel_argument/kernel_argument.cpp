#include <stdexcept>
#include <kernel_argument/kernel_argument.h>

namespace ktt
{

KernelArgument::KernelArgument(const ArgumentId id, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType) :
    id(id),
    numberOfElements(numberOfElements),
    elementSizeInBytes(elementSizeInBytes),
    argumentDataType(dataType),
    argumentMemoryLocation(memoryLocation),
    argumentAccessType(accessType),
    argumentUploadType(uploadType),
    referencedData(nullptr),
    dataOwned(true),
    persistentFlag(false),
    userBuffer(false)
{
    if (numberOfElements == 0)
    {
        throw std::runtime_error("Size of kernel argument must be greater than zero");
    }
    prepareData();
}

KernelArgument::KernelArgument(const ArgumentId id, void* data, const size_t numberOfElements, const size_t elementSizeInBytes,
    const ArgumentDataType dataType, const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType,
    const ArgumentUploadType uploadType, const bool dataOwned) :
    id(id),
    numberOfElements(numberOfElements),
    elementSizeInBytes(elementSizeInBytes),
    argumentDataType(dataType),
    argumentMemoryLocation(memoryLocation),
    argumentAccessType(accessType),
    argumentUploadType(uploadType),
    referencedData(nullptr),
    dataOwned(dataOwned),
    persistentFlag(false),
    userBuffer(false)
{
    if (numberOfElements == 0)
    {
        throw std::runtime_error("Size of kernel argument must be greater than zero");
    }

    if (dataOwned)
    {
        initializeData(data);
    }
    else
    {
        referencedData = data;
    }
}

KernelArgument::KernelArgument(const ArgumentId id, const void* data, const size_t numberOfElements, const size_t elementSizeInBytes,
    const ArgumentDataType dataType, const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType,
    const ArgumentUploadType uploadType) :
    id(id),
    numberOfElements(numberOfElements),
    elementSizeInBytes(elementSizeInBytes),
    argumentDataType(dataType),
    argumentMemoryLocation(memoryLocation),
    argumentAccessType(accessType),
    argumentUploadType(uploadType),
    referencedData(nullptr),
    dataOwned(true),
    persistentFlag(false),
    userBuffer(false)
{
    if (numberOfElements == 0 && data != nullptr)
    {
        throw std::runtime_error("Size of kernel argument must be greater than zero");
    }

    if (data != nullptr)
    {
        initializeData(data);
    }
}

KernelArgument::KernelArgument(const ArgumentId id, const size_t bufferSize, const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType) :
    id(id),
    elementSizeInBytes(elementSize),
    argumentDataType(dataType),
    argumentMemoryLocation(memoryLocation),
    argumentAccessType(accessType),
    argumentUploadType(ArgumentUploadType::Vector),
    referencedData(nullptr),
    dataOwned(false),
    persistentFlag(false),
    userBuffer(true)
{
    numberOfElements = bufferSize / elementSize;
}

void KernelArgument::updateData(void* data, const size_t numberOfElements)
{
    if (numberOfElements == 0)
    {
        throw std::runtime_error("Size of kernel argument must be greater than zero");
    }

    this->numberOfElements = numberOfElements;
    if (dataOwned)
    {
        initializeData(data);
    }
    else
    {
        referencedData = data;
    }
}

void KernelArgument::updateData(const void* data, const size_t numberOfElements)
{
    if (numberOfElements == 0 && data != nullptr)
    {
        throw std::runtime_error("Size of kernel argument must be greater than zero");
    }

    this->numberOfElements = numberOfElements;
    if (data != nullptr)
    {
        initializeData(data);
    }
}

void KernelArgument::setPersistentFlag(const bool flag)
{
    if (hasUserBuffer())
    {
        throw std::runtime_error("Persistent flag cannot be changed for kernel argument with user buffer");
    }

    persistentFlag = flag;
}

ArgumentId KernelArgument::getId() const
{
    return id;
}

size_t KernelArgument::getNumberOfElements() const
{
    return numberOfElements;
}

size_t KernelArgument::getElementSizeInBytes() const
{
    return elementSizeInBytes;
}

size_t KernelArgument::getDataSizeInBytes() const
{
    return numberOfElements * elementSizeInBytes;
}

ArgumentDataType KernelArgument::getDataType() const
{
    return argumentDataType;
}

ArgumentMemoryLocation KernelArgument::getMemoryLocation() const
{
    return argumentMemoryLocation;
}

ArgumentAccessType KernelArgument::getAccessType() const
{
    return argumentAccessType;
}

ArgumentUploadType KernelArgument::getUploadType() const
{
    return argumentUploadType;
}

const void* KernelArgument::getData() const
{
    if (!dataOwned)
    {
        return referencedData;
    }

    return ownedData.data();
}

void* KernelArgument::getData()
{
    return const_cast<void*>(static_cast<const KernelArgument*>(this)->getData());
}

bool KernelArgument::hasOwnedData() const
{
    return dataOwned;
}

bool KernelArgument::isPersistent() const
{
    return persistentFlag;
}

bool KernelArgument::hasUserBuffer() const
{
    return userBuffer;
}

bool KernelArgument::operator==(const KernelArgument& other) const
{
    return id == other.id;
}

bool KernelArgument::operator!=(const KernelArgument& other) const
{
    return !(*this == other);
}

void KernelArgument::initializeData(const void* data)
{
    prepareData();
    std::memcpy(getData(), data, getDataSizeInBytes());
}

void KernelArgument::prepareData()
{
    ownedData.resize(getDataSizeInBytes());
}

} // namespace ktt
