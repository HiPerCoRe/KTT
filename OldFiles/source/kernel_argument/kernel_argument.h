#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <enum/argument_access_type.h>
#include <enum/argument_data_type.h>
#include <enum/argument_memory_location.h>
#include <enum/argument_upload_type.h>
#include <ktt_types.h>

namespace ktt
{

class KernelArgument
{
public:
    // Constructors
    explicit KernelArgument(const ArgumentId id, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType);
    explicit KernelArgument(const ArgumentId id, void* data, const size_t numberOfElements, const size_t elementSizeInBytes,
        const ArgumentDataType dataType, const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType,
        const ArgumentUploadType uploadType, const bool dataOwned);
    explicit KernelArgument(const ArgumentId id, const void* data, const size_t numberOfElements, const size_t elementSizeInBytes,
        const ArgumentDataType dataType, const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType,
        const ArgumentUploadType uploadType);
    explicit KernelArgument(const ArgumentId id, const size_t bufferSize, const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType);

    // Core methods
    void updateData(void* data, const size_t numberOfElements);
    void updateData(const void* data, const size_t numberOfElements);
    void setPersistentFlag(const bool flag);

    // Getters
    ArgumentId getId() const;
    size_t getNumberOfElements() const;
    size_t getElementSizeInBytes() const;
    size_t getDataSizeInBytes() const;
    ArgumentDataType getDataType() const;
    ArgumentMemoryLocation getMemoryLocation() const;
    ArgumentAccessType getAccessType() const;
    ArgumentUploadType getUploadType() const;
    const void* getData() const;
    void* getData();
    bool hasOwnedData() const;
    bool isPersistent() const;
    bool hasUserBuffer() const;

    template <typename T>
    const T* getDataWithType() const
    {
        if (dataOwned)
        {
            return reinterpret_cast<const T*>(ownedData.data());
        }

        return reinterpret_cast<const T*>(referencedData);
    }

    template <typename T>
    const size_t getNumberOfElementsWithType() const
    {
        return getDataSizeInBytes() / sizeof(T);
    }

    // Operators
    bool operator==(const KernelArgument& other) const;
    bool operator!=(const KernelArgument& other) const;

private:
    // Attributes
    ArgumentId id;
    size_t numberOfElements;
    size_t elementSizeInBytes;
    ArgumentDataType argumentDataType;
    ArgumentMemoryLocation argumentMemoryLocation;
    ArgumentAccessType argumentAccessType;
    ArgumentUploadType argumentUploadType;
    std::vector<uint8_t> ownedData;
    void* referencedData;
    bool dataOwned;
    bool persistentFlag;
    bool userBuffer;

    // Helper methods
    void initializeData(const void* data);
    void prepareData();
};

} // namespace ktt
