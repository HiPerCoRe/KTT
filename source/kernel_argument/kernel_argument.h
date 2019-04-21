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
        const ArgumentUploadType uploadType, const bool dataCopied);
    explicit KernelArgument(const ArgumentId id, const void* data, const size_t numberOfElements, const size_t elementSizeInBytes,
        const ArgumentDataType dataType, const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType,
        const ArgumentUploadType uploadType);

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
    template <typename T> std::vector<T> getDataWithType() const
    {
        std::vector<T> result;
        size_t dataSize = getDataSizeInBytes();
        result.resize(dataSize / sizeof(T));

        if (dataCopied)
        {
            std::memcpy(result.data(), copiedData.data(), dataSize);
        }
        else
        {
            std::memcpy(result.data(), referencedData, dataSize);
        }

        return result;
    }
    bool hasCopiedData() const;
    bool isPersistent() const;

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
    std::vector<uint8_t> copiedData;
    void* referencedData;
    bool dataCopied;
    bool persistentFlag;

    // Helper methods
    void initializeData(const void* data);
    void prepareData();
};

} // namespace ktt
