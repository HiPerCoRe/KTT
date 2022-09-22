#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <KernelArgument/ArgumentAccessType.h>
#include <KernelArgument/ArgumentDataType.h>
#include <KernelArgument/ArgumentManagementType.h>
#include <KernelArgument/ArgumentMemoryLocation.h>
#include <KernelArgument/ArgumentMemoryType.h>
#include <KernelArgument/ArgumentOwnership.h>
#include <KttTypes.h>

namespace ktt
{

class KernelArgument
{
public:
    explicit KernelArgument(const ArgumentId id, const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
        const ArgumentManagementType managementType, const std::string& symbolName = "");

    void SetReferencedData(void* data, const size_t dataSize);
    void SetOwnedData(const void* data, const size_t dataSize);
    void SetOwnedDataFromFile(const std::string& file);
    void SetOwnedDataFromGenerator(const std::string& generatorFunction, const size_t dataSize);
    void SetUserBuffer(const size_t dataSize);

    ArgumentId GetId() const;
    size_t GetElementSize() const;
    ArgumentDataType GetDataType() const;
    ArgumentMemoryLocation GetMemoryLocation() const;
    ArgumentAccessType GetAccessType() const;
    ArgumentMemoryType GetMemoryType() const;
    ArgumentManagementType GetManagementType() const;
    const std::string& GetSymbolName() const;

    uint64_t GetNumberOfElements() const;
    size_t GetDataSize() const;
    const void* GetData() const;
    void* GetData();
    void SaveData(const std::string& file) const;
    bool HasOwnedData() const;
    bool HasUserBuffer() const;

    template <typename T>
    const T* GetDataWithType() const;

    template <typename T>
    uint64_t GetNumberOfElementsWithType() const;

    static std::vector<KernelArgument*> GetArgumentsWithMemoryType(const std::vector<KernelArgument*>& arguments,
        const ArgumentMemoryType type);

private:
    ArgumentId m_Id;
    size_t m_ElementSize;
    size_t m_DataSize;
    ArgumentDataType m_DataType;
    ArgumentMemoryLocation m_MemoryLocation;
    ArgumentAccessType m_AccessType;
    ArgumentMemoryType m_MemoryType;
    ArgumentManagementType m_ManagementType;
    ArgumentOwnership m_Ownership;
    std::string m_SymbolName;
    std::vector<uint8_t> m_Data;
    void* m_ReferencedData;
};

} // namespace ktt

#include <KernelArgument/KernelArgument.inl>
