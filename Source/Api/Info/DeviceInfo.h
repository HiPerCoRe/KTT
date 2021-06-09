/** @file DeviceInfo.h
  * Information about compute API devices.
  */
#pragma once

#include <cstdint>
#include <string>

#include <Api/Info/DeviceType.h>
#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

/** @class DeviceInfo
  * Class which holds information about a compute API device.
  */
class KTT_API DeviceInfo
{
public:
    /** @fn explicit DeviceInfo(const DeviceIndex index, const std::string& name)
      * Constructor which creates new device info object.
      * @param index Index of device assigned by KTT framework.
      * @param name Name of device retrieved from compute API.
      */
    explicit DeviceInfo(const DeviceIndex index, const std::string& name);

    /** @fn DeviceIndex GetIndex() const
      * Getter for index of device assigned by KTT framework.
      * @return Index of device assigned by KTT framework.
      */
    DeviceIndex GetIndex() const;

    /** @fn const std::string& GetName() const
      * Getter for name of device retrieved from compute API.
      * @return Name of device retrieved from compute API.
      */
    const std::string& GetName() const;

    /** @fn const std::string& GetVendor() const
      * Getter for name of device vendor retrieved from compute API.
      * @return Name of device vendor retrieved from compute API.
      */
    const std::string& GetVendor() const;

    /** @fn const std::string& GetExtensions() const
      * Getter for list of supported device extensions retrieved from compute API.
      * @return List of supported device extensions retrieved from compute API.
      */
    const std::string& GetExtensions() const;

    /** @fn DeviceType GetDeviceType() const
      * Getter for type of device. See ::DeviceType for more information.
      * @return Type of device.
      */
    DeviceType GetDeviceType() const;

    /** @fn std::string GetDeviceTypeString() const
      * Getter for type of device converted to string. See ::DeviceType for more information.
      * @return Type of device converted to string.
      */
    std::string GetDeviceTypeString() const;

    /** @fn uint64_t GetGlobalMemorySize() const
      * Getter for global memory size of device retrieved from compute API.
      * @return Global memory size of device retrieved from compute API.
      */
    uint64_t GetGlobalMemorySize() const;

    /** @fn uint64_t GetLocalMemorySize() const
      * Getter for local memory (shared memory in CUDA) size of device retrieved from compute API.
      * @return Local memory size of device retrieved from compute API.
      */
    uint64_t GetLocalMemorySize() const;

    /** @fn uint64_t GetMaxConstantBufferSize() const
      * Getter for constant memory size of device retrieved from compute API.
      * @return Constant memory size of device retrieved from compute API.
      */
    uint64_t GetMaxConstantBufferSize() const;

    /** @fn uint64_t GetMaxWorkGroupSize() const
      * Getter for maximum work-group (thread block in CUDA) size of device retrieved from compute API.
      * @return Maximum work-group size of device retrieved from compute API.
      */
    uint64_t GetMaxWorkGroupSize() const;

    /** @fn uint32_t GetMaxComputeUnits() const
      * Getter for maximum parallel compute units (multiprocessors in CUDA) count of device retrieved from compute API.
      * @return Maximum parallel compute units count of device retrieved from compute API.
      */
    uint32_t GetMaxComputeUnits() const;

    /** @fn std::string GetString() const
      * Converts device info to string.
      * @return String containing information about the device.
      */
    std::string GetString() const;

    /** @fn void SetVendor(const std::string& vendor)
      * Setter for name of device vendor.
      * @param vendor Name of device vendor.
      */
    void SetVendor(const std::string& vendor);

    /** @fn void SetExtensions(const std::string& extensions)
      * Setter for list of supported device extensions.
      * @param extensions List of supported device extensions.
      */
    void SetExtensions(const std::string& extensions);

    /** @fn void SetDeviceType(const DeviceType deviceType)
      * Setter for type of device.
      * @param deviceType Type of device.
      */
    void SetDeviceType(const DeviceType deviceType);

    /** @fn void SetGlobalMemorySize(const uint64_t globalMemorySize)
      * Setter for global memory size of device.
      * @param globalMemorySize Global memory size of device.
      */
    void SetGlobalMemorySize(const uint64_t globalMemorySize);

    /** @fn void SetLocalMemorySize(const uint64_t localMemorySize)
      * Setter for local memory size of device.
      * @param localMemorySize Local memory size of device.
      */
    void SetLocalMemorySize(const uint64_t localMemorySize);

    /** @fn void SetMaxConstantBufferSize(const uint64_t maxConstantBufferSize)
      * Setter for constant memory size of device.
      * @param maxConstantBufferSize Constant memory size of device.
      */
    void SetMaxConstantBufferSize(const uint64_t maxConstantBufferSize);

    /** @fn void SetMaxWorkGroupSize(const uint64_t maxWorkGroupSize)
      * Setter for maximum work-group size of device.
      * @param maxWorkGroupSize Maximum work-group size of device.
      */
    void SetMaxWorkGroupSize(const uint64_t maxWorkGroupSize);

    /** @fn void SetMaxComputeUnits(const uint32_t maxComputeUnits)
      * Setter for maximum compute units count of device.
      * @param maxComputeUnits Maximum compute units count of device.
      */
    void SetMaxComputeUnits(const uint32_t maxComputeUnits);

private:
    DeviceIndex m_Index;
    std::string m_Name;
    std::string m_Vendor;
    std::string m_Extensions;
    DeviceType m_DeviceType;
    uint64_t m_GlobalMemorySize;
    uint64_t m_LocalMemorySize;
    uint64_t m_MaxConstantBufferSize;
    uint64_t m_MaxWorkGroupSize;
    uint32_t m_MaxComputeUnits;
};

} // namespace ktt
