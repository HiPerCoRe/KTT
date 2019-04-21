/** @file device_info.h
  * Functionality related to retrieving information about compute API devices.
  */
#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <enum/device_type.h>
#include <ktt_platform.h>
#include <ktt_types.h>

namespace ktt
{

/** @class DeviceInfo
  * Class which holds information about a compute API device.
  */
class KTT_API DeviceInfo
{
public:
    /** @fn explicit DeviceInfo(const DeviceIndex device, const std::string& name)
      * Constructor which creates new device info object.
      * @param device Index of device assigned by KTT framework.
      * @param name Name of device retrieved from compute API.
      */
    explicit DeviceInfo(const DeviceIndex device, const std::string& name);

    /** @fn DeviceIndex getId() const
      * Getter for index of device assigned by KTT framework.
      * @return Index of device assigned by KTT framework.
      */
    DeviceIndex getId() const;

    /** @fn const std::string& getName() const
      * Getter for name of device retrieved from compute API.
      * @return Name of device retrieved from compute API.
      */
    const std::string& getName() const;

    /** @fn const std::string& getVendor() const
      * Getter for name of device vendor retrieved from compute API.
      * @return Name of device vendor retrieved from compute API.
      */
    const std::string& getVendor() const;

    /** @fn const std::string& getExtensions() const
      * Getter for list of supported device extensions retrieved from compute API.
      * @return List of supported device extensions retrieved from compute API.
      */
    const std::string& getExtensions() const;

    /** @fn DeviceType getDeviceType() const
      * Getter for type of device. See ::DeviceType for more information.
      * @return Type of device.
      */
    DeviceType getDeviceType() const;

    /** @fn std::string getDeviceTypeAsString() const
      * Getter for type of device converted to string. See ::DeviceType for more information.
      * @return Type of device converted to string.
      */
    std::string getDeviceTypeAsString() const;

    /** @fn uint64_t getGlobalMemorySize() const
      * Getter for global memory size of device retrieved from compute API.
      * @return Global memory size of device retrieved from compute API.
      */
    uint64_t getGlobalMemorySize() const;

    /** @fn uint64_t getLocalMemorySize() const
      * Getter for local memory (shared memory in CUDA) size of device retrieved from compute API.
      * @return Local memory size of device retrieved from compute API.
      */
    uint64_t getLocalMemorySize() const;

    /** @fn uint64_t getMaxConstantBufferSize() const
      * Getter for constant memory size of device retrieved from compute API.
      * @return Constant memory size of device retrieved from compute API.
      */
    uint64_t getMaxConstantBufferSize() const;

    /** @fn uint32_t getMaxComputeUnits() const
      * Getter for maximum parallel compute units (multiprocessors in CUDA) count of device retrieved from compute API.
      * @return Maximum parallel compute units count of device retrieved from compute API.
      */
    uint32_t getMaxComputeUnits() const;

    /** @fn size_t getMaxWorkGroupSize() const
      * Getter for maximum work-group (thread block in CUDA) size of device retrieved from compute API.
      * @return Maximum work-group size of device retrieved from compute API.
      */
    size_t getMaxWorkGroupSize() const;

    /** @fn void setVendor(const std::string& vendor)
      * Setter for name of device vendor.
      * @param vendor Name of device vendor.
      */
    void setVendor(const std::string& vendor);

    /** @fn void setExtensions(const std::string& extensions)
      * Setter for list of supported device extensions.
      * @param extensions List of supported device extensions.
      */
    void setExtensions(const std::string& extensions);

    /** @fn void setDeviceType(const DeviceType deviceType)
      * Setter for type of device.
      * @param deviceType Type of device.
      */
    void setDeviceType(const DeviceType deviceType);

    /** @fn void setGlobalMemorySize(const uint64_t globalMemorySize)
      * Setter for global memory size of device.
      * @param globalMemorySize Global memory size of device.
      */
    void setGlobalMemorySize(const uint64_t globalMemorySize);

    /** @fn void setLocalMemorySize(const uint64_t localMemorySize)
      * Setter for local memory size of device.
      * @param localMemorySize Local memory size of device.
      */
    void setLocalMemorySize(const uint64_t localMemorySize);

    /** @fn void setMaxConstantBufferSize(const uint64_t maxConstantBufferSize)
      * Setter for constant memory size of device.
      * @param maxConstantBufferSize Constant memory size of device.
      */
    void setMaxConstantBufferSize(const uint64_t maxConstantBufferSize);

    /** @fn void setMaxComputeUnits(const uint32_t maxComputeUnits)
      * Setter for maximum compute units count of device.
      * @param maxComputeUnits Maximum compute units count of device.
      */
    void setMaxComputeUnits(const uint32_t maxComputeUnits);

    /** @fn void setMaxWorkGroupSize(const size_t maxWorkGroupSize)
      * Setter for maximum work-group size of device.
      * @param maxWorkGroupSize Maximum work-group size of device.
      */
    void setMaxWorkGroupSize(const size_t maxWorkGroupSize);

private:
    DeviceIndex id;
    std::string name;
    std::string vendor;
    std::string extensions;
    DeviceType deviceType;
    uint64_t globalMemorySize;
    uint64_t localMemorySize;
    uint64_t maxConstantBufferSize;
    uint32_t maxComputeUnits;
    size_t maxWorkGroupSize;
};

/** @fn std::ostream& operator<<(std::ostream& outputTarget, const DeviceInfo& deviceInfo)
  * Output operator for device info class.
  * @param outputTarget Location where information about device will be printed.
  * @param deviceInfo Device info object that will be printed.
  * @return Output target to support chaining of output operations.
  */
KTT_API std::ostream& operator<<(std::ostream& outputTarget, const DeviceInfo& deviceInfo);

} // namespace ktt
