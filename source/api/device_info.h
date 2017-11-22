/** @file device_info.h
  * @brief Functionality related to retrieving information about compute API devices.
  */
#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include "ktt_platform.h"
#include "enum/device_type.h"

namespace ktt
{

/** @class DeviceInfo
  * @brief Class which holds information about a compute API device.
  */
class KTT_API DeviceInfo
{
public:
    /** @fn explicit DeviceInfo(const size_t id, const std::string& name)
      * @brief Constructor, which creates new device info object.
      * @param id Id of device assigned by KTT library.
      * @param name Name of device retrieved from compute API.
      */
    explicit DeviceInfo(const size_t id, const std::string& name);

    /** @fn size_t getId() const
      * @brief Getter for id of device assigned by KTT library.
      * @return Id of device assigned by KTT library.
      */
    size_t getId() const;

    /** @fn std::string getName() const
      * @brief Getter for name of device retrieved from compute API.
      * @return Name of device retrieved from compute API.
      */
    std::string getName() const;

    /** @fn std::string getVendor() const
      * @brief Getter for name of device vendor retrieved from compute API.
      * @return Name of device vendor retrieved from compute API.
      */
    std::string getVendor() const;

    /** @fn std::string getExtensions() const
      * @brief Getter for list of supported device extensions retrieved from compute API.
      * @return List of supported device extensions retrieved from compute API.
      */
    std::string getExtensions() const;

    /** @fn DeviceType getDeviceType() const
      * @brief Getter for type of device. See DeviceType for more information.
      * @return Type of device.
      */
    DeviceType getDeviceType() const;

    /** @fn std::string getDeviceTypeAsString() const
      * @brief Getter for type of device converted to string. See DeviceType for more information.
      * @return Type of device converted to string.
      */
    std::string getDeviceTypeAsString() const;

    /** @fn uint64_t getGlobalMemorySize() const
      * @brief Getter for global memory size of device retrieved from compute API.
      * @return Global memory size of device retrieved from compute API.
      */
    uint64_t getGlobalMemorySize() const;

    /** @fn uint64_t getLocalMemorySize() const
      * @brief Getter for local memory (shared memory in CUDA) size of device retrieved from compute API.
      * @return Local memory size of device retrieved from compute API.
      */
    uint64_t getLocalMemorySize() const;

    /** @fn uint64_t getMaxConstantBufferSize() const
      * @brief Getter for constant memory size of device retrieved from compute API.
      * @return Constant memory size of device retrieved from compute API.
      */
    uint64_t getMaxConstantBufferSize() const;

    /** @fn uint32_t getMaxComputeUnits() const
      * @brief Getter for maximum parallel compute units (multiprocessors in CUDA) count of device retrieved from compute API.
      * @return Maximum parallel compute units count of device retrieved from compute API.
      */
    uint32_t getMaxComputeUnits() const;

    /** @fn size_t getMaxWorkGroupSize() const
      * @brief Getter for maximum work-group (thread block in CUDA) size of device retrieved from compute API.
      * @return Maximum work-group size of device retrieved from compute API.
      */
    size_t getMaxWorkGroupSize() const;

    /** @fn void setVendor(const std::string& vendor)
      * @brief Setter for name of device vendor.
      * @param vendor Name of device vendor.
      */
    void setVendor(const std::string& vendor);

    /** @fn void setExtensions(const std::string& extensions)
      * @brief Setter for list of supported device extensions.
      * @param extensions List of supported device extensions.
      */
    void setExtensions(const std::string& extensions);

    /** @fn void setDeviceType(const DeviceType& deviceType)
      * @brief Setter for type of device.
      * @param deviceType Type of device.
      */
    void setDeviceType(const DeviceType& deviceType);

    /** @fn void setGlobalMemorySize(const uint64_t globalMemorySize)
      * @brief Setter for global memory size of device.
      * @param globalMemorySize Global memory size of device.
      */
    void setGlobalMemorySize(const uint64_t globalMemorySize);

    /** @fn void setLocalMemorySize(const uint64_t localMemorySize)
      * @brief Setter for local memory size of device.
      * @param localMemorySize Local memory size of device.
      */
    void setLocalMemorySize(const uint64_t localMemorySize);

    /** @fn void setMaxConstantBufferSize(const uint64_t maxConstantBufferSize)
      * @brief Setter for constant memory size of device.
      * @param maxConstantBufferSize Constant memory size of device.
      */
    void setMaxConstantBufferSize(const uint64_t maxConstantBufferSize);

    /** @fn void setMaxComputeUnits(const uint32_t maxComputeUnits)
      * @brief Setter for maximum compute units count of device.
      * @param maxComputeUnits Maximum compute units count of device.
      */
    void setMaxComputeUnits(const uint32_t maxComputeUnits);

    /** @fn void setMaxWorkGroupSize(const size_t maxWorkGroupSize)
      * @brief Setter for maximum work-group size of device.
      * @param maxWorkGroupSize Maximum work-group size of device.
      */
    void setMaxWorkGroupSize(const size_t maxWorkGroupSize);

private:
    size_t id;
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
  * @brief Output operator for device info class.
  * @param outputTarget Location where information about device is printed.
  * @param deviceInfo Device info object that is printed.
  * @return Output target to support chain of output operations.
  */
KTT_API std::ostream& operator<<(std::ostream& outputTarget, const DeviceInfo& deviceInfo);

} // namespace ktt
