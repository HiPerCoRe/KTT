#pragma once

#if defined(_WIN32) && !defined(KTT_TESTS)
    #pragma warning(disable : 4251)
    #if defined(KTT_LIBRARY)
        #define KTT_API __declspec(dllexport)
    #else
        #define KTT_API __declspec(dllimport)
    #endif // KTT_LIBRARY
#else
    #define KTT_API
#endif // _WIN32

#include <cstdint>
#include <iostream>
#include <string>

#include "../enum/device_type.h"

namespace ktt
{

class KTT_API DeviceInfo
{
public:
    explicit DeviceInfo(const size_t id, const std::string& name);

    size_t getId() const;
    std::string getName() const;
    std::string getVendor() const;
    std::string getExtensions() const;
    DeviceType getDeviceType() const;
    uint64_t getGlobalMemorySize() const;
    uint64_t getLocalMemorySize() const;
    uint64_t getMaxConstantBufferSize() const;
    uint32_t getMaxComputeUnits() const;
    size_t getMaxWorkGroupSize() const;

    void setVendor(const std::string& vendor);
    void setExtensions(const std::string& extensions);
    void setDeviceType(const DeviceType& deviceType);
    void setGlobalMemorySize(const uint64_t globalMemorySize);
    void setLocalMemorySize(const uint64_t localMemorySize);
    void setMaxConstantBufferSize(const uint64_t maxConstantBufferSize);
    void setMaxComputeUnits(const uint32_t maxComputeUnits);
    void setMaxWorkGroupSize(const size_t maxWorkGroupSize);

    KTT_API friend std::ostream& operator<<(std::ostream&, const DeviceInfo&);

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

std::string deviceTypeToString(const DeviceType& deviceType);
KTT_API std::ostream& operator<<(std::ostream& outputTarget, const DeviceInfo& deviceInfo);

} // namespace ktt
