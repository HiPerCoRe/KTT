/** @file device_type.h
  * @brief Definition of enum for type of a device.
  */
#pragma once

namespace ktt
{

/** @enum DeviceType
  * @brief Enum for type of a device. It is based on device types supported by OpenCL API.
  */
enum class DeviceType
{
    /** @brief Device type corresponding to CPU device type in OpenCL.
      */
    CPU,

    /** @brief Device type corresponding to GPU device type in OpenCL. All available devices in CUDA API will also have this device type.
      */
    GPU,

    /** @brief Device type corresponding to accelerator device type in OpenCL.
      */
    Accelerator,

    /** @brief Device type corresponding to default device type in OpenCL.
      */
    Default,

    /** @brief Device type corresponding to custom device type in OpenCL.
      */
    Custom
};

} // namespace ktt
