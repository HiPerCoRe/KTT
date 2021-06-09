/** @file DeviceType.h
  * Type of compute device.
  */
#pragma once

namespace ktt
{

/** @enum DeviceType
  * Enum for type of compute device. Based on device types available in OpenCL API.
  */
enum class DeviceType
{
    /** Device is a CPU.
      */
    CPU,

    /** Device is a GPU. All available devices in CUDA API and Vulkan will have this device type.
      */
    GPU,

    /** Device has type other than CPU or GPU.
      */
    Custom
};

} // namespace ktt
