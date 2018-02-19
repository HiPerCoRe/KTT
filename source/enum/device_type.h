/** @file device_type.h
  * Definition of enum for type of a device.
  */
#pragma once

namespace ktt
{

/** @enum DeviceType
  * Enum for type of a device. It is based on device types supported by OpenCL API.
  */
enum class DeviceType
{
    /** Device type corresponding to CPU device type in OpenCL.
      */
    CPU,

    /** Device type corresponding to GPU device type in OpenCL. All available devices in CUDA API will also have this device type.
      */
    GPU,

    /** Device type corresponding to accelerator device type in OpenCL.
      */
    Accelerator,

    /** Device type corresponding to default device type in OpenCL.
      */
    Default,

    /** Device type corresponding to custom device type in OpenCL.
      */
    Custom
};

} // namespace ktt
