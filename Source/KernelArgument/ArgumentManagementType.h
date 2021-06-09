/** @file ArgumentManagementType.h
  * Management type of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentManagementType
  * Enum for management type of kernel arguments. Specifies who is responsible for managing vector kernel arguments.
  */
enum class ArgumentManagementType
{
    /** Vector kernel arguments are managed automatically by the framework.
      */
    Framework,

    /** Vector kernel arguments are managed by user. This means that user is responsible for uploading and downloading argument data
      * into compute API buffers at the right time. This can be achieved by utilizing ComputeInterface methods such as UploadBuffer()
      * and DownloadBuffer().
      */
    User
};

} // namespace ktt
