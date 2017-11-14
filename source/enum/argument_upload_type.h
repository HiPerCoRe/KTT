/** @file argument_upload_type.h
  * @brief Definition of enum for upload type of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentUploadType
  * @brief Enum for upload type of kernel arguments. Specifies which compute API function should be used internally by KTT library to make
  * the argument accessible to kernel functions.
  */
enum class ArgumentUploadType
{
    /** @brief Argument will be uploaded as a scalar. Scalar arguments are uploaded into kernel as a local copy.
      */
    Scalar,

    /** @brief Argument will be uploaded as a vector.
      */
    Vector,

    /** @brief Argument will be located in local memory. Kernel arguments cannot be directly transferred into local memory from host memory.
      * Assigning local memory argument to kernel from KTT API simply means that the compute API will allocate enough local memory to hold number
      * of elements specified for the argument.
      */
    Local
};

} // namespace ktt
