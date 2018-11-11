/** @file output_descriptor.h
  * Functionality related to retrieving kernel output with KTT API.
  */
#pragma once

#include <cstddef>
#include <ktt_platform.h>
#include <ktt_types.h>

namespace ktt
{

/** @class OutputDescriptor
  * Class which can be used to retrieve kernel argument data when calling certain KTT API methods.
  */
class KTT_API OutputDescriptor
{
public:
    /** @fn explicit OutputDescriptor(const ArgumentId id, void* outputDestination)
      * Constructor, which creates new output descriptor object for specified kernel argument.
      * @param id Id of vector argument which will be retrieved.
      * @param outputDestination Pointer to destination where vector argument data will be copied. Destination buffer size needs to be equal or
      * greater than argument size.
      */
    explicit OutputDescriptor(const ArgumentId id, void* outputDestination);

    /** @fn explicit OutputDescriptor(const ArgumentId id, void* outputDestination, const size_t outputSizeInBytes)
      * Constructor, which creates new output descriptor object for specified kernel argument.
      * @param id Id of vector argument which will be retrieved.
      * @param outputDestination Pointer to destination where vector argument data will be copied. Destination buffer size needs to be equal or
      * greater than specified output size.
      * @param outputSizeInBytes Size of output in bytes which will be copied to specified destination, starting with first byte in argument.
      */
    explicit OutputDescriptor(const ArgumentId id, void* outputDestination, const size_t outputSizeInBytes);

    /** @fn ArgumentId getArgumentId() const
      * Getter for id of argument tied to output descriptor.
      * @return Id of argument tied to output descriptor.
      */
    ArgumentId getArgumentId() const;

    /** @fn void* getOutputDestination() const
      * Getter for pointer to destination buffer tied to output descriptor.
      * @return Pointer to destination buffer tied to output descriptor.
      */
    void* getOutputDestination() const;

    /** @fn size_t getOutputSizeInBytes() const
      * Getter for data size retrieved with output descriptor.
      * @return Data size retrieved with output descriptor. Returns 0 if entire argument is retrieved.
      */
    size_t getOutputSizeInBytes() const;

private:
    ArgumentId argumentId;
    void* outputDestination;
    size_t outputSizeInBytes;
};

} // namespace ktt
