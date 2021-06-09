/** @file BufferOutputDescriptor.h
  * Kernel buffer output retrieval.
  */
#pragma once

#include <cstddef>

#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

/** @class BufferOutputDescriptor
  * Class which can be used to retrieve kernel argument data when calling certain KTT API methods.
  */
class KTT_API BufferOutputDescriptor
{
public:
    /** @fn explicit BufferOutputDescriptor(const ArgumentId id, void* outputDestination)
      * Constructor, which creates new output descriptor object for specified kernel argument.
      * @param id Id of vector argument which will be retrieved.
      * @param outputDestination Pointer to destination where vector argument data will be copied. Destination buffer size
      * needs to be equal or greater than argument size.
      */
    explicit BufferOutputDescriptor(const ArgumentId id, void* outputDestination);

    /** @fn explicit BufferOutputDescriptor(const ArgumentId id, void* outputDestination, const size_t outputSize)
      * Constructor, which creates new output descriptor object for specified kernel argument.
      * @param id Id of vector argument which will be retrieved.
      * @param outputDestination Pointer to destination where vector argument data will be copied. Destination buffer size
      * needs to be equal or greater than specified output size.
      * @param outputSize Size of output in bytes which will be copied to specified destination, starting with the first
      * byte in argument.
      */
    explicit BufferOutputDescriptor(const ArgumentId id, void* outputDestination, const size_t outputSize);

    /** @fn ArgumentId GetArgumentId() const
      * Getter for id of argument tied to output descriptor.
      * @return Id of argument tied to output descriptor.
      */
    ArgumentId GetArgumentId() const;

    /** @fn void* GetOutputDestination() const
      * Getter for pointer to destination buffer tied to output descriptor.
      * @return Pointer to destination buffer tied to output descriptor.
      */
    void* GetOutputDestination() const;

    /** @fn size_t GetOutputSize() const
      * Getter for data size retrieved with output descriptor.
      * @return Data size in bytes retrieved with output descriptor. Returns 0 if entire argument is retrieved.
      */
    size_t GetOutputSize() const;

private:
    ArgumentId m_ArgumentId;
    void* m_OutputDestination;
    size_t m_OutputSize;
};

} // namespace ktt
