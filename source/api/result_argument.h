#pragma once

#if defined(_MSC_VER) && !defined(KTT_TESTS)
    #pragma warning(disable : 4251)
    #if defined(KTT_LIBRARY)
        #define KTT_API __declspec(dllexport)
    #else
        #define KTT_API __declspec(dllimport)
    #endif // KTT_LIBRARY
#else
    #define KTT_API
#endif // _MSC_VER

#include <cstddef>
#include <vector>

#include "../enum/argument_data_type.h"
#include "../enum/argument_memory_type.h"

namespace ktt
{

class KTT_API ResultArgument
{
public:
    explicit ResultArgument(const size_t id, const void* data, const size_t numberOfElements, const size_t elementSizeInBytes,
        const ArgumentDataType& argumentDataType, const ArgumentMemoryType& argumentMemoryType);

    size_t getId() const;
    size_t getNumberOfElements() const;
    size_t getElementSizeInBytes() const;
    size_t getDataSizeInBytes() const;
    ArgumentDataType getArgumentDataType() const;
    ArgumentMemoryType getArgumentMemoryType() const;
    const void* getData() const;

private:
    size_t id;
    size_t numberOfElements;
    size_t elementSizeInBytes;
    ArgumentDataType argumentDataType;
    ArgumentMemoryType argumentMemoryType;
    const void* data;
};

} // namespace ktt
