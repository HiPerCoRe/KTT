#pragma once

#ifndef KTT_API
#if defined(_MSC_VER) && !defined(KTT_TESTS)
    #pragma warning(disable : 4251) // MSVC irrelevant warning (as long as there are no public attributes)
    #if defined(KTT_LIBRARY)
        #define KTT_API __declspec(dllexport)
    #else
        #define KTT_API __declspec(dllimport)
    #endif // KTT_LIBRARY
#else
    #define KTT_API
#endif // _MSC_VER
#endif // KTT_API

#include <cstdint>
#include <vector>

#include "half.hpp"
#include "enum/argument_data_type.h"

namespace ktt
{

using half_float::half;

class KTT_API ResultArgument
{
public:
    explicit ResultArgument(const size_t id, const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType);

    size_t getId() const;
    size_t getNumberOfElements() const;
    size_t getDataSizeInBytes() const;
    const void* getData() const;
    void* getData();

private:
    size_t id;
    size_t numberOfElements;
    ArgumentDataType argumentDataType;
    std::vector<int8_t> dataChar;
    std::vector<uint8_t> dataUnsignedChar;
    std::vector<int16_t> dataShort;
    std::vector<uint16_t> dataUnsignedShort;
    std::vector<int32_t> dataInt;
    std::vector<uint32_t> dataUnsignedInt;
    std::vector<int64_t> dataLong;
    std::vector<uint64_t> dataUnsignedLong;
    std::vector<half> dataHalf;
    std::vector<float> dataFloat;
    std::vector<double> dataDouble;

    size_t getElementSizeInBytes() const;
    void initializeData(const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType);
    void prepareData(const size_t numberOfElements, const ArgumentDataType& argumentDataType);
};

} // namespace ktt
