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

namespace ktt
{

class KTT_API ArgumentOutputDescriptor
{
public:
    explicit ArgumentOutputDescriptor(const size_t argumentId, void* outputDestination, const size_t outputSizeInBytes);

    size_t getArgumentId() const;
    void* getOutputDestination() const;
    size_t getOutputSizeInBytes() const;

private:
    size_t argumentId;
    void* outputDestination;
    size_t outputSizeInBytes;
};

} // namespace ktt
