/** @file KttPlatform.h
  * Preprocessor definitions which ensure compatibility for multiple compilers and KTT version definitions.
  */
#pragma once

#include <cstdint>
#include <string>

#ifndef KTT_API
#if defined(_MSC_VER)
    #pragma warning(disable : 4251) // Irrelevant MSVC warning as long as exported classes have no public attributes.
    
    #if defined(KTT_LIBRARY)
        #define KTT_API __declspec(dllexport)
    #else
        #define KTT_API __declspec(dllimport)
    #endif // KTT_LIBRARY

    #define KTT_VIRTUAL_API virtual
    #define KTT_VISIBILITY_HIDDEN
#else
    #define KTT_API
    #define KTT_VIRTUAL_API
    #define KTT_VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif // _MSC_VER
#endif // KTT_API

/** Major version of KTT framework. First number in KTT version description.
  */
#define KTT_VERSION_MAJOR 2

/** Minor version of KTT framework. Second number in KTT version description.
  */
#define KTT_VERSION_MINOR 1

/** Patch version of KTT framework. Third number in KTT version description.
  */
#define KTT_VERSION_PATCH 0

namespace ktt
{

/** @fn uint32_t GetKttVersion()
  * Returns the current KTT framework version in integer format.
  * @return KTT framework version as integer xxxyyyzzz.
  */
KTT_API uint32_t GetKttVersion();

/** @fn std::string GetKttVersionString()
  * Returns the current KTT framework version in string format.
  * @return KTT framework version as string x.y.z.
  */
KTT_API std::string GetKttVersionString();

} // namespace ktt
