/** @file TuningLoaderPlatform.h
  * Preprocessor definitions which ensure compatibility for multiple compilers.
  */
#pragma once

#include <cstdint>
#include <string>

#ifndef KTT_LOADER_API
#if defined(_MSC_VER)
    #pragma warning(disable : 4251) // Irrelevant MSVC warning as long as exported classes have no public attributes.
    
    #if defined(KTT_LOADER_LIBRARY)
        #define KTT_LOADER_API __declspec(dllexport)
    #else
        #define KTT_LOADER_API __declspec(dllimport)
    #endif // KTT_LOADER_LIBRARY
#else
    #define KTT_LOADER_API
#endif // _MSC_VER
#endif // KTT_LOADER_API
