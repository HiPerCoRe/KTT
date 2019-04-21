/** @file ktt_platform.h
  * Preprocessor definitions which ensure compatibility for multiple compilers and KTT version definitions.
  */
#pragma once

#ifndef KTT_API
#if defined(_MSC_VER) && !defined(KTT_TESTS)
    #pragma warning(disable : 4251) // Irrelevant MSVC warning as long as there are no public attributes
    #if defined(KTT_LIBRARY)
        #define KTT_API __declspec(dllexport)
    #else
        #define KTT_API __declspec(dllimport)
    #endif // KTT_LIBRARY
#else
    #define KTT_API
#endif // _MSC_VER
#endif // KTT_API

/** Major version of KTT framework. First number in KTT version description.
  */
#define KTT_VERSION_MAJOR 1

/** Minor version of KTT framework. Second number in KTT version description.
  */
#define KTT_VERSION_MINOR 1

/** Patch version of KTT framework. Third number in KTT version description.
  */
#define KTT_VERSION_PATCH 0
