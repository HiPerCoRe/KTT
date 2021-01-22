#pragma once

#ifdef KTT_API_CUDA

#if defined(KTT_PROFILING_CUPTI_LEGACY) || defined(KTT_PROFILING_CUPTI)

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324) // Structure padding due to alignment specifier in CUPTI API
#endif // _MSC_VER

#include <cupti.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif // _MSC_VER

#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI

#endif // KTT_API_CUDA
