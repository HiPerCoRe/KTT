#pragma once

#ifdef KTT_API_VULKAN

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC system_header
#endif // __GNUC__

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100) // Unreferenced parameter
#pragma warning(disable : 4127) // Constant conditional expression
#pragma warning(disable : 4189) // Local variable initialized but not referenced
#pragma warning(disable : 4324) // Structure padding due to alignment specifier
#endif // _MSC_VER

#include <vk_mem_alloc.h>

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif // __GNUC__

#if defined(_MSC_VER)
#pragma warning(pop)
#endif // _MSC_VER

#endif // KTT_API_VULKAN
