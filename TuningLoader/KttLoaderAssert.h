#pragma once

#include <string>

#if defined(KTT_CONFIGURATION_DEBUG)
    #define KttLoaderAssert(Expression, Message) ktt::LoaderAssertInternal(#Expression, Expression, __FILE__, __LINE__, Message);
    #define KttLoaderError(Message) ktt::LoaderAssertInternal("KTT loader error encountered", false, __FILE__, __LINE__, Message);
#else
    #define KttLoaderAssert(Expression, Message) /* Asserts disabled */;
    #define KttLoaderError(Message) /* Asserts disabled */;
#endif

namespace ktt
{

void LoaderAssertInternal(const char* expressionString, const bool expression, const char* file, const int line,
    const std::string& message);

} // namespace ktt
