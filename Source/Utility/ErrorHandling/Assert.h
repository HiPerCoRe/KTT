#pragma once

#include <string>

#if defined(KTT_CONFIGURATION_DEBUG)
    #define KttAssert(Expression, Message) ktt::AssertInternal(#Expression, Expression, __FILE__, __LINE__, Message);
    #define KttError(Message) ktt::AssertInternal("KTT error encountered", false, __FILE__, __LINE__, Message);
#else
    #define KttAssert(Expression, Message) /* Asserts disabled */;
    #define KttError(Message) /* Asserts disabled */;
#endif

namespace ktt
{

void AssertInternal(const char* expressionString, const bool expression, const char* file, const int line,
    const std::string& message);

} // namespace ktt
