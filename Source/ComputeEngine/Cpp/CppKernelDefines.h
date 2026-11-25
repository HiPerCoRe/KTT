#ifndef KTT_CPP_KERNEL_DEFINES_H
#define KTT_CPP_KERNEL_DEFINES_H

#include <cstdint>

// This header defines the scalar argument type of the KTT C++ backend. KTT injects the full text
// of this file as a preamble into every JIT-compiled C++ kernel and also emits it as a real file
// into the JIT temporary directory (with an include path added to the kernel compile command), so
// a kernel can use the type without any include, or include "ComputeEngine/Cpp/CppKernelDefines.h"
// explicitly.
// Kernel authors must not define their own ScalarType; the definition comes from KTT.
//
// NOTE for KTT developers: Source/ComputeEngine/Cpp/CppKernelDefines.generated.h is generated from
// this file by premake5.lua during project generation and must never be edited by hand.

namespace ktt
{

// Scalar kernel argument value. KTT writes the member that matches the argument's declared data
// type; the kernel reads the member that matches its own scalar type. Member names match the
// ktt::ArgumentDataType enumerators: Char, UnsignedChar, Short, UnsignedShort, Int, UnsignedInt,
// Long, UnsignedLong, Float, Double. Half scalars are carried in UnsignedShort as the IEEE 754
// binary16 bit pattern, because half is not a type in the C++ standard.
union ScalarType
{
    uint8_t UnsignedChar;
    uint16_t UnsignedShort;
    uint32_t UnsignedInt;
    uint64_t UnsignedLong;
    int8_t Char;
    int16_t Short;
    int32_t Int;
    int64_t Long;
    float Float;
    double Double;
};

} // namespace ktt

#endif // KTT_CPP_KERNEL_DEFINES_H
