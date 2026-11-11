#ifndef KTT_CPP_KERNEL_DEFINES_H
#define KTT_CPP_KERNEL_DEFINES_H

#include <cstdint>

// This header defines the scalar argument type of the KTT C++ backend. KTT injects the full text
// of this file as a preamble into every JIT-compiled C++ kernel and also emits it as a real file
// into the JIT temporary directory (with an include path added to the kernel compile command), so
// a kernel can use the type without any include, or include "CppKernelDefines.h" explicitly.
// Kernel authors must not define their own ScalarType; the definition comes from KTT.
//
// NOTE for KTT developers: Source/ComputeEngine/Cpp/CppKernelDefines.generated.h is generated from
// this file by premake5.lua during project generation and must never be edited by hand.

namespace ktt
{

// Scalar kernel argument value. KTT writes the member that matches the argument's declared data
// type; the kernel reads the member that matches its own scalar type. Member names follow the
// <cstdint> fixed-width names, with these mappings from ktt::ArgumentDataType:
// Char->i8, UnsignedChar->u8, Short->i16, UnsignedShort->u16, Int->i32, UnsignedInt->u32,
// Long->i64, UnsignedLong->u64, Float->f, Double->d, Half->u16 (IEEE 754 binary16 bit pattern).
union ScalarType
{
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f;
    double d;
};

} // namespace ktt

#endif // KTT_CPP_KERNEL_DEFINES_H
