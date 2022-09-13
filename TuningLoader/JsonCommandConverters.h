#pragma once

#include <json.hpp>

#include <Commands/AddArgumentCommand.h>
#include <Commands/AddKernelCommand.h>
#include <Commands/CompilerOptionsCommand.h>
#include <Commands/ConstraintCommand.h>
#include <Commands/CreateTunerCommand.h>
#include <Commands/ModifierCommand.h>
#include <Commands/OutputCommand.h>
#include <Commands/ParameterCommand.h>
#include <Commands/SharedMemoryCommand.h>
#include <Commands/TimeUnitCommand.h>
#include <ArgumentFillType.h>

namespace ktt
{

using json = nlohmann::json;

NLOHMANN_JSON_SERIALIZE_ENUM(ArgumentFillType,
{
    {ArgumentFillType::Constant, "Constant"},
    {ArgumentFillType::Ascending, "Ascending"},
    {ArgumentFillType::Random, "Random"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ArgumentAccessType,
{
    {ArgumentAccessType::Undefined, "Undefined"},
    {ArgumentAccessType::ReadOnly, "ReadOnly"},
    {ArgumentAccessType::WriteOnly, "WriteOnly"},
    {ArgumentAccessType::ReadWrite, "ReadWrite"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ArgumentDataType,
{
    {ArgumentDataType::Char, "bool"},
    {ArgumentDataType::Char, "int8"},
    {ArgumentDataType::UnsignedChar, "uint8"},
    {ArgumentDataType::Short, "int16"},
    {ArgumentDataType::UnsignedShort, "uint16"},
    {ArgumentDataType::Int, "int32"},
    {ArgumentDataType::UnsignedInt, "uint32"},
    {ArgumentDataType::Long, "int64"},
    {ArgumentDataType::UnsignedLong, "uint64"},
    {ArgumentDataType::Half, "half"},
    {ArgumentDataType::Custom, "half2"},
    {ArgumentDataType::Custom, "half4"},
    {ArgumentDataType::Custom, "half8"},
    {ArgumentDataType::Custom, "half16"},
    {ArgumentDataType::Float, "float"},
    {ArgumentDataType::Custom, "float2"},
    {ArgumentDataType::Custom, "float4"},
    {ArgumentDataType::Custom, "float8"},
    {ArgumentDataType::Custom, "float16"},
    {ArgumentDataType::Double, "double"},
    {ArgumentDataType::Custom, "double2"},
    {ArgumentDataType::Custom, "double4"},
    {ArgumentDataType::Custom, "double8"},
    {ArgumentDataType::Custom, "double16"},
    {ArgumentDataType::Custom, "custom"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ArgumentMemoryType,
{
    {ArgumentMemoryType::Scalar, "Scalar"},
    {ArgumentMemoryType::Vector, "Vector"},
    {ArgumentMemoryType::Local, "Local"},
    {ArgumentMemoryType::Symbol, "Symbol"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ComputeApi,
{
    {ComputeApi::OpenCL, "OpenCL"},
    {ComputeApi::CUDA, "CUDA"},
    {ComputeApi::Vulkan, "Vulkan"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ModifierAction,
{
    {ModifierAction::Add, "Add"},
    {ModifierAction::Subtract, "Subtract"},
    {ModifierAction::Multiply, "Multiply"},
    {ModifierAction::Divide, "Divide"},
    {ModifierAction::DivideCeil, "DivideCeil"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ModifierDimension,
{
    {ModifierDimension::X, "X"},
    {ModifierDimension::Y, "Y"},
    {ModifierDimension::Z, "Z"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ModifierType,
{
    {ModifierType::Global, "Global"},
    {ModifierType::Local, "Local"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ParameterValueType,
{
    {ParameterValueType::Int, "Int"},
    {ParameterValueType::UnsignedInt, "UnsignedInt"},
    {ParameterValueType::Double, "Double"},
    {ParameterValueType::Float, "Float"},
    {ParameterValueType::Bool, "Bool"},
    {ParameterValueType::String, "String"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(OutputFormat,
{
    {OutputFormat::JSON, "JSON"},
    {OutputFormat::XML, "XML"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(TimeUnit,
{
    {TimeUnit::Nanoseconds, "Nanoseconds"},
    {TimeUnit::Microseconds, "Microseconds"},
    {TimeUnit::Milliseconds, "Milliseconds"},
    {TimeUnit::Seconds, "Seconds"}
});

void to_json(json& j, const DimensionVector& vector);
void from_json(const json& j, DimensionVector& vector);

void from_json(const json& j, AddArgumentCommand& command);
void from_json(const json& j, AddKernelCommand& command);
void from_json(const json& j, CompilerOptionsCommand& command);
void from_json(const json& j, ConstraintCommand& command);
void from_json(const json& j, CreateTunerCommand& command);
void from_json(const json& j, ModifierCommand& command);
void from_json(const json& j, OutputCommand& command);
void from_json(const json& j, ParameterCommand& command);
void from_json(const json& j, SharedMemoryCommand& command);
void from_json(const json& j, TimeUnitCommand& command);

} // namespace ktt
