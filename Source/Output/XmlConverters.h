#pragma once

#include <string>
#include <pugixml.hpp>

#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/KernelResult.h>
#include <Output/TunerMetadata.h>
#include <KttTypes.h>

namespace ktt
{

std::string ComputeApiToString(const ComputeApi api);
std::string TimeUnitToString(const TimeUnit unit);
std::string ResultStatusToString(const ResultStatus status);
std::string ProfilingCounterTypeToString(const ProfilingCounterType type);

ComputeApi ComputeApiFromString(const std::string& string);
TimeUnit TimeUnitFromString(const std::string& string);
ResultStatus ResultStatusFromString(const std::string& string);
ProfilingCounterType ProfilingCounterTypeFromString(const std::string& string);

void AppendMetadata(pugi::xml_node parent, const TunerMetadata& metadata);
TunerMetadata ParseMetadata(const pugi::xml_node node);

void AppendUserData(pugi::xml_node parent, const UserData& data);
UserData ParseUserData(const pugi::xml_node node);

void AppendPair(pugi::xml_node parent, const ParameterPair& pair);
ParameterPair ParsePair(const pugi::xml_node node);

void AppendVector(pugi::xml_node parent, const DimensionVector& vector, const std::string& tag);
DimensionVector ParseVector(const pugi::xml_node node, const std::string& tag);

void AppendConfiguration(pugi::xml_node parent, const KernelConfiguration& configuration);
KernelConfiguration ParseConfiguration(const pugi::xml_node node);

void AppendCounter(pugi::xml_node parent, const KernelProfilingCounter& counter);
KernelProfilingCounter ParseCounter(const pugi::xml_node node);

void AppendCompilationData(pugi::xml_node parent, const KernelCompilationData& data);
KernelCompilationData ParseCompilationData(const pugi::xml_node node);

void AppendProfilingData(pugi::xml_node parent, const KernelProfilingData& data);
KernelProfilingData ParseProfilingData(const pugi::xml_node node);

void AppendComputationResult(pugi::xml_node parent, const ComputationResult& result);
ComputationResult ParseComputationResult(const pugi::xml_node node);

void AppendKernelResult(pugi::xml_node parent, const KernelResult& result);
KernelResult ParseKernelResult(const pugi::xml_node node);

} // namespace ktt
