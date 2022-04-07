#pragma once

#include <json.hpp>

#include <TuningLoader/Commands/AddKernelCommand.h>
#include <TuningLoader/Commands/CreateTunerCommand.h>
#include <TuningLoader/Commands/ParameterCommand.h>

namespace ktt
{

using json = nlohmann::json;

void from_json(const json& j, AddKernelCommand& command);
void from_json(const json& j, CreateTunerCommand& command);
void from_json(const json& j, ParameterCommand& command);

} // namespace ktt
