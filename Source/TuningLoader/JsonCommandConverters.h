#pragma once

#include <json.hpp>

#include <TuningLoader/Commands/AddArgumentCommand.h>
#include <TuningLoader/Commands/AddKernelCommand.h>
#include <TuningLoader/Commands/ConstraintCommand.h>
#include <TuningLoader/Commands/CreateTunerCommand.h>
#include <TuningLoader/Commands/ModifierCommand.h>
#include <TuningLoader/Commands/OutputCommand.h>
#include <TuningLoader/Commands/ParameterCommand.h>
#include <TuningLoader/Commands/TimeUnitCommand.h>

namespace ktt
{

using json = nlohmann::json;

void from_json(const json& j, AddArgumentCommand& command);
void from_json(const json& j, AddKernelCommand& command);
void from_json(const json& j, ConstraintCommand& command);
void from_json(const json& j, CreateTunerCommand& command);
void from_json(const json& j, ModifierCommand& command);
void from_json(const json& j, OutputCommand& command);
void from_json(const json& j, ParameterCommand& command);
void from_json(const json& j, TimeUnitCommand& command);

} // namespace ktt
