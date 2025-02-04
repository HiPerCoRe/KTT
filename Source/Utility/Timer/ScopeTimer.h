#pragma once

#include <KttTypes.h>

namespace ktt
{

template <typename Scope>
Nanoseconds RunScopeTimer(Scope scope);

} // namespace ktt

#include <Utility/Timer/ScopeTimer.inl>
