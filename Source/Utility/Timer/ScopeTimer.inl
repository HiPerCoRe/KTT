#pragma once

#include <Utility/Timer/ScopeTimer.h>
#include <Utility/Timer/Timer.h>

namespace ktt
{

template <typename Scope>
Nanoseconds RunScopeTimer(Scope scope)
{
    Timer timer;
    timer.Start();

    scope();

    timer.Stop();
    return timer.GetElapsedTime();
}

} // namespace ktt
