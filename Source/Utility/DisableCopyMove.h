#pragma once

namespace ktt
{

class DisableCopy
{
public:
    DisableCopy() = default;

    DisableCopy(const DisableCopy&) = delete;
    DisableCopy& operator=(const DisableCopy&) = delete;

    DisableCopy(DisableCopy&&) = default;
    DisableCopy& operator=(DisableCopy&&) = default;
};

class DisableMove
{
public:
    DisableMove() = default;

    DisableMove(const DisableMove&) = default;
    DisableMove& operator=(const DisableMove&) = default;

    DisableMove(DisableMove&&) = delete;
    DisableMove& operator=(DisableMove&&) = delete;
};

class DisableCopyMove : public DisableCopy, public DisableMove
{
public:
    DisableCopyMove() = default;
};

} // namespace ktt
