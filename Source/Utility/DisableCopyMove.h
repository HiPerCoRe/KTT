#pragma once

namespace ktt
{

class DisableCopy
{
public:
    DisableCopy() = default;
    DisableCopy(const DisableCopy&) = delete;
    void operator=(const DisableCopy&) = delete;
};

class DisableMove
{
public:
    DisableMove() = default;
    DisableMove(DisableMove&&) = delete;
    void operator=(DisableMove&&) = delete;
};

class DisableCopyMove : public DisableCopy, public DisableMove
{
public:
    DisableCopyMove() = default;
};

} // namespace ktt
