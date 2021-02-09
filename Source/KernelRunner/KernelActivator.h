#pragma once

#include <KernelRunner/ComputeLayer.h>
#include <KttTypes.h>

namespace ktt
{

class KernelActivator
{
public:
    explicit KernelActivator(ComputeLayer& computeLayer, const KernelId id);
    ~KernelActivator();

private:
    ComputeLayer& m_ComputeLayer;
};

} // namespace ktt
