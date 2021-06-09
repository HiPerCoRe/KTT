#include <KernelRunner/KernelActivator.h>

namespace ktt
{

KernelActivator::KernelActivator(ComputeLayer& computeLayer, const KernelId id) :
    m_ComputeLayer(computeLayer)
{
    computeLayer.SetActiveKernel(id);
}

KernelActivator::~KernelActivator()
{
    m_ComputeLayer.ClearActiveKernel();
}

} // namespace ktt
