#include <kernel/kernel_parameter_pack.h>

namespace ktt
{

KernelParameterPack::KernelParameterPack() :
    name("")
{}

KernelParameterPack::KernelParameterPack(const std::string& name, const std::vector<std::string>& parameterNames) :
    name(name),
    parameterNames(parameterNames)
{}

const std::string& KernelParameterPack::getName() const
{
    return name;
}

const std::vector<std::string>& KernelParameterPack::getParameterNames() const
{
    return parameterNames;
}

size_t KernelParameterPack::getParameterCount() const
{
    return parameterNames.size();
}

bool KernelParameterPack::containsParameter(const std::string& parameterName) const
{
    for (const auto& name : parameterNames)
    {
        if (name == parameterName)
        {
            return true;
        }
    }

    return false;
}

bool KernelParameterPack::operator==(const KernelParameterPack& other) const
{
    return name == other.name;
}

bool KernelParameterPack::operator!=(const KernelParameterPack& other) const
{
    return !(*this == other);
}

} // namespace ktt
