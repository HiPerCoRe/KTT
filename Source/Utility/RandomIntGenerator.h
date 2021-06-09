#pragma once

#include <random>
#include <set>

namespace ktt
{

template <typename IntegerType>
class RandomIntGenerator
{
public:
    RandomIntGenerator();

    IntegerType Generate(const IntegerType min, const IntegerType max, const std::set<IntegerType>& excluded);

private:
    std::random_device m_Device;
    std::default_random_engine m_Engine;

    IntegerType GenerateNumberInRange(const IntegerType min, const IntegerType max);
};

} // namespace ktt

#include <Utility/RandomIntGenerator.inl>
