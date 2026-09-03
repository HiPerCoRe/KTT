#include <type_traits>

#include <Utility/RandomIntGenerator.h>

namespace ktt
{

template <typename IntegerType>
RandomIntGenerator<IntegerType>::RandomIntGenerator() :
    m_Engine(std::random_device{}())
{
    static_assert(std::is_integral_v<IntegerType>, "Only integer types are supported");
}

template <typename IntegerType>
RandomIntGenerator<IntegerType>::RandomIntGenerator(const uint64_t seed) :
    m_Engine(static_cast<std::default_random_engine::result_type>(seed))
{
    static_assert(std::is_integral_v<IntegerType>, "Only integer types are supported");
}

template <typename IntegerType>
IntegerType RandomIntGenerator<IntegerType>::Generate(const IntegerType min, const IntegerType max,
    const std::set<IntegerType>& excluded)
{
    IntegerType number = GenerateNumberInRange(min, max - static_cast<IntegerType>(excluded.size()));

    for (const auto excludedNumber : excluded)
    {
        if (number >= excludedNumber)
        {
            ++number;
        }
    }

    return number;
}

template <typename IntegerType>
IntegerType RandomIntGenerator<IntegerType>::GenerateNumberInRange(const IntegerType min, const IntegerType max)
{
    const IntegerType number = m_Engine();
    const IntegerType result = min + (number % (max - min + 1));
    return result;
}

} // namespace ktt
