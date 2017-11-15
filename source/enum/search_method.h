/** @file search_method.h
  * @brief Definition of enum for search method used to explore configuration space during kernel tuning.
  */
#pragma once

namespace ktt
{

/** @enum SearchMethod
  * @brief Enum for search method used to explore configuration space during kernel tuning.
  */
enum class SearchMethod
{
    /** @brief All kernel configurations will be explored. No additional search parameters are needed.
      */
    FullSearch,

    /** @brief Explores random fraction of kernel configurations. The fraction size is controlled with parameter.
      */
    RandomSearch,

    /** @brief Explores fraction of kernel configurations using particle swarm optimization method. The fraction size is controlled with parameter.
      * Additional parameters specify swarm size and swarm influences.
      */
    PSO,

    /** @brief Explores fraction of kernel configurations using simulated annealing method. The fraction size is controlled with parameter.
      * Additional parameter specifies maximum temperature.
      */
    Annealing
};

} // namespace ktt
