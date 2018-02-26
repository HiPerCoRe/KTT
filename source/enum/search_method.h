/** @file search_method.h
  * Definition of enum for search method used to explore configuration space during kernel tuning.
  */
#pragma once

namespace ktt
{

/** @enum SearchMethod
  * Enum for search method used to explore configuration space during kernel tuning.
  */
enum class SearchMethod
{
    /** All kernel configurations will be explored. No additional search parameters are needed.
      */
    FullSearch,

    /** Explores random fraction of kernel configurations. The fraction size is controlled with parameter.
      */
    RandomSearch,

    /** Explores fraction of kernel configurations using simulated annealing method. The fraction size is controlled with parameter.
      * Additional parameter specifies maximum temperature.
      */
    Annealing,

    /** Explores fraction of kernel configurations using Markov chain Monte Carlo method. The fraction size is controlled with parameter.
      */
    MCMC
};

} // namespace ktt
