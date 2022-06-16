/** @file ProfileBasedSearcher.h
  * Searcher which explores configurations according to observed bottlenecks
  * and ML model created on historical data (on the same tuning space, but 
  * possibly different HW and input size). For more information, see
  * J. Filipovic et al. Using hardware performance counters to speed up 
  * autotuning convergence on GPUs. JPDC, vol. 160, 2021.
  */
#pragma once

#include <Ktt.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

void SetProfileBasedSearcher(ktt::Tuner& tuner, ktt::KernelId kernel, std::string model) {
    py::module_ searcher = py::module_::import("ProfileBasedSearcher");
    py::gil_scoped_acquire acquire;
    searcher.attr("executeSearcher")(&tuner, kernel, model);
}

