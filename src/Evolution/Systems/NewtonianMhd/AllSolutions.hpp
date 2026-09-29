// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticData/NewtonianMhd/BrioWu.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/AlfvenWave.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::InitialData {
/// The initial data that can be used with the Newtonian MHD system.
///
/// The conducting-sphere problem is three dimensional, so no initial data are
/// available in lower dimensions yet.
template <size_t Dim>
using initial_data_list = tmpl::conditional_t<
    Dim == 3,
    tmpl::list<AnalyticData::BrioWu, Solutions::AlfvenWave>,
    tmpl::list<>>;

/// The initial data whose magnetic field can be split into a static background
/// and an evolved perturbation.
template <size_t Dim>
using background_magnetic_field_initial_data_list =
    tmpl::conditional_t<Dim == 3, tmpl::list<>,
                        tmpl::list<>>;
}  // namespace NewtonianMhd::InitialData
