// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/NewtonianMhd/BrioWu.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianMhd/ConductorFlow.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianMhd/OrszagTangVortex.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/AlfvenWave.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::InitialData {
/// The initial data that can be used with the Newtonian MHD system.
using initial_data_list =
    tmpl::list<AnalyticData::BrioWu, AnalyticData::ConductorFlow,
               AnalyticData::OrszagTangVortex, Solutions::AlfvenWave>;

/// The initial data whose magnetic field can be split into a static background
/// and an evolved perturbation.
///
/// Such data must supply `Tags::BackgroundMagneticFieldVolume` holding the
/// static, curl-free and divergence-free part of the field.
using background_magnetic_field_initial_data_list =
    tmpl::list<AnalyticData::ConductorFlow, Solutions::AlfvenWave>;
}  // namespace NewtonianMhd::InitialData
