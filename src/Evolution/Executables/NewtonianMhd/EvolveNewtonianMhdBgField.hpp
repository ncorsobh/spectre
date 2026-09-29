// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Executables/NewtonianMhd/NewtonianMhdBase.hpp"
#include "Evolution/Systems/NewtonianMhd/AllSolutions.hpp"

/// Newtonian MHD with background-field splitting: the static, curl-free and
/// divergence-free part of the initial magnetic field is held fixed in
/// \f$B_0\f$ and only the perturbation \f$B_1\f$ is evolved.
using EvolutionMetavars = NewtonianMhdMetavars<
    NewtonianMhd::InitialData::background_magnetic_field_initial_data_list<3>,
    true>;
