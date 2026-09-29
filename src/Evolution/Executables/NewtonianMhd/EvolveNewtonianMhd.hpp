// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Executables/NewtonianMhd/NewtonianMhdBase.hpp"
#include "Evolution/Systems/NewtonianMhd/AllSolutions.hpp"

/// Standard Newtonian MHD: the whole magnetic field is evolved.
using EvolutionMetavars =
    NewtonianMhdMetavars<NewtonianMhd::InitialData::initial_data_list<3>,
                         false>;
