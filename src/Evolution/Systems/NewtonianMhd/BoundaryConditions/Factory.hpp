// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/ConductorReflection.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/Reflection.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::BoundaryConditions {
/// Typelist of standard BoundaryConditions
template <bool UseBackgroundMagneticField = false>
using standard_boundary_conditions =
    tmpl::list<ConductorReflection<UseBackgroundMagneticField>,
               DemandOutgoingCharSpeeds<UseBackgroundMagneticField>,
               DirichletAnalytic<UseBackgroundMagneticField>,
               Reflection<UseBackgroundMagneticField>,
               domain::BoundaryConditions::Periodic<BoundaryCondition>>;
}  // namespace NewtonianMhd::BoundaryConditions
