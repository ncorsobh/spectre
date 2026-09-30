// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace NewtonianMhd::BoundaryConditions {
BoundaryCondition::BoundaryCondition(CkMigrateMessage* const msg)
    : domain::BoundaryConditions::BoundaryCondition(msg) {}

void BoundaryCondition::pup(PUP::er& p) {
  domain::BoundaryConditions::BoundaryCondition::pup(p);
}

#define INSTANTIATION(r, data)
INSTANTIATION(~, ~)

#undef INSTANTIATION
}  // namespace NewtonianMhd::BoundaryConditions
