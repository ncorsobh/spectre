// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"

#include <pup.h>

#include "Utilities/GenerateInstantiations.hpp"

namespace NewtonianMhd::fd {
Reconstructor::Reconstructor(CkMigrateMessage* const msg) : PUP::able(msg) {}

void Reconstructor::pup(PUP::er& p) { PUP::able::pup(p); }

#define INSTANTIATION(r, data)
INSTANTIATION(~, ~)

#undef INSTANTIATION
}  // namespace NewtonianMhd::fd
