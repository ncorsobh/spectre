// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Rusanov.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::BoundaryCorrections {
template <bool UseBackgroundMagneticField = false>
using standard_boundary_corrections =
    tmpl::list<Hll<UseBackgroundMagneticField>,
               Rusanov<UseBackgroundMagneticField>>;
}  // namespace NewtonianMhd::BoundaryCorrections
