// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/NewtonianMhd/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/Source.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::Sources {
/// All the available source terms.
template <bool UseBackgroundMagneticField = false>
using all_sources = tmpl::list<NoSource<UseBackgroundMagneticField>>;
}  // namespace NewtonianMhd::Sources
