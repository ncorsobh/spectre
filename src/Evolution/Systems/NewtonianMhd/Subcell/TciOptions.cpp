// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Subcell/TciOptions.hpp"

#include <pup.h>

#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace NewtonianMhd::subcell {
TciOptions::TciOptions() = default;
TciOptions::TciOptions(const double minimum_density_in,
                       const double minimum_pressure_in,
                       const double safety_factor_for_magnetic_field_in,
                       const std::optional<double> magnetic_field_cutoff_in)
    : minimum_density(minimum_density_in),
      minimum_pressure(minimum_pressure_in),
      safety_factor_for_magnetic_field(safety_factor_for_magnetic_field_in),
      magnetic_field_cutoff(magnetic_field_cutoff_in) {}

void TciOptions::pup(PUP::er& p) {
  p | minimum_density;
  p | minimum_pressure;
  p | safety_factor_for_magnetic_field;
  p | magnetic_field_cutoff;
}
}  // namespace NewtonianMhd::subcell
