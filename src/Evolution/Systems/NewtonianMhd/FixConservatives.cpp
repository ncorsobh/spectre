// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/FixConservatives.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd {

FixConservatives::FixConservatives(
    const double minimum_density, const double cutoff_density,
    const double safety_factor_for_magnetic_field,
    const double safety_factor_for_momentum_density, const bool enable,
    const Options::Context& context)
    : minimum_density_(minimum_density),
      cutoff_density_(cutoff_density),
      one_minus_safety_factor_for_magnetic_field_(
          1.0 - safety_factor_for_magnetic_field),
      one_minus_safety_factor_for_momentum_density_(
          1.0 - safety_factor_for_momentum_density),
      enable_(enable) {
  if (minimum_density_ > cutoff_density_) {
    PARSE_ERROR(context, "The minimum density ("
                             << minimum_density_
                             << ") must be less than or equal to the cutoff "
                                "density ("
                             << cutoff_density_ << ").");
  }
  if (safety_factor_for_magnetic_field >= 1.0) {
    PARSE_ERROR(context, "SafetyFactorForB ("
                             << safety_factor_for_magnetic_field
                             << ") must be less than 1.");
  }
  if (safety_factor_for_momentum_density >= 1.0) {
    PARSE_ERROR(context, "SafetyFactorForS ("
                             << safety_factor_for_momentum_density
                             << ") must be less than 1.");
  }
}

void FixConservatives::pup(PUP::er& p) {
  p | minimum_density_;
  p | cutoff_density_;
  p | one_minus_safety_factor_for_magnetic_field_;
  p | one_minus_safety_factor_for_momentum_density_;
  p | enable_;
}

bool FixConservatives::operator()(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field_cons) const {
  if (not enable_) {
    return false;
  }
  bool needed_fixing = false;
  const size_t num_points = get(*mass_density_cons).size();
  for (size_t point = 0; point < num_points; ++point) {
    double& density = get(*mass_density_cons)[point];
    double& energy = get(*energy_density)[point];

    if (density < cutoff_density_) {
      density = minimum_density_;
      needed_fixing = true;
    }

    double magnetic_field_squared = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      magnetic_field_squared += square(magnetic_field_cons->get(i)[point]);
    }
    const double magnetic_field_bound =
        2.0 * one_minus_safety_factor_for_magnetic_field_ * energy;
    if (magnetic_field_squared > magnetic_field_bound) {
      // Rescaling rather than clipping keeps the direction of B.
      const double rescale =
          sqrt(std::max(magnetic_field_bound, 0.0) / magnetic_field_squared);
      for (size_t i = 0; i < 3; ++i) {
        magnetic_field_cons->get(i)[point] *= rescale;
      }
      magnetic_field_squared = std::max(magnetic_field_bound, 0.0);
      needed_fixing = true;
    }

    double momentum_density_squared = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      momentum_density_squared += square(momentum_density->get(i)[point]);
    }
    // |S|^2 <= 2 (1 - eps_S) rho (e - |B|^2/2) is equivalent to a
    // non-negative specific internal energy.
    const double momentum_density_bound =
        2.0 * one_minus_safety_factor_for_momentum_density_ * density *
        (energy - 0.5 * magnetic_field_squared);
    if (momentum_density_squared > momentum_density_bound) {
      const double rescale = sqrt(std::max(momentum_density_bound, 0.0) /
                                  momentum_density_squared);
      for (size_t i = 0; i < 3; ++i) {
        momentum_density->get(i)[point] *= rescale;
      }
      needed_fixing = true;
    }
  }
  return needed_fixing;
}

bool operator==(const FixConservatives& lhs, const FixConservatives& rhs) {
  return lhs.minimum_density_ == rhs.minimum_density_ and
         lhs.cutoff_density_ == rhs.cutoff_density_ and
         lhs.one_minus_safety_factor_for_magnetic_field_ ==
             rhs.one_minus_safety_factor_for_magnetic_field_ and
         lhs.one_minus_safety_factor_for_momentum_density_ ==
             rhs.one_minus_safety_factor_for_momentum_density_ and
         lhs.enable_ == rhs.enable_;
}

bool operator!=(const FixConservatives& lhs, const FixConservatives& rhs) {
  return not(lhs == rhs);
}

}  // namespace NewtonianMhd

#define INSTANTIATION(_, data)

INSTANTIATION(~, ~)

#undef INSTANTIATION
