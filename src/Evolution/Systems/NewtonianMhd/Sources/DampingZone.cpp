// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Sources/DampingZone.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"

namespace NewtonianMhd::Sources {
template <bool UseBackgroundMagneticField>
DampingZone<UseBackgroundMagneticField>::DampingZone(
    const double sponge_inner_radius, const double sponge_outer_radius,
    const double damping_timescale, const double asymptotic_velocity,
    const double background_density, const double background_pressure,
    const Options::Context& context)
    : sponge_inner_radius_(sponge_inner_radius),
      sponge_outer_radius_(sponge_outer_radius),
      damping_timescale_(damping_timescale),
      asymptotic_velocity_(asymptotic_velocity),
      background_density_(background_density),
      background_pressure_(background_pressure) {
  if (sponge_outer_radius_ <= sponge_inner_radius_) {
    PARSE_ERROR(context, "SpongeOuterRadius ("
                             << sponge_outer_radius_
                             << ") must be larger than SpongeInnerRadius ("
                             << sponge_inner_radius_ << ").");
  }
  if (damping_timescale_ <= 0.0) {
    PARSE_ERROR(context, "DampingTimescale (" << damping_timescale_
                                              << ") must be positive.");
  }
  if (background_density_ <= 0.0) {
    PARSE_ERROR(context, "BackgroundDensity (" << background_density_
                                               << ") must be positive.");
  }
  if (background_pressure_ <= 0.0) {
    PARSE_ERROR(context, "BackgroundPressure (" << background_pressure_
                                                << ") must be positive.");
  }
}

template <bool UseBackgroundMagneticField>
DampingZone<UseBackgroundMagneticField>::DampingZone(CkMigrateMessage* msg)
    : Source<UseBackgroundMagneticField>{msg} {}

template <bool UseBackgroundMagneticField>
void DampingZone<UseBackgroundMagneticField>::pup(PUP::er& p) {
  Source<UseBackgroundMagneticField>::pup(p);
  p | sponge_inner_radius_;
  p | sponge_outer_radius_;
  p | damping_timescale_;
  p | asymptotic_velocity_;
  p | background_density_;
  p | background_pressure_;
}

template <bool UseBackgroundMagneticField>
auto DampingZone<UseBackgroundMagneticField>::get_clone() const
    -> std::unique_ptr<Source<UseBackgroundMagneticField>> {
  return std::make_unique<DampingZone<UseBackgroundMagneticField>>(*this);
}

template <bool UseBackgroundMagneticField>
void DampingZone<UseBackgroundMagneticField>::operator()(
    const gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> source_energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> source_divergence_cleaning_field,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, 3>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, 3>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field,
    const tnsr::I<DataVector, 3>& /*velocity*/,
    const Scalar<DataVector>& /*pressure*/,
    BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
    /*background_magnetic_field*/,
    const EquationsOfState::EquationOfState<false, 2>& eos,
    const tnsr::I<DataVector, 3>& coords, const double /*time*/) const {
  const DataVector radius = get(magnitude(coords));
  const size_t num_points = radius.size();

  const double target_specific_internal_energy =
      get(eos.specific_internal_energy_from_density_and_pressure(
          Scalar<double>{background_density_},
          Scalar<double>{background_pressure_}));
  const double target_momentum_density =
      background_density_ * asymptotic_velocity_;
  const double target_energy_density =
      background_density_ *
      (target_specific_internal_energy + 0.5 * square(asymptotic_velocity_));

  for (size_t point = 0; point < num_points; ++point) {
    const double damping_rate =
        smoothstep<1>(sponge_inner_radius_, sponge_outer_radius_,
                      radius[point]) /
        damping_timescale_;
    if (damping_rate == 0.0) {
      continue;
    }

    get(*source_mass_density_cons)[point] -=
        damping_rate * (get(mass_density_cons)[point] - background_density_);
    for (size_t i = 0; i < 3; ++i) {
      // The undisturbed wind blows along the last coordinate axis.
      const double target = i == 3 - 1 ? target_momentum_density : 0.0;
      source_momentum_density->get(i)[point] -=
          damping_rate * (momentum_density.get(i)[point] - target);
      source_magnetic_field->get(i)[point] -=
          damping_rate * magnetic_field.get(i)[point];
    }
    get(*source_energy_density)[point] -=
        damping_rate * (get(energy_density)[point] - target_energy_density);
    get(*source_divergence_cleaning_field)[point] -=
        damping_rate * get(divergence_cleaning_field)[point];
  }
}

template <bool UseBackgroundMagneticField>
bool operator==(const DampingZone<UseBackgroundMagneticField>& lhs,
                const DampingZone<UseBackgroundMagneticField>& rhs) {
  return lhs.sponge_inner_radius_ == rhs.sponge_inner_radius_ and
         lhs.sponge_outer_radius_ == rhs.sponge_outer_radius_ and
         lhs.damping_timescale_ == rhs.damping_timescale_ and
         lhs.asymptotic_velocity_ == rhs.asymptotic_velocity_ and
         lhs.background_density_ == rhs.background_density_ and
         lhs.background_pressure_ == rhs.background_pressure_;
}

template <bool UseBackgroundMagneticField>
bool operator!=(const DampingZone<UseBackgroundMagneticField>& lhs,
                const DampingZone<UseBackgroundMagneticField>& rhs) {
  return not(lhs == rhs);
}

template <bool UseBackgroundMagneticField>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID DampingZone<UseBackgroundMagneticField>::my_PUP_ID = 0;

#define USE_BG(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                    \
  template class DampingZone<USE_BG(data)>;                       \
  template bool operator==(const DampingZone<USE_BG(data)>& lhs,  \
                           const DampingZone<USE_BG(data)>& rhs); \
  template bool operator!=(const DampingZone<USE_BG(data)>& lhs,  \
                           const DampingZone<USE_BG(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (true, false))

#undef INSTANTIATION
#undef USE_BG
}  // namespace NewtonianMhd::Sources
