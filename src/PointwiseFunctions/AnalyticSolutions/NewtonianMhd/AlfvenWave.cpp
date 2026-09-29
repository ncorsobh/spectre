// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/AlfvenWave.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace NewtonianMhd::Solutions {
namespace {
constexpr double two_pi = 2.0 * M_PI;

double magnitude_of(const std::array<double, 3>& vector) {
  return sqrt(square(vector[0]) + square(vector[1]) + square(vector[2]));
}

std::array<double, 3> cross(const std::array<double, 3>& lhs,
                            const std::array<double, 3>& rhs) {
  return {{(lhs[1] * rhs[2]) - (lhs[2] * rhs[1]),
           (lhs[2] * rhs[0]) - (lhs[0] * rhs[2]),
           (lhs[0] * rhs[1]) - (lhs[1] * rhs[0])}};
}

void normalize(const gsl::not_null<std::array<double, 3>*> vector) {
  const double norm = magnitude_of(*vector);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(*vector, i) /= norm;
  }
}
}  // namespace

AlfvenWave::AlfvenWave(const std::array<double, 3>& wavevector,
                       const double background_density,
                       const double background_pressure,
                       const double parallel_magnetic_field,
                       const double amplitude, const double adiabatic_index,
                       const Options::Context& context)
    : wavevector_(wavevector),
      background_density_(background_density),
      background_pressure_(background_pressure),
      parallel_magnetic_field_(parallel_magnetic_field),
      amplitude_(amplitude),
      equation_of_state_(adiabatic_index) {
  const double wavevector_magnitude = magnitude_of(wavevector_);
  if (wavevector_magnitude == 0.0) {
    PARSE_ERROR(context, "The WaveVector must not be zero.");
  }
  if (adiabatic_index <= 1.0) {
    PARSE_ERROR(context, "AdiabaticIndex (" << adiabatic_index
                                            << ") must be larger than 1.");
  }

  std::array<double, 3> propagation_direction = wavevector_;
  normalize(make_not_null(&propagation_direction));

  // Build a right-handed triad by crossing with whichever axis is least
  // aligned with the propagation direction, which keeps the cross product well
  // conditioned.
  size_t least_aligned_axis = 0;
  for (size_t i = 1; i < 3; ++i) {
    if (std::abs(gsl::at(propagation_direction, i)) <
        std::abs(gsl::at(propagation_direction, least_aligned_axis))) {
      least_aligned_axis = i;
    }
  }
  std::array<double, 3> axis{{0.0, 0.0, 0.0}};
  gsl::at(axis, least_aligned_axis) = 1.0;

  first_transverse_direction_ = cross(propagation_direction, axis);
  normalize(make_not_null(&first_transverse_direction_));
  second_transverse_direction_ =
      cross(propagation_direction, first_transverse_direction_);

  angular_frequency_ = two_pi * wavevector_magnitude * alfven_speed();
}

AlfvenWave::AlfvenWave(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> AlfvenWave::get_clone()
    const {
  return std::make_unique<AlfvenWave>(*this);
}

double AlfvenWave::alfven_speed() const {
  return parallel_magnetic_field_ / sqrt(background_density_);
}

void AlfvenWave::pup(PUP::er& p) {
  InitialData::pup(p);
  p | wavevector_;
  p | first_transverse_direction_;
  p | second_transverse_direction_;
  p | background_density_;
  p | background_pressure_;
  p | parallel_magnetic_field_;
  p | amplitude_;
  p | angular_frequency_;
  p | equation_of_state_;
}

DataVector AlfvenWave::phase(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                             const double t) const {
  DataVector result(get<0>(x).size(), -angular_frequency_ * t);
  for (size_t i = 0; i < 3; ++i) {
    result += two_pi * gsl::at(wavevector_, i) * x.get(i);
  }
  return result;
}

tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataVector>>
AlfvenWave::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x, const double /*t*/,
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>> /*meta*/) const {
  return {make_with_value<Scalar<DataVector>>(x, background_density_)};
}

tuples::TaggedTuple<hydro::Tags::Pressure<DataVector>> AlfvenWave::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x, const double /*t*/,
    tmpl::list<hydro::Tags::Pressure<DataVector>> /*meta*/) const {
  return {make_with_value<Scalar<DataVector>>(x, background_pressure_)};
}

tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataVector>>
AlfvenWave::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x, const double /*t*/,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataVector>> /*meta*/)
    const {
  return {make_with_value<Scalar<DataVector>>(
      x,
      get(equation_of_state_.specific_internal_energy_from_density_and_pressure(
          Scalar<double>{background_density_},
          Scalar<double>{background_pressure_})))};
}

tuples::TaggedTuple<
    hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>
AlfvenWave::variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                      const double t,
                      tmpl::list<hydro::Tags::SpatialVelocity<
                          DataVector, 3, Frame::Inertial>> /*meta*/) const {
  const DataVector wave_phase = phase(x, t);
  const DataVector cos_phase = cos(wave_phase);
  const DataVector sin_phase = sin(wave_phase);
  const double velocity_amplitude = -amplitude_ * alfven_speed();

  auto velocity =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    velocity.get(i) = velocity_amplitude *
                      ((cos_phase * gsl::at(first_transverse_direction_, i)) +
                       (sin_phase * gsl::at(second_transverse_direction_, i)));
  }
  return {std::move(velocity)};
}

tuples::TaggedTuple<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>
AlfvenWave::variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                      const double t,
                      tmpl::list<hydro::Tags::MagneticField<
                          DataVector, 3, Frame::Inertial>> /*meta*/) const {
  const DataVector wave_phase = phase(x, t);
  const DataVector cos_phase = cos(wave_phase);
  const DataVector sin_phase = sin(wave_phase);
  const double transverse_amplitude = amplitude_ * parallel_magnetic_field_;

  std::array<double, 3> propagation_direction = wavevector_;
  normalize(make_not_null(&propagation_direction));

  auto magnetic_field =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    magnetic_field.get(i) =
        (parallel_magnetic_field_ * gsl::at(propagation_direction, i)) +
        (transverse_amplitude *
         ((cos_phase * gsl::at(first_transverse_direction_, i)) +
          (sin_phase * gsl::at(second_transverse_direction_, i))));
  }
  return {std::move(magnetic_field)};
}

tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataVector>>
AlfvenWave::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x, const double /*t*/,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataVector>> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

bool operator==(const AlfvenWave& lhs, const AlfvenWave& rhs) {
  return lhs.wavevector_ == rhs.wavevector_ and
         lhs.background_density_ == rhs.background_density_ and
         lhs.background_pressure_ == rhs.background_pressure_ and
         lhs.parallel_magnetic_field_ == rhs.parallel_magnetic_field_ and
         lhs.amplitude_ == rhs.amplitude_ and
         lhs.equation_of_state_ == rhs.equation_of_state_;
}

bool operator!=(const AlfvenWave& lhs, const AlfvenWave& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID AlfvenWave::my_PUP_ID = 0;  // NOLINT

}  // namespace NewtonianMhd::Solutions
