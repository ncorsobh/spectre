// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/NewtonianMhd/OrszagTangVortex.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace NewtonianMhd::AnalyticData {

double OrszagTangVortex::Pressure::suggested_value() {
  return 1.0 / (4.0 * M_PI);
}

double OrszagTangVortex::MagneticFieldAmplitude::suggested_value() {
  return 1.0 / sqrt(4.0 * M_PI);
}

OrszagTangVortex::OrszagTangVortex(const double adiabatic_index,
                                   const double pressure,
                                   const double magnetic_field_amplitude,
                                   const Options::Context& context)
    : adiabatic_index_(adiabatic_index),
      pressure_(pressure),
      magnetic_field_amplitude_(magnetic_field_amplitude),
      equation_of_state_(adiabatic_index) {
  if (adiabatic_index <= 1.0) {
    PARSE_ERROR(context, "AdiabaticIndex (" << adiabatic_index
                                            << ") must be larger than 1.");
  }
}

OrszagTangVortex::OrszagTangVortex(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData>
OrszagTangVortex::get_clone() const {
  return std::make_unique<OrszagTangVortex>(*this);
}

void OrszagTangVortex::pup(PUP::er& p) {
  InitialData::pup(p);
  p | adiabatic_index_;
  p | pressure_;
  p | magnetic_field_amplitude_;
  p | equation_of_state_;
}

tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataVector>>
OrszagTangVortex::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>> /*meta*/) const {
  return {make_with_value<Scalar<DataVector>>(
      x, square(adiabatic_index_) * pressure_)};
}

tuples::TaggedTuple<hydro::Tags::Pressure<DataVector>>
OrszagTangVortex::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::Pressure<DataVector>> /*meta*/) const {
  return {make_with_value<Scalar<DataVector>>(x, adiabatic_index_ * pressure_)};
}

tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataVector>>
OrszagTangVortex::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataVector>> /*meta*/)
    const {
  return {equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<hydro::Tags::RestMassDensity<DataVector>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataVector>>{})),
      get<hydro::Tags::Pressure<DataVector>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataVector>>{})))};
}

tuples::TaggedTuple<
    hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>
OrszagTangVortex::variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                            tmpl::list<hydro::Tags::SpatialVelocity<
                                DataVector, 3, Frame::Inertial>> /*meta*/) {
  auto velocity =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0);
  get<0>(velocity) = -sin(2.0 * M_PI * get<1>(x));
  get<1>(velocity) = sin(2.0 * M_PI * get<0>(x));
  return {std::move(velocity)};
}

tuples::TaggedTuple<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>
OrszagTangVortex::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::MagneticField<DataVector, 3,
                                          Frame::Inertial>> /*meta*/) const {
  auto magnetic_field =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0);
  get<0>(magnetic_field) =
      -magnetic_field_amplitude_ * sin(2.0 * M_PI * get<1>(x));
  get<1>(magnetic_field) =
      magnetic_field_amplitude_ * sin(4.0 * M_PI * get<0>(x));
  return {std::move(magnetic_field)};
}

tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataVector>>
OrszagTangVortex::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataVector>> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

bool operator==(const OrszagTangVortex& lhs, const OrszagTangVortex& rhs) {
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.pressure_ == rhs.pressure_ and
         lhs.magnetic_field_amplitude_ == rhs.magnetic_field_amplitude_ and
         lhs.equation_of_state_ == rhs.equation_of_state_;
}

bool operator!=(const OrszagTangVortex& lhs, const OrszagTangVortex& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID OrszagTangVortex::my_PUP_ID = 0;  // NOLINT

}  // namespace NewtonianMhd::AnalyticData
