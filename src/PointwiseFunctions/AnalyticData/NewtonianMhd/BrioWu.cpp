// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/NewtonianMhd/BrioWu.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace NewtonianMhd::AnalyticData {

BrioWu::BrioWu(const double adiabatic_index, const double left_density,
               const double left_pressure,
               const double left_transverse_magnetic_field,
               const double right_density, const double right_pressure,
               const double right_transverse_magnetic_field,
               const double parallel_magnetic_field,
               const Options::Context& context)
    : left_density_(left_density),
      left_pressure_(left_pressure),
      left_transverse_magnetic_field_(left_transverse_magnetic_field),
      right_density_(right_density),
      right_pressure_(right_pressure),
      right_transverse_magnetic_field_(right_transverse_magnetic_field),
      parallel_magnetic_field_(parallel_magnetic_field),
      equation_of_state_(adiabatic_index) {
  if (adiabatic_index <= 1.0) {
    PARSE_ERROR(context, "AdiabaticIndex (" << adiabatic_index
                                            << ") must be larger than 1.");
  }
}

BrioWu::BrioWu(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> BrioWu::get_clone()
    const {
  return std::make_unique<BrioWu>(*this);
}

void BrioWu::pup(PUP::er& p) {
  InitialData::pup(p);
  p | left_density_;
  p | left_pressure_;
  p | left_transverse_magnetic_field_;
  p | right_density_;
  p | right_pressure_;
  p | right_transverse_magnetic_field_;
  p | parallel_magnetic_field_;
  p | equation_of_state_;
}

tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataVector>> BrioWu::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>> /*meta*/) const {
  auto density = make_with_value<Scalar<DataVector>>(x, left_density_);
  for (size_t i = 0; i < get<0>(x).size(); ++i) {
    if (get<0>(x)[i] > 0.0) {
      get(density)[i] = right_density_;
    }
  }
  return {std::move(density)};
}

tuples::TaggedTuple<hydro::Tags::Pressure<DataVector>> BrioWu::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::Pressure<DataVector>> /*meta*/) const {
  auto pressure = make_with_value<Scalar<DataVector>>(x, left_pressure_);
  for (size_t i = 0; i < get<0>(x).size(); ++i) {
    if (get<0>(x)[i] > 0.0) {
      get(pressure)[i] = right_pressure_;
    }
  }
  return {std::move(pressure)};
}

tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataVector>>
BrioWu::variables(
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
BrioWu::variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                  tmpl::list<hydro::Tags::SpatialVelocity<
                      DataVector, 3, Frame::Inertial>> /*meta*/) {
  return {make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0)};
}

tuples::TaggedTuple<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>
BrioWu::variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                  tmpl::list<hydro::Tags::MagneticField<
                      DataVector, 3, Frame::Inertial>> /*meta*/) const {
  auto magnetic_field =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0);
  get<0>(magnetic_field) = parallel_magnetic_field_;
  get<1>(magnetic_field) = left_transverse_magnetic_field_;
  for (size_t i = 0; i < get<0>(x).size(); ++i) {
    if (get<0>(x)[i] > 0.0) {
      get<1>(magnetic_field)[i] = right_transverse_magnetic_field_;
    }
  }
  return {std::move(magnetic_field)};
}

tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataVector>>
BrioWu::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataVector>> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

bool operator==(const BrioWu& lhs, const BrioWu& rhs) {
  return lhs.left_density_ == rhs.left_density_ and
         lhs.left_pressure_ == rhs.left_pressure_ and
         lhs.left_transverse_magnetic_field_ ==
             rhs.left_transverse_magnetic_field_ and
         lhs.right_density_ == rhs.right_density_ and
         lhs.right_pressure_ == rhs.right_pressure_ and
         lhs.right_transverse_magnetic_field_ ==
             rhs.right_transverse_magnetic_field_ and
         lhs.parallel_magnetic_field_ == rhs.parallel_magnetic_field_ and
         lhs.equation_of_state_ == rhs.equation_of_state_;
}

bool operator!=(const BrioWu& lhs, const BrioWu& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID BrioWu::my_PUP_ID = 0;  // NOLINT

}  // namespace NewtonianMhd::AnalyticData
