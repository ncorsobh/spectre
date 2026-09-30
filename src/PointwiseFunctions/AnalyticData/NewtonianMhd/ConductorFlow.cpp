// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/NewtonianMhd/ConductorFlow.hpp"

#include <array>
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

ConductorFlow::ConductorFlow(
    const double adiabatic_index, const double background_density,
    const double background_pressure, const double asymptotic_velocity,
    const double magnetic_field_strength, const double conductor_internal_field,
    const double moment_tilt_angle, const double moment_azimuthal,
    const double conductor_radius, const bool magnetized,
    const Options::Context& context)
    : background_density_(background_density),
      background_pressure_(background_pressure),
      asymptotic_velocity_(asymptotic_velocity),
      magnetic_field_strength_(magnetic_field_strength),
      conductor_internal_field_(conductor_internal_field),
      moment_tilt_angle_(moment_tilt_angle),
      moment_azimuthal_(moment_azimuthal),
      conductor_radius_(conductor_radius),
      magnetized_(magnetized),
      equation_of_state_(adiabatic_index) {
  if (adiabatic_index <= 1.0) {
    PARSE_ERROR(context, "AdiabaticIndex (" << adiabatic_index
                                            << ") must be larger than 1.");
  }
  if (conductor_radius_ <= 0.0) {
    PARSE_ERROR(context, "ConductorRadius (" << conductor_radius_
                                             << ") must be positive.");
  }
}

ConductorFlow::ConductorFlow(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> ConductorFlow::get_clone()
    const {
  return std::make_unique<ConductorFlow>(*this);
}

void ConductorFlow::pup(PUP::er& p) {
  InitialData::pup(p);
  p | background_density_;
  p | background_pressure_;
  p | asymptotic_velocity_;
  p | magnetic_field_strength_;
  p | conductor_internal_field_;
  p | moment_tilt_angle_;
  p | moment_azimuthal_;
  p | conductor_radius_;
  p | magnetized_;
  p | equation_of_state_;
}

tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataVector>>
ConductorFlow::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>> /*meta*/) const {
  return {make_with_value<Scalar<DataVector>>(x, background_density_)};
}

tuples::TaggedTuple<hydro::Tags::Pressure<DataVector>> ConductorFlow::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::Pressure<DataVector>> /*meta*/) const {
  return {make_with_value<Scalar<DataVector>>(x, background_pressure_)};
}

tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataVector>>
ConductorFlow::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
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
ConductorFlow::variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                         tmpl::list<hydro::Tags::SpatialVelocity<
                             DataVector, 3, Frame::Inertial>> /*meta*/) const {
  auto velocity =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0);
  const size_t num_points = get<0>(x).size();
  for (size_t point = 0; point < num_points; ++point) {
    const double x_coord = get<0>(x)[point];
    const double y_coord = get<1>(x)[point];
    const double z_coord = get<2>(x)[point];
    const double radius =
        sqrt(square(x_coord) + square(y_coord) + square(z_coord));
    if (radius <= conductor_radius_) {
      continue;
    }
    const double cos_theta = z_coord / radius;
    const double sin_theta = sqrt(square(x_coord) + square(y_coord)) / radius;
    const double radius_ratio = conductor_radius_ / radius;

    double velocity_r = 0.0;
    double velocity_theta = 0.0;
    if (magnetized_) {
      velocity_r = asymptotic_velocity_ * cos_theta *
                   (1.0 - (1.5 * radius_ratio) + (0.5 * cube(radius_ratio)));
      velocity_theta =
          -asymptotic_velocity_ * sin_theta *
          (1.0 - (0.75 * radius_ratio) - (0.25 * cube(radius_ratio)));
    } else {
      velocity_r =
          asymptotic_velocity_ * cos_theta * (1.0 - cube(radius_ratio));
      velocity_theta = -asymptotic_velocity_ * sin_theta *
                       (1.0 + (0.5 * cube(radius_ratio)));
    }

    // The azimuthal unit vectors are degenerate on the axis, where sin_theta
    // vanishes and the theta-component of the velocity vanishes with it.
    const double cos_phi =
        sin_theta == 0.0 ? 1.0 : x_coord / (radius * sin_theta);
    const double sin_phi =
        sin_theta == 0.0 ? 0.0 : y_coord / (radius * sin_theta);

    get<0>(velocity)[point] = (velocity_r * sin_theta * cos_phi) +
                              (velocity_theta * cos_theta * cos_phi);
    get<1>(velocity)[point] = (velocity_r * sin_theta * sin_phi) +
                              (velocity_theta * cos_theta * sin_phi);
    get<2>(velocity)[point] =
        (velocity_r * cos_theta) - (velocity_theta * sin_theta);
  }
  return {std::move(velocity)};
}

tnsr::I<DataVector, 3, Frame::Inertial> ConductorFlow::static_magnetic_field(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
  auto field = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0);
  const double half_radius_cubed = 0.5 * cube(conductor_radius_);
  const std::array<double, 3> reaction_moment{
      {-half_radius_cubed * magnetic_field_strength_, 0.0, 0.0}};
  const std::array<double, 3> moment_direction{
      {cos(moment_tilt_angle_),
       -sin(moment_tilt_angle_) * sin(moment_azimuthal_),
       sin(moment_tilt_angle_) * cos(moment_azimuthal_)}};
  std::array<double, 3> conductor_moment{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(conductor_moment, i) = half_radius_cubed *
                                   conductor_internal_field_ *
                                   gsl::at(moment_direction, i);
  }

  const size_t num_points = get<0>(x).size();
  for (size_t point = 0; point < num_points; ++point) {
    const std::array<double, 3> position{
        {get<0>(x)[point], get<1>(x)[point], get<2>(x)[point]}};
    const double radius =
        sqrt(square(position[0]) + square(position[1]) + square(position[2]));

    if (radius < conductor_radius_) {
      // Uniform interior field of a uniformly magnetized sphere.
      for (size_t i = 0; i < 3; ++i) {
        field.get(i)[point] =
            2.0 * gsl::at(conductor_moment, i) / cube(conductor_radius_);
      }
      continue;
    }

    std::array<double, 3> radial_unit_vector{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(radial_unit_vector, i) = gsl::at(position, i) / radius;
    }
    const double one_over_radius_cubed = 1.0 / cube(radius);

    double reaction_dot_radial = 0.0;
    double conductor_dot_radial = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      reaction_dot_radial +=
          gsl::at(reaction_moment, i) * gsl::at(radial_unit_vector, i);
      conductor_dot_radial +=
          gsl::at(conductor_moment, i) * gsl::at(radial_unit_vector, i);
    }

    for (size_t i = 0; i < 3; ++i) {
      const double reaction_dipole =
          (((3.0 * reaction_dot_radial) * gsl::at(radial_unit_vector, i)) -
           gsl::at(reaction_moment, i)) *
          one_over_radius_cubed;
      const double conductor_dipole =
          (((3.0 * conductor_dot_radial) * gsl::at(radial_unit_vector, i)) -
           gsl::at(conductor_moment, i)) *
          one_over_radius_cubed;
      field.get(i)[point] = reaction_dipole + conductor_dipole;
    }
    // The uniform piece points along +x.
    get<0>(field)[point] += magnetic_field_strength_;
  }
  return field;
}

tuples::TaggedTuple<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>
ConductorFlow::variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                         tmpl::list<hydro::Tags::MagneticField<
                             DataVector, 3, Frame::Inertial>> /*meta*/) const {
  return {static_magnetic_field(x)};
}

tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataVector>>
ConductorFlow::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataVector>> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<NewtonianMhd::Tags::BackgroundMagneticFieldVolume<>>
ConductorFlow::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<NewtonianMhd::Tags::BackgroundMagneticFieldVolume<>> /*meta*/)
    const {
  return {static_magnetic_field(x)};
}

tuples::TaggedTuple<NewtonianMhd::Tags::MagneticFieldCons<>>
ConductorFlow::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<NewtonianMhd::Tags::MagneticFieldCons<>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0)};
}

bool operator==(const ConductorFlow& lhs, const ConductorFlow& rhs) {
  return lhs.background_density_ == rhs.background_density_ and
         lhs.background_pressure_ == rhs.background_pressure_ and
         lhs.asymptotic_velocity_ == rhs.asymptotic_velocity_ and
         lhs.magnetic_field_strength_ == rhs.magnetic_field_strength_ and
         lhs.conductor_internal_field_ == rhs.conductor_internal_field_ and
         lhs.moment_tilt_angle_ == rhs.moment_tilt_angle_ and
         lhs.moment_azimuthal_ == rhs.moment_azimuthal_ and
         lhs.conductor_radius_ == rhs.conductor_radius_ and
         lhs.magnetized_ == rhs.magnetized_ and
         lhs.equation_of_state_ == rhs.equation_of_state_;
}

bool operator!=(const ConductorFlow& lhs, const ConductorFlow& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID ConductorFlow::my_PUP_ID = 0;  // NOLINT

}  // namespace NewtonianMhd::AnalyticData
