// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianMhd/OrszagTangVortex.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {
using OrszagTangVortex = NewtonianMhd::AnalyticData::OrszagTangVortex;
using Density = hydro::Tags::RestMassDensity<DataVector>;
using Pressure = hydro::Tags::Pressure<DataVector>;
using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>;
using MagneticField =
    hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>;

constexpr double adiabatic_index = 5.0 / 3.0;
const double pressure_parameter = 1.0 / (4.0 * M_PI);
const double field_amplitude = 1.0 / sqrt(4.0 * M_PI);

OrszagTangVortex make_data() {
  return OrszagTangVortex{adiabatic_index, pressure_parameter, field_amplitude};
}

tnsr::I<DataVector, 3> sample_points() {
  tnsr::I<DataVector, 3> x{4_st};
  get<0>(x) = DataVector{0.0, 0.25, 0.5, 0.125};
  get<1>(x) = DataVector{0.0, 0.5, 0.25, 0.375};
  get<2>(x) = DataVector{0.0, 0.1, -0.2, 0.05};
  return x;
}

void test_profiles() {
  const auto data = make_data();
  const auto x = sample_points();
  const auto vars = data.variables(
      x, tmpl::list<Density, Pressure, Velocity, MagneticField>{});

  // Density and pressure are uniform, with rho = gamma P.
  const DataVector expected_pressure{4, adiabatic_index * pressure_parameter};
  CHECK_ITERABLE_APPROX(get(get<Pressure>(vars)), expected_pressure);
  CHECK_ITERABLE_APPROX(get(get<Density>(vars)),
                        DataVector{adiabatic_index * expected_pressure});

  const DataVector expected_vx = -sin(2.0 * M_PI * get<1>(x));
  const DataVector expected_vy = sin(2.0 * M_PI * get<0>(x));
  const DataVector zero{4, 0.0};
  CHECK_ITERABLE_APPROX(get<0>(get<Velocity>(vars)), expected_vx);
  CHECK_ITERABLE_APPROX(get<1>(get<Velocity>(vars)), expected_vy);
  CHECK_ITERABLE_APPROX(get<2>(get<Velocity>(vars)), zero);

  const DataVector expected_bx = -field_amplitude * sin(2.0 * M_PI * get<1>(x));
  const DataVector expected_by = field_amplitude * sin(4.0 * M_PI * get<0>(x));
  CHECK_ITERABLE_APPROX(get<0>(get<MagneticField>(vars)), expected_bx);
  CHECK_ITERABLE_APPROX(get<1>(get<MagneticField>(vars)), expected_by);
  CHECK_ITERABLE_APPROX(get<2>(get<MagneticField>(vars)), zero);
}

// Both the velocity and the magnetic field must be divergence-free, since
// div(v) and div(B) of the initial data are identically zero.
void test_fields_are_divergence_free() {
  const auto data = make_data();
  const double delta = 1.0e-6;
  const DataVector sample_x{0.13, 0.42, 0.77};
  const DataVector sample_y{0.31, 0.64, 0.05};

  const auto field_at = [&data, &sample_y](const DataVector& x_values,
                                           const DataVector& y_values) {
    tnsr::I<DataVector, 3> point{x_values.size()};
    get<0>(point) = x_values;
    get<1>(point) = y_values;
    get<2>(point) = DataVector{x_values.size(), 0.0};
    (void)sample_y;
    return data.variables(point, tmpl::list<Velocity, MagneticField>{});
  };

  const auto plus_x = field_at(sample_x + delta, sample_y);
  const auto minus_x = field_at(sample_x - delta, sample_y);
  const auto plus_y = field_at(sample_x, sample_y + delta);
  const auto minus_y = field_at(sample_x, sample_y - delta);

  const DataVector div_velocity =
      (get<0>(get<Velocity>(plus_x)) - get<0>(get<Velocity>(minus_x)) +
       get<1>(get<Velocity>(plus_y)) - get<1>(get<Velocity>(minus_y))) /
      (2.0 * delta);
  const DataVector div_magnetic_field = (get<0>(get<MagneticField>(plus_x)) -
                                         get<0>(get<MagneticField>(minus_x)) +
                                         get<1>(get<MagneticField>(plus_y)) -
                                         get<1>(get<MagneticField>(minus_y))) /
                                        (2.0 * delta);

  const Approx finite_difference = Approx::custom().epsilon(1.0e-8).scale(1.0);
  const DataVector zero{sample_x.size(), 0.0};
  CHECK_ITERABLE_CUSTOM_APPROX(div_velocity, zero, finite_difference);
  CHECK_ITERABLE_CUSTOM_APPROX(div_magnetic_field, zero, finite_difference);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.AnalyticData.OrszagTangVortex",
                  "[Unit][PointwiseFunctions]") {
  test_profiles();
  test_fields_are_divergence_free();

  const auto created =
      TestHelpers::test_factory_creation<evolution::initial_data::InitialData,
                                         OrszagTangVortex>(
          "OrszagTangVortex:\n"
          "  AdiabaticIndex: 1.6666666666666667\n"
          "  Pressure: 0.07957747154594767\n"
          "  MagneticFieldAmplitude: 0.28209479177387814\n");
  CHECK(dynamic_cast<const OrszagTangVortex&>(*created) == make_data());
  test_serialization(make_data());
}
