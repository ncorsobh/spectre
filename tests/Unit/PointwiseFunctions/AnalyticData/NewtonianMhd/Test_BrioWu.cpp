// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianMhd/BrioWu.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"

namespace {
using BrioWu = NewtonianMhd::AnalyticData::BrioWu;
using Density = hydro::Tags::RestMassDensity<DataVector>;
using Pressure = hydro::Tags::Pressure<DataVector>;
using SpecificInternalEnergy = hydro::Tags::SpecificInternalEnergy<DataVector>;
using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>;
using MagneticField =
    hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>;
using DivergenceCleaningField =
    hydro::Tags::DivergenceCleaningField<DataVector>;

constexpr double adiabatic_index = 2.0;
constexpr double left_density = 1.0;
constexpr double left_pressure = 1.0;
constexpr double left_transverse_field = 1.0;
constexpr double right_density = 0.125;
constexpr double right_pressure = 0.1;
constexpr double right_transverse_field = -1.0;
constexpr double parallel_field = 0.75;

BrioWu make_data() {
  return BrioWu{adiabatic_index,        left_density,  left_pressure,
                left_transverse_field,  right_density, right_pressure,
                right_transverse_field, parallel_field};
}

// Points either side of the discontinuity, plus a pair straddling it closely
// enough to catch an off-by-one in the comparison.
tnsr::I<DataVector, 3> sample_points() {
  tnsr::I<DataVector, 3> x{4_st};
  get<0>(x) = DataVector{-0.5, -1.0e-14, 1.0e-14, 0.5};
  get<1>(x) = DataVector{0.3, -0.2, 0.7, -0.4};
  get<2>(x) = DataVector{-0.1, 0.6, -0.8, 0.2};
  return x;
}

void test_states() {
  const auto data = make_data();
  const auto x = sample_points();
  const auto vars = data.variables(
      x, tmpl::list<Density, Pressure, SpecificInternalEnergy, Velocity,
                    MagneticField, DivergenceCleaningField>{});

  const DataVector expected_density{left_density, left_density, right_density,
                                    right_density};
  const DataVector expected_pressure{left_pressure, left_pressure,
                                     right_pressure, right_pressure};
  const DataVector expected_transverse_field{
      left_transverse_field, left_transverse_field, right_transverse_field,
      right_transverse_field};
  CHECK_ITERABLE_APPROX(get(get<Density>(vars)), expected_density);
  CHECK_ITERABLE_APPROX(get(get<Pressure>(vars)), expected_pressure);
  CHECK_ITERABLE_APPROX(get<1>(get<MagneticField>(vars)),
                        expected_transverse_field);

  // The normal field is what makes this an MHD rather than a hydrodynamic
  // Riemann problem, and must be continuous for div(B) = 0.
  const DataVector expected_parallel_field{4, parallel_field};
  CHECK_ITERABLE_APPROX(get<0>(get<MagneticField>(vars)),
                        expected_parallel_field);

  const DataVector zero{4, 0.0};
  CHECK_ITERABLE_APPROX(get<2>(get<MagneticField>(vars)), zero);
  CHECK_ITERABLE_APPROX(get(get<DivergenceCleaningField>(vars)), zero);
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_APPROX(get<Velocity>(vars).get(i), zero);
  }

  // An ideal fluid with gamma = 2 has e = P / rho.
  const DataVector expected_energy = expected_pressure / expected_density;
  CHECK_ITERABLE_APPROX(get(get<SpecificInternalEnergy>(vars)),
                        expected_energy);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.AnalyticData.BrioWu",
                  "[Unit][PointwiseFunctions]") {
  test_states();

  const auto created =
      TestHelpers::test_factory_creation<evolution::initial_data::InitialData,
                                         BrioWu>(
          "BrioWu:\n"
          "  AdiabaticIndex: 2.0\n"
          "  LeftDensity: 1.0\n"
          "  LeftPressure: 1.0\n"
          "  LeftTransverseMagneticField: 1.0\n"
          "  RightDensity: 0.125\n"
          "  RightPressure: 0.1\n"
          "  RightTransverseMagneticField: -1.0\n"
          "  ParallelMagneticField: 0.75\n");
  CHECK(dynamic_cast<const BrioWu&>(*created) == make_data());
  test_serialization(make_data());
}
