// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/DampingZone.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/Source.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Gsl.hpp"

namespace {
constexpr size_t dim = 3;
constexpr double sponge_inner_radius = 4.0;
constexpr double sponge_outer_radius = 8.0;
constexpr double damping_timescale = 2.0;
constexpr double asymptotic_velocity = 0.25;
constexpr double background_density = 1.5;
constexpr double background_pressure = 0.7;
constexpr double adiabatic_index = 5.0 / 3.0;

struct SourceOutput {
  Scalar<DataVector> mass_density;
  tnsr::I<DataVector, dim> momentum_density;
  Scalar<DataVector> energy_density;
  tnsr::I<DataVector, dim> magnetic_field;
  Scalar<DataVector> divergence_cleaning_field;
};

// Evaluates the source at points placed at the given radii along the z axis,
// starting from zeroed source terms as `TimeDerivativeTerms` does.
SourceOutput evaluate(const NewtonianMhd::Sources::Source<>& source,
                      const DataVector& radii) {
  const size_t num_points = radii.size();
  const EquationsOfState::IdealFluid<false> equation_of_state{adiabatic_index};

  tnsr::I<DataVector, dim> coords{num_points};
  get<0>(coords) = 0.0;
  get<1>(coords) = 0.0;
  get<2>(coords) = radii;

  const Scalar<DataVector> mass_density_cons{DataVector{num_points, 3.0}};
  tnsr::I<DataVector, dim> momentum_density{num_points};
  get<0>(momentum_density) = 0.4;
  get<1>(momentum_density) = -0.2;
  get<2>(momentum_density) = 0.9;
  const Scalar<DataVector> energy_density{DataVector{num_points, 6.0}};
  tnsr::I<DataVector, dim> magnetic_field{num_points};
  get<0>(magnetic_field) = 0.3;
  get<1>(magnetic_field) = -0.1;
  get<2>(magnetic_field) = 0.2;
  const Scalar<DataVector> divergence_cleaning_field{
      DataVector{num_points, 0.5}};
  const tnsr::I<DataVector, dim> velocity{DataVector{num_points, 0.0}};
  const Scalar<DataVector> pressure{DataVector{num_points, 1.0}};
  SourceOutput result{
      .mass_density = Scalar<DataVector>{DataVector{num_points, 0.0}},
      .momentum_density = tnsr::I<DataVector, dim>{DataVector{num_points, 0.0}},
      .energy_density = Scalar<DataVector>{DataVector{num_points, 0.0}},
      .magnetic_field = tnsr::I<DataVector, dim>{DataVector{num_points, 0.0}},
      .divergence_cleaning_field =
          Scalar<DataVector>{DataVector{num_points, 0.0}}};
  source(make_not_null(&result.mass_density),
         make_not_null(&result.momentum_density),
         make_not_null(&result.energy_density),
         make_not_null(&result.magnetic_field),
         make_not_null(&result.divergence_cleaning_field), mass_density_cons,
         momentum_density, energy_density, magnetic_field,
         divergence_cleaning_field, velocity, pressure,
         NewtonianMhd::NoBackgroundMagneticField{}, equation_of_state, coords,
         0.0);
  return result;
}

NewtonianMhd::Sources::DampingZone<> make_damping_zone() {
  return NewtonianMhd::Sources::DampingZone<>{
      sponge_inner_radius, sponge_outer_radius, damping_timescale,
      asymptotic_velocity, background_density,  background_pressure};
}

void test_damping_zone() {
  const auto damping_zone = make_damping_zone();
  // Inside the sponge, at its inner edge, in its middle, at its outer edge, and
  // beyond it.
  const DataVector radii{1.0, sponge_inner_radius, 6.0, sponge_outer_radius,
                         12.0};
  const auto result = evaluate(damping_zone, radii);

  // No damping at or inside the inner edge.
  CHECK(get(result.mass_density)[0] == 0.0);
  CHECK(get(result.mass_density)[1] == 0.0);
  CHECK(get(result.divergence_cleaning_field)[0] == 0.0);

  // Full strength at and beyond the outer edge.
  const double full_rate = 1.0 / damping_timescale;
  const EquationsOfState::IdealFluid<false> equation_of_state{adiabatic_index};
  const double target_energy_density =
      background_density *
      (get(equation_of_state.specific_internal_energy_from_density_and_pressure(
           Scalar<double>{background_density},
           Scalar<double>{background_pressure})) +
       0.5 * square(asymptotic_velocity));
  for (const size_t point : {3_st, 4_st}) {
    CHECK(get(result.mass_density)[point] ==
          approx(-full_rate * (3.0 - background_density)));
    CHECK(get<0>(result.momentum_density)[point] == approx(-full_rate * 0.4));
    CHECK(get<2>(result.momentum_density)[point] ==
          approx(-full_rate *
                 (0.9 - (background_density * asymptotic_velocity))));
    CHECK(get(result.energy_density)[point] ==
          approx(-full_rate * (6.0 - target_energy_density)));
    CHECK(get<1>(result.magnetic_field)[point] == approx(full_rate * 0.1));
    CHECK(get(result.divergence_cleaning_field)[point] ==
          approx(-full_rate * 0.5));
  }

  // Partial, strictly increasing damping across the sponge.
  CHECK(get(result.divergence_cleaning_field)[2] < 0.0);
  CHECK(get(result.divergence_cleaning_field)[2] >
        get(result.divergence_cleaning_field)[3]);

  // Every variable is driven by the same rate, so the ratios of the sources to
  // their deviations from target agree.
  const double rate_from_psi = -get(result.divergence_cleaning_field)[2] / 0.5;
  CHECK(get(result.mass_density)[2] ==
        approx(-rate_from_psi * (3.0 - background_density)));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.Sources.DampingZone",
                  "[Unit][Evolution]") {
  test_damping_zone();

  const auto created =
      TestHelpers::test_factory_creation<NewtonianMhd::Sources::Source<>,
                                         NewtonianMhd::Sources::DampingZone<>>(
          "DampingZone:\n"
          "  SpongeInnerRadius: 4.0\n"
          "  SpongeOuterRadius: 8.0\n"
          "  DampingTimescale: 2.0\n"
          "  AsymptoticVelocity: 0.25\n"
          "  BackgroundDensity: 1.5\n"
          "  BackgroundPressure: 0.7\n");
  CHECK(dynamic_cast<const NewtonianMhd::Sources::DampingZone<>&>(*created) ==
        make_damping_zone());
  test_serialization(make_damping_zone());
}
