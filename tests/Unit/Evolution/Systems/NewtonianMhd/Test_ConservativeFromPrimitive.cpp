// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/PrimitiveFromConservative.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Gsl.hpp"

namespace {

void test_round_trip(const gsl::not_null<std::mt19937*> generator) {
  const size_t num_points = 5;
  const EquationsOfState::IdealFluid<false> equation_of_state{5.0 / 3.0};

  std::uniform_real_distribution<> positive_distribution(1.0, 2.0);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  const auto mass_density = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&positive_distribution), DataVector(num_points));
  const auto specific_internal_energy =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&positive_distribution),
          DataVector(num_points));
  const auto velocity = make_with_random_values<tnsr::I<DataVector, 3>>(
      generator, make_not_null(&distribution), DataVector(num_points));
  const auto magnetic_field = make_with_random_values<tnsr::I<DataVector, 3>>(
      generator, make_not_null(&distribution), DataVector(num_points));
  const auto divergence_cleaning_field =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&distribution), DataVector(num_points));

  Scalar<DataVector> mass_density_cons(num_points);
  tnsr::I<DataVector, 3> momentum_density(num_points);
  Scalar<DataVector> energy_density(num_points);
  tnsr::I<DataVector, 3> magnetic_field_cons(num_points);
  Scalar<DataVector> divergence_cleaning_field_cons(num_points);
  NewtonianMhd::ConservativeFromPrimitive::apply(
      make_not_null(&mass_density_cons), make_not_null(&momentum_density),
      make_not_null(&energy_density), make_not_null(&magnetic_field_cons),
      make_not_null(&divergence_cleaning_field_cons), mass_density, velocity,
      specific_internal_energy, magnetic_field, divergence_cleaning_field);

  // The magnetic field and the cleaning field are pure identity maps.
  CHECK_ITERABLE_APPROX(magnetic_field_cons, magnetic_field);
  CHECK_ITERABLE_APPROX(divergence_cleaning_field_cons,
                        divergence_cleaning_field);

  Scalar<DataVector> recovered_mass_density(num_points);
  tnsr::I<DataVector, 3> recovered_velocity(num_points);
  Scalar<DataVector> recovered_specific_internal_energy(num_points);
  Scalar<DataVector> recovered_pressure(num_points);
  tnsr::I<DataVector, 3> recovered_magnetic_field(num_points);
  Scalar<DataVector> recovered_divergence_cleaning_field(num_points);
  NewtonianMhd::PrimitiveFromConservative::apply(
      make_not_null(&recovered_mass_density),
      make_not_null(&recovered_velocity),
      make_not_null(&recovered_specific_internal_energy),
      make_not_null(&recovered_pressure),
      make_not_null(&recovered_magnetic_field),
      make_not_null(&recovered_divergence_cleaning_field), mass_density_cons,
      momentum_density, energy_density, magnetic_field_cons,
      divergence_cleaning_field_cons, equation_of_state);

  CHECK_ITERABLE_APPROX(recovered_mass_density, mass_density);
  CHECK_ITERABLE_APPROX(recovered_velocity, velocity);
  CHECK_ITERABLE_APPROX(recovered_specific_internal_energy,
                        specific_internal_energy);
  CHECK_ITERABLE_APPROX(recovered_magnetic_field, magnetic_field);
  CHECK_ITERABLE_APPROX(recovered_divergence_cleaning_field,
                        divergence_cleaning_field);
  CHECK_ITERABLE_APPROX(recovered_pressure,
                        equation_of_state.pressure_from_density_and_energy(
                            mass_density, specific_internal_energy));
}

void test_energy_density() {
  const DataVector one{1, 1.0};
  const Scalar<DataVector> mass_density{2.0 * one};
  const Scalar<DataVector> specific_internal_energy{1.5 * one};
  tnsr::I<DataVector, 3> velocity{one.size()};
  get<0>(velocity) = 3.0 * one;
  get<1>(velocity) = 0.0 * one;
  get<2>(velocity) = 4.0 * one;
  tnsr::I<DataVector, 3> magnetic_field{one.size()};
  get<0>(magnetic_field) = 0.0 * one;
  get<1>(magnetic_field) = 2.0 * one;
  get<2>(magnetic_field) = 0.0 * one;
  const Scalar<DataVector> divergence_cleaning_field{0.5 * one};

  Scalar<DataVector> mass_density_cons(one.size());
  tnsr::I<DataVector, 3> momentum_density(one.size());
  Scalar<DataVector> energy_density(one.size());
  tnsr::I<DataVector, 3> magnetic_field_cons(one.size());
  Scalar<DataVector> divergence_cleaning_field_cons(one.size());
  NewtonianMhd::ConservativeFromPrimitive::apply(
      make_not_null(&mass_density_cons), make_not_null(&momentum_density),
      make_not_null(&energy_density), make_not_null(&magnetic_field_cons),
      make_not_null(&divergence_cleaning_field_cons), mass_density, velocity,
      specific_internal_energy, magnetic_field, divergence_cleaning_field);

  // rho (v^2/2 + eps) + |B1|^2/2 = 2 (25/2 + 3/2) + 2 = 30
  CHECK_ITERABLE_APPROX(get(energy_density), 30.0 * one);
  CHECK_ITERABLE_APPROX(get<0>(momentum_density), 6.0 * one);
  CHECK_ITERABLE_APPROX(get<2>(momentum_density), 8.0 * one);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.ConservativeFromPrimitive",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test_round_trip(make_not_null(&generator));
  test_energy_density();
}
