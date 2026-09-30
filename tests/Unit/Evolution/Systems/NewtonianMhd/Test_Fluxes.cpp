// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace {

struct Fluxes3D {
  tnsr::I<DataVector, 3> mass_density;
  tnsr::IJ<DataVector, 3> momentum_density;
  tnsr::I<DataVector, 3> energy_density;
  tnsr::IJ<DataVector, 3> magnetic_field;
  tnsr::I<DataVector, 3> divergence_cleaning_field;
};

Fluxes3D compute_fluxes(const tnsr::I<DataVector, 3>& momentum_density,
                        const Scalar<DataVector>& energy_density,
                        const tnsr::I<DataVector, 3>& magnetic_field,
                        const Scalar<DataVector>& divergence_cleaning_field,
                        const tnsr::I<DataVector, 3>& velocity,
                        const Scalar<DataVector>& pressure,
                        const tnsr::I<DataVector, 3>& background_magnetic_field,
                        const double divergence_cleaning_speed) {
  const size_t num_points = get(energy_density).size();
  Fluxes3D result{
      .mass_density = tnsr::I<DataVector, 3>(num_points),
      .momentum_density = tnsr::IJ<DataVector, 3>(num_points),
      .energy_density = tnsr::I<DataVector, 3>(num_points),
      .magnetic_field = tnsr::IJ<DataVector, 3>(num_points),
      .divergence_cleaning_field = tnsr::I<DataVector, 3>(num_points)};
  NewtonianMhd::ComputeFluxes<true>::apply(
      make_not_null(&result.mass_density),
      make_not_null(&result.momentum_density),
      make_not_null(&result.energy_density),
      make_not_null(&result.magnetic_field),
      make_not_null(&result.divergence_cleaning_field), momentum_density,
      energy_density, magnetic_field, divergence_cleaning_field, velocity,
      pressure, divergence_cleaning_speed, background_magnetic_field);
  return result;
}

// Uniform state with v = (0.3, 0, 0), B1 = (0, 0.5, 0), B0 = 0, psi = 0, so
// that every flux component can be written down by hand.
void test_known_state() {
  const DataVector one{3, 1.0};
  const double divergence_cleaning_speed = 1.5;

  tnsr::I<DataVector, 3> velocity{one.size()};
  get<0>(velocity) = 0.3 * one;
  get<1>(velocity) = 0.0 * one;
  get<2>(velocity) = 0.0 * one;
  tnsr::I<DataVector, 3> momentum_density{one.size()};
  get<0>(momentum_density) = 0.6 * one;
  get<1>(momentum_density) = 0.0 * one;
  get<2>(momentum_density) = 0.0 * one;
  tnsr::I<DataVector, 3> magnetic_field{one.size()};
  get<0>(magnetic_field) = 0.0 * one;
  get<1>(magnetic_field) = 0.5 * one;
  get<2>(magnetic_field) = 0.0 * one;
  const tnsr::I<DataVector, 3> background_magnetic_field{
      DataVector(one.size(), 0.0)};
  const Scalar<DataVector> energy_density{5.0 * one};
  const Scalar<DataVector> pressure{1.0 * one};
  const Scalar<DataVector> divergence_cleaning_field{0.0 * one};

  const auto fluxes =
      compute_fluxes(momentum_density, energy_density, magnetic_field,
                     divergence_cleaning_field, velocity, pressure,
                     background_magnetic_field, divergence_cleaning_speed);

  // p_tot = P + |B1|^2 / 2 = 1 + 0.125 = 1.125
  CHECK_ITERABLE_APPROX(get<0>(fluxes.mass_density), 0.6 * one);
  CHECK_ITERABLE_APPROX(get<1>(fluxes.mass_density), 0.0 * one);

  CHECK_ITERABLE_APPROX((get<0, 0>(fluxes.momentum_density)), 1.305 * one);
  CHECK_ITERABLE_APPROX((get<1, 1>(fluxes.momentum_density)), 0.875 * one);
  CHECK_ITERABLE_APPROX((get<2, 2>(fluxes.momentum_density)), 1.125 * one);
  CHECK_ITERABLE_APPROX((get<0, 1>(fluxes.momentum_density)), 0.0 * one);

  // v . B1 = 0, so the magnetic term drops out of the energy flux.
  CHECK_ITERABLE_APPROX(get<0>(fluxes.energy_density), 1.8375 * one);
  CHECK_ITERABLE_APPROX(get<1>(fluxes.energy_density), 0.0 * one);

  // flux[j][i] = F^j(B^i); F^x(B^y) = v^x B^y = 0.15.
  CHECK_ITERABLE_APPROX((get<0, 1>(fluxes.magnetic_field)), 0.15 * one);
  CHECK_ITERABLE_APPROX((get<1, 0>(fluxes.magnetic_field)), -0.15 * one);
  CHECK_ITERABLE_APPROX((get<0, 0>(fluxes.magnetic_field)), 0.0 * one);

  CHECK_ITERABLE_APPROX(get<1>(fluxes.divergence_cleaning_field), 1.125 * one);
  CHECK_ITERABLE_APPROX(get<0>(fluxes.divergence_cleaning_field), 0.0 * one);
}

// The momentum flux must stay symmetric in its two indices even with a
// background field, and the cleaning field must enter only the diagonal of the
// induction flux.
void test_random_state_identities(
    const gsl::not_null<std::mt19937*> generator) {
  const size_t num_points = 5;
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  std::uniform_real_distribution<> positive_distribution(1.0, 2.0);
  const auto random_vector = [&generator, &distribution]() {
    return make_with_random_values<tnsr::I<DataVector, 3>>(
        generator, make_not_null(&distribution), DataVector(num_points));
  };
  const auto random_scalar = [&generator, &positive_distribution]() {
    return make_with_random_values<Scalar<DataVector>>(
        generator, make_not_null(&positive_distribution),
        DataVector(num_points));
  };

  const auto magnetic_field = random_vector();
  const auto background_magnetic_field = random_vector();
  const auto velocity = random_vector();
  const auto mass_density = random_scalar();
  auto momentum_density = velocity;
  for (size_t i = 0; i < 3; ++i) {
    momentum_density.get(i) *= get(mass_density);
  }
  const auto energy_density = random_scalar();
  const auto pressure = random_scalar();
  const auto divergence_cleaning_field = random_scalar();
  const double divergence_cleaning_speed = 1.5;

  const auto fluxes =
      compute_fluxes(momentum_density, energy_density, magnetic_field,
                     divergence_cleaning_field, velocity, pressure,
                     background_magnetic_field, divergence_cleaning_speed);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < i; ++j) {
      CHECK_ITERABLE_APPROX(fluxes.momentum_density.get(i, j),
                            fluxes.momentum_density.get(j, i));
      // The off-diagonal induction flux is antisymmetric: psi only appears on
      // the diagonal.
      CHECK_ITERABLE_APPROX(fluxes.magnetic_field.get(i, j),
                            DataVector(-fluxes.magnetic_field.get(j, i)));
    }
    CHECK_ITERABLE_APPROX(fluxes.magnetic_field.get(i, i),
                          get(divergence_cleaning_field));
  }
}

// Setting B0 = 0 must reproduce the standard MHD momentum flux
// rho v^i v^j + (P + |B|^2/2) delta^{ij} - B^i B^j.
void test_zero_background_reduces_to_mhd(
    const gsl::not_null<std::mt19937*> generator) {
  const size_t num_points = 5;
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto momentum_density = make_with_random_values<tnsr::I<DataVector, 3>>(
      generator, make_not_null(&distribution), DataVector(num_points));
  const auto magnetic_field = make_with_random_values<tnsr::I<DataVector, 3>>(
      generator, make_not_null(&distribution), DataVector(num_points));
  const auto velocity = make_with_random_values<tnsr::I<DataVector, 3>>(
      generator, make_not_null(&distribution), DataVector(num_points));
  const auto energy_density = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), DataVector(num_points));
  const auto pressure = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), DataVector(num_points));
  const auto divergence_cleaning_field =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&distribution), DataVector(num_points));
  const tnsr::I<DataVector, 3> background_magnetic_field{
      DataVector(num_points, 0.0)};
  const double divergence_cleaning_speed = 1.5;

  const auto fluxes =
      compute_fluxes(momentum_density, energy_density, magnetic_field,
                     divergence_cleaning_field, velocity, pressure,
                     background_magnetic_field, divergence_cleaning_speed);

  DataVector magnetic_pressure(num_points, 0.0);
  DataVector velocity_dot_magnetic_field(num_points, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    magnetic_pressure += 0.5 * square(magnetic_field.get(i));
    velocity_dot_magnetic_field += velocity.get(i) * magnetic_field.get(i);
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      DataVector expected = momentum_density.get(i) * velocity.get(j) -
                            magnetic_field.get(i) * magnetic_field.get(j);
      if (i == j) {
        expected += get(pressure) + magnetic_pressure;
      }
      CHECK_ITERABLE_APPROX(fluxes.momentum_density.get(j, i), expected);

      // flux[j][i] = F^j(B^i)
      DataVector expected_induction = velocity.get(j) * magnetic_field.get(i) -
                                      magnetic_field.get(j) * velocity.get(i);
      if (i == j) {
        expected_induction += get(divergence_cleaning_field);
      }
      CHECK_ITERABLE_APPROX(fluxes.magnetic_field.get(j, i),
                            expected_induction);
    }
    const DataVector expected_energy =
        (get(energy_density) + get(pressure) + magnetic_pressure) *
            velocity.get(i) -
        magnetic_field.get(i) * velocity_dot_magnetic_field;
    CHECK_ITERABLE_APPROX(fluxes.energy_density.get(i), expected_energy);
    CHECK_ITERABLE_APPROX(
        fluxes.divergence_cleaning_field.get(i),
        DataVector(square(divergence_cleaning_speed) * magnetic_field.get(i)));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.Fluxes", "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test_known_state();
  test_random_state_identities(make_not_null(&generator));
  test_zero_background_reduces_to_mhd(make_not_null(&generator));
}
