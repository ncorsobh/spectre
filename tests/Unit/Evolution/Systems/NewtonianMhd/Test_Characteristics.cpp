// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/Characteristics.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {

// With no magnetic field at all the fast magnetosonic speed degenerates to the
// sound speed and the outer speeds reduce to the Euler ones.
void test_hydrodynamic_limit() {
  const DataVector one{4, 1.0};
  const double divergence_cleaning_speed = 1.5;
  const Scalar<DataVector> mass_density{2.0 * one};
  const Scalar<DataVector> sound_speed_squared{0.36 * one};
  const tnsr::I<DataVector, 3> zero_field{DataVector(one.size(), 0.0)};

  tnsr::I<DataVector, 3> velocity{one.size()};
  get<0>(velocity) = 0.2 * one;
  get<1>(velocity) = 0.0 * one;
  get<2>(velocity) = 0.0 * one;
  tnsr::i<DataVector, 3> normal{one.size()};
  get<0>(normal) = 1.0 * one;
  get<1>(normal) = 0.0 * one;
  get<2>(normal) = 0.0 * one;

  Scalar<DataVector> fast_speed(one.size());
  NewtonianMhd::fast_magnetosonic_speed<true>(make_not_null(&fast_speed),
                                              mass_density, sound_speed_squared,
                                              zero_field, zero_field);
  CHECK_ITERABLE_APPROX(get(fast_speed), 0.6 * one);

  const auto speeds = NewtonianMhd::characteristic_speeds<true>(
      mass_density, velocity, sound_speed_squared, zero_field, normal,
      divergence_cleaning_speed, zero_field);
  CHECK_ITERABLE_APPROX(speeds[0],
                        DataVector(-divergence_cleaning_speed * one));
  CHECK_ITERABLE_APPROX(speeds[8], DataVector(divergence_cleaning_speed * one));
  CHECK_ITERABLE_APPROX(speeds[1], DataVector(-0.4 * one));
  CHECK_ITERABLE_APPROX(speeds[7], DataVector(0.8 * one));
  // Alfven and slow speeds vanish, leaving the entropy speed v_n.
  for (size_t i = 2; i < 7; ++i) {
    CHECK_ITERABLE_APPROX(gsl::at(speeds, i), DataVector(0.2 * one));
  }
}

// Field along the propagation direction: the normal Alfven speed is
// |B_x| / sqrt(rho) and the fast speed is sqrt(c_s^2 + B_x^2 / rho).
void test_field_aligned_with_normal() {
  const DataVector one{4, 1.0};
  const double divergence_cleaning_speed = 2.0;
  const Scalar<DataVector> mass_density{4.0 * one};
  const Scalar<DataVector> sound_speed_squared{0.25 * one};
  const tnsr::I<DataVector, 3> velocity{DataVector(one.size(), 0.0)};

  tnsr::I<DataVector, 3> magnetic_field{one.size()};
  get<0>(magnetic_field) = 2.0 * one;
  get<1>(magnetic_field) = 0.0 * one;
  get<2>(magnetic_field) = 0.0 * one;
  tnsr::I<DataVector, 3> background_magnetic_field{one.size()};
  get<0>(background_magnetic_field) = 4.0 * one;
  get<1>(background_magnetic_field) = 0.0 * one;
  get<2>(background_magnetic_field) = 0.0 * one;
  tnsr::i<DataVector, 3> normal{one.size()};
  get<0>(normal) = 1.0 * one;
  get<1>(normal) = 0.0 * one;
  get<2>(normal) = 0.0 * one;

  // B_total = 6, rho = 4, so c_An = 3 and c_f = sqrt(0.25 + 9).
  const double expected_alfven_speed = 3.0;
  const double expected_fast_speed = sqrt(9.25);

  Scalar<DataVector> fast_speed(one.size());
  NewtonianMhd::fast_magnetosonic_speed<true>(
      make_not_null(&fast_speed), mass_density, sound_speed_squared,
      magnetic_field, background_magnetic_field);
  CHECK_ITERABLE_APPROX(get(fast_speed), DataVector(expected_fast_speed * one));

  const auto speeds = NewtonianMhd::characteristic_speeds<true>(
      mass_density, velocity, sound_speed_squared, magnetic_field, normal,
      divergence_cleaning_speed, background_magnetic_field);
  CHECK_ITERABLE_APPROX(speeds[1], DataVector(-expected_fast_speed * one));
  CHECK_ITERABLE_APPROX(speeds[7], DataVector(expected_fast_speed * one));
  CHECK_ITERABLE_APPROX(speeds[2], DataVector(-expected_alfven_speed * one));
  CHECK_ITERABLE_APPROX(speeds[6], DataVector(expected_alfven_speed * one));
  // Slow speed is min(c_s, c_An) = 0.5.
  CHECK_ITERABLE_APPROX(speeds[3], DataVector(-0.5 * one));
  CHECK_ITERABLE_APPROX(speeds[5], DataVector(0.5 * one));
  CHECK_ITERABLE_APPROX(speeds[4], DataVector(0.0 * one));
}

// The speeds must always come out ordered, whatever the state.
void test_ordering(const gsl::not_null<std::mt19937*> generator) {
  const size_t num_points = 5;
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  std::uniform_real_distribution<> positive_distribution(0.5, 2.0);
  const double divergence_cleaning_speed = 10.0;

  const auto mass_density = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&positive_distribution), DataVector(num_points));
  const auto sound_speed_squared = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&positive_distribution), DataVector(num_points));
  const auto velocity = make_with_random_values<tnsr::I<DataVector, 3>>(
      generator, make_not_null(&distribution), DataVector(num_points));
  const auto magnetic_field = make_with_random_values<tnsr::I<DataVector, 3>>(
      generator, make_not_null(&distribution), DataVector(num_points));
  const auto background_magnetic_field =
      make_with_random_values<tnsr::I<DataVector, 3>>(
          generator, make_not_null(&distribution), DataVector(num_points));
  auto normal = make_with_random_values<tnsr::i<DataVector, 3>>(
      generator, make_not_null(&distribution), DataVector(num_points));
  DataVector normal_magnitude(num_points, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    normal_magnitude += square(normal.get(i));
  }
  normal_magnitude = sqrt(normal_magnitude);
  for (size_t i = 0; i < 3; ++i) {
    normal.get(i) /= normal_magnitude;
  }

  const auto speeds = NewtonianMhd::characteristic_speeds<true>(
      mass_density, velocity, sound_speed_squared, magnetic_field, normal,
      divergence_cleaning_speed, background_magnetic_field);
  for (size_t i = 0; i + 1 < speeds.size(); ++i) {
    for (size_t point = 0; point < num_points; ++point) {
      CHECK(gsl::at(speeds, i)[point] <= approx(gsl::at(speeds, i + 1)[point]));
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.Characteristics", "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test_hydrodynamic_limit();
  test_field_aligned_with_normal();
  test_ordering(make_not_null(&generator));
}
