// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/FixConservatives.hpp"
#include "Evolution/Systems/NewtonianMhd/PrimitiveFromConservative.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Gsl.hpp"

namespace {
constexpr size_t Dim = 3;

// The specific internal energy that `PrimitiveFromConservative` will recover.
double recovered_specific_internal_energy(
    const double density, const std::array<double, Dim>& momentum_density,
    const double energy, const std::array<double, Dim>& magnetic_field) {
  double momentum_squared = 0.0;
  double magnetic_squared = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    momentum_squared += square(gsl::at(momentum_density, i));
    magnetic_squared += square(gsl::at(magnetic_field, i));
  }
  return (energy - 0.5 * magnetic_squared) / density -
         0.5 * momentum_squared / square(density);
}

void test_fixing(const double density_in, const double energy_in,
                 const std::array<double, Dim>& momentum_density_in,
                 const std::array<double, Dim>& magnetic_field_in,
                 const bool expect_fixing) {
  CAPTURE(density_in);
  CAPTURE(energy_in);
  const NewtonianMhd::FixConservatives<Dim> fixer{1.0e-12, 1.0e-12, 1.0e-12,
                                                  1.0e-12, true};

  Scalar<DataVector> density{DataVector{1, density_in}};
  Scalar<DataVector> energy{DataVector{1, energy_in}};
  tnsr::I<DataVector, Dim, Frame::Inertial> momentum_density{1_st};
  tnsr::I<DataVector, Dim, Frame::Inertial> magnetic_field{1_st};
  for (size_t i = 0; i < Dim; ++i) {
    momentum_density.get(i) = DataVector{1, gsl::at(momentum_density_in, i)};
    magnetic_field.get(i) = DataVector{1, gsl::at(magnetic_field_in, i)};
  }

  const bool fixed =
      fixer(make_not_null(&density), make_not_null(&momentum_density),
            make_not_null(&energy), make_not_null(&magnetic_field));
  CHECK(fixed == expect_fixing);

  // Whatever the input, the state the fixer leaves behind must yield a
  // non-negative internal energy, which is the whole point of the class.
  std::array<double, Dim> fixed_momentum{};
  std::array<double, Dim> fixed_field{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(fixed_momentum, i) = momentum_density.get(i)[0];
    gsl::at(fixed_field, i) = magnetic_field.get(i)[0];
  }
  CHECK(recovered_specific_internal_energy(get(density)[0], fixed_momentum,
                                           get(energy)[0], fixed_field) >= 0.0);

  // The energy density is never touched, and the fixes only ever shrink the
  // momentum density and magnetic field.
  CHECK(get(energy)[0] == energy_in);
  double momentum_squared = 0.0;
  double momentum_squared_in = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    momentum_squared += square(gsl::at(fixed_momentum, i));
    momentum_squared_in += square(gsl::at(momentum_density_in, i));
  }
  CHECK(momentum_squared <= momentum_squared_in * (1.0 + 1.0e-12));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianMhd.FixConservatives",
                  "[Unit][Evolution]") {
  // A physical state is left alone.
  test_fixing(1.2, 5.0, {{0.1, -0.2, 0.3}}, {{0.4, 0.1, -0.2}}, false);

  // Magnetic energy exceeding the total energy is rescaled away.
  test_fixing(1.2, 0.5, {{0.0, 0.0, 0.0}}, {{3.0, 0.0, 0.0}}, true);

  // Kinetic energy exceeding what is left after the magnetic part.
  test_fixing(1.2, 0.5, {{5.0, 0.0, 0.0}}, {{0.1, 0.0, 0.0}}, true);

  // Both at once, plus a density below the cutoff.
  test_fixing(1.0e-20, 0.5, {{5.0, 1.0, -2.0}}, {{3.0, -1.0, 2.0}}, true);

  // Disabled, so nothing happens even to an unphysical state.
  {
    const NewtonianMhd::FixConservatives<Dim> disabled{1.0e-12, 1.0e-12,
                                                       1.0e-12, 1.0e-12, false};
    Scalar<DataVector> density{DataVector{1, 1.0}};
    Scalar<DataVector> energy{DataVector{1, 0.1}};
    tnsr::I<DataVector, Dim, Frame::Inertial> momentum_density{1_st, 10.0};
    tnsr::I<DataVector, Dim, Frame::Inertial> magnetic_field{1_st, 10.0};
    CHECK_FALSE(
        disabled(make_not_null(&density), make_not_null(&momentum_density),
                 make_not_null(&energy), make_not_null(&magnetic_field)));
    CHECK(get(density)[0] == 1.0);
    CHECK(get<0>(momentum_density)[0] == 10.0);
  }

  const auto fixer =
      TestHelpers::test_creation<NewtonianMhd::FixConservatives<Dim>>(
          "MinimumValueOfDensity: 1.0e-12\n"
          "CutoffDensity: 1.0e-11\n"
          "SafetyFactorForB: 1.0e-12\n"
          "SafetyFactorForS: 1.0e-12\n"
          "Enable: true\n");
  test_serialization(fixer);
  CHECK(fixer == fixer);
  CHECK_FALSE(fixer != fixer);
}
