// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianMhd/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Gsl.hpp"

namespace {

// With NoSource the only non-flux term is the GLM constraint damping, and the
// fluxes must agree with ComputeFluxes evaluated on the same state.
void test_glm_damping_and_fluxes() {
  constexpr size_t dim = 3;
  const DataVector one{4, 1.0};
  const size_t num_points = one.size();
  const double divergence_cleaning_speed = 1.5;
  const double constraint_damping_parameter = 0.1;

  const Scalar<DataVector> mass_density_cons{2.0 * one};
  tnsr::I<DataVector, dim> velocity{num_points};
  get<0>(velocity) = 0.3 * one;
  get<1>(velocity) = -0.1 * one;
  get<2>(velocity) = 0.2 * one;
  auto momentum_density = velocity;
  for (size_t i = 0; i < dim; ++i) {
    momentum_density.get(i) *= get(mass_density_cons);
  }
  tnsr::I<DataVector, dim> magnetic_field{num_points};
  get<0>(magnetic_field) = 0.1 * one;
  get<1>(magnetic_field) = 0.5 * one;
  get<2>(magnetic_field) = -0.2 * one;
  tnsr::I<DataVector, dim> background_magnetic_field_volume{num_points};
  get<0>(background_magnetic_field_volume) = 0.4 * one;
  get<1>(background_magnetic_field_volume) = 0.0 * one;
  get<2>(background_magnetic_field_volume) = 0.3 * one;
  const Scalar<DataVector> energy_density{5.0 * one};
  const Scalar<DataVector> pressure{1.0 * one};
  const Scalar<DataVector> divergence_cleaning_field{0.25 * one};
  const tnsr::I<DataVector, dim> coords{DataVector(num_points, 0.0)};
  const EquationsOfState::IdealFluid<false> equation_of_state{5.0 / 3.0};
  const NewtonianMhd::Sources::NoSource<true> source{};

  Scalar<DataVector> dt_mass_density(num_points);
  tnsr::I<DataVector, dim> dt_momentum_density(num_points);
  Scalar<DataVector> dt_energy_density(num_points);
  tnsr::I<DataVector, dim> dt_magnetic_field(num_points);
  Scalar<DataVector> dt_divergence_cleaning_field(num_points);
  tnsr::I<DataVector, dim> mass_density_flux(num_points);
  tnsr::IJ<DataVector, dim> momentum_density_flux(num_points);
  tnsr::I<DataVector, dim> energy_density_flux(num_points);
  tnsr::IJ<DataVector, dim> magnetic_field_flux(num_points);
  tnsr::I<DataVector, dim> divergence_cleaning_field_flux(num_points);
  Scalar<DataVector> magnetic_pressure(num_points);
  tnsr::I<DataVector, dim> background_magnetic_field(num_points);

  NewtonianMhd::TimeDerivativeTerms<true>::apply(
      make_not_null(&dt_mass_density), make_not_null(&dt_momentum_density),
      make_not_null(&dt_energy_density), make_not_null(&dt_magnetic_field),
      make_not_null(&dt_divergence_cleaning_field),
      make_not_null(&mass_density_flux), make_not_null(&momentum_density_flux),
      make_not_null(&energy_density_flux), make_not_null(&magnetic_field_flux),
      make_not_null(&divergence_cleaning_field_flux),
      make_not_null(&magnetic_pressure),
      make_not_null(&background_magnetic_field), mass_density_cons,
      momentum_density, energy_density, magnetic_field,
      divergence_cleaning_field, velocity, pressure, divergence_cleaning_speed,
      constraint_damping_parameter, equation_of_state, coords, 0.0, source,
      background_magnetic_field_volume);

  CHECK_ITERABLE_APPROX(get(dt_mass_density), 0.0 * one);
  CHECK_ITERABLE_APPROX(get(dt_energy_density), 0.0 * one);
  for (size_t i = 0; i < dim; ++i) {
    CHECK_ITERABLE_APPROX(dt_momentum_density.get(i), 0.0 * one);
    CHECK_ITERABLE_APPROX(dt_magnetic_field.get(i), 0.0 * one);
  }
  CHECK_ITERABLE_APPROX(get(dt_divergence_cleaning_field),
                        DataVector(-constraint_damping_parameter *
                                   divergence_cleaning_speed * 0.25 * one));

  tnsr::I<DataVector, dim> expected_mass_density_flux(num_points);
  tnsr::IJ<DataVector, dim> expected_momentum_density_flux(num_points);
  tnsr::I<DataVector, dim> expected_energy_density_flux(num_points);
  tnsr::IJ<DataVector, dim> expected_magnetic_field_flux(num_points);
  tnsr::I<DataVector, dim> expected_divergence_cleaning_field_flux(num_points);
  NewtonianMhd::ComputeFluxes<true>::apply(
      make_not_null(&expected_mass_density_flux),
      make_not_null(&expected_momentum_density_flux),
      make_not_null(&expected_energy_density_flux),
      make_not_null(&expected_magnetic_field_flux),
      make_not_null(&expected_divergence_cleaning_field_flux), momentum_density,
      energy_density, magnetic_field, divergence_cleaning_field, velocity,
      pressure, divergence_cleaning_speed, background_magnetic_field_volume);

  CHECK_ITERABLE_APPROX(mass_density_flux, expected_mass_density_flux);
  CHECK_ITERABLE_APPROX(momentum_density_flux, expected_momentum_density_flux);
  CHECK_ITERABLE_APPROX(energy_density_flux, expected_energy_density_flux);
  CHECK_ITERABLE_APPROX(magnetic_field_flux, expected_magnetic_field_flux);
  CHECK_ITERABLE_APPROX(divergence_cleaning_field_flux,
                        expected_divergence_cleaning_field_flux);
  CHECK_ITERABLE_APPROX(background_magnetic_field,
                        background_magnetic_field_volume);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.TimeDerivativeTerms",
                  "[Unit][Evolution]") {
  test_glm_damping_and_fluxes();
}
