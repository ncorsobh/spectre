// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/PrimitiveFromConservative.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <size_t Dim>
struct PrimitiveFromConservativeProxyThermoDim1 {
  static void apply_helper(
      const gsl::not_null<Scalar<DataVector>*> mass_density,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      const gsl::not_null<Scalar<DataVector>*> pressure,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density) {
    NewtonianMhd::PrimitiveFromConservative<Dim, false>::apply(
        mass_density, velocity, specific_internal_energy, pressure,
        mass_density_cons, momentum_density, energy_density,
        EquationsOfState::PolytropicFluid<false>(1.4, 5.0 / 3.0));
  }
};

template <size_t Dim>
struct PrimitiveFromConservativeProxyThermoDim2 {
  static void apply_helper(
      const gsl::not_null<Scalar<DataVector>*> mass_density,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      const gsl::not_null<Scalar<DataVector>*> pressure,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density) {
    NewtonianMhd::PrimitiveFromConservative<Dim, false>::apply(
        mass_density, velocity, specific_internal_energy, pressure,
        mass_density_cons, momentum_density, energy_density,
        EquationsOfState::IdealFluid<false>(5.0 / 3.0));
  }
};

template <size_t Dim>
void test_primitive_from_conservative(const DataVector& used_for_size) {
  pypp::check_with_random_values<3>(
      &PrimitiveFromConservativeProxyThermoDim1<Dim>::apply_helper,
      "PrimitiveFromConservative",
      {"mass_density", "velocity", "specific_internal_energy", "pressure_1d"},
      {{{0.0, 1.0}, {-2.0, 2.0}, {0.0, 3.0}}}, used_for_size);

  pypp::check_with_random_values<3>(
      &PrimitiveFromConservativeProxyThermoDim2<Dim>::apply_helper,
      "PrimitiveFromConservative",
      {"mass_density", "velocity", "specific_internal_energy", "pressure_2d"},
      {{{0.0, 1.0}, {-2.0, 2.0}, {0.0, 3.0}}}, used_for_size);
}

template <bool ThermoDim1>
struct MhdPrimitiveFromConservativeProxy {
  static void apply_helper(
      const gsl::not_null<Scalar<DataVector>*> mass_density,
      const gsl::not_null<tnsr::I<DataVector, 3>*> velocity,
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      const gsl::not_null<Scalar<DataVector>*> pressure,
      const gsl::not_null<tnsr::I<DataVector, 3>*> magnetic_field,
      const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, 3>& magnetic_field_cons,
      const Scalar<DataVector>& divergence_cleaning_field_cons) {
    if constexpr (ThermoDim1) {
      NewtonianMhd::PrimitiveFromConservative<3, true>::apply(
          mass_density, velocity, specific_internal_energy, pressure,
          magnetic_field, divergence_cleaning_field, mass_density_cons,
          momentum_density, energy_density, magnetic_field_cons,
          divergence_cleaning_field_cons,
          EquationsOfState::PolytropicFluid<false>(1.4, 5.0 / 3.0));
    } else {
      NewtonianMhd::PrimitiveFromConservative<3, true>::apply(
          mass_density, velocity, specific_internal_energy, pressure,
          magnetic_field, divergence_cleaning_field, mass_density_cons,
          momentum_density, energy_density, magnetic_field_cons,
          divergence_cleaning_field_cons,
          EquationsOfState::IdealFluid<false>(5.0 / 3.0));
    }
  }
};

void test_primitive_from_conservative_mhd(const DataVector& used_for_size) {
  pypp::check_with_random_values<5>(
      &MhdPrimitiveFromConservativeProxy<true>::apply_helper,
      "PrimitiveFromConservative",
      {"mhd_mass_density", "mhd_velocity", "mhd_specific_internal_energy",
       "mhd_pressure_1d", "mhd_magnetic_field",
       "mhd_divergence_cleaning_field"},
      {{{0.0, 1.0}, {-2.0, 2.0}, {0.0, 3.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
      used_for_size);

  pypp::check_with_random_values<5>(
      &MhdPrimitiveFromConservativeProxy<false>::apply_helper,
      "PrimitiveFromConservative",
      {"mhd_mass_density", "mhd_velocity", "mhd_specific_internal_energy",
       "mhd_pressure_2d", "mhd_magnetic_field",
       "mhd_divergence_cleaning_field"},
      {{{0.0, 1.0}, {-2.0, 2.0}, {0.0, 3.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
      used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianMhd.PrimitiveFromConservative",
    "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianMhd"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_primitive_from_conservative, (1, 2, 3))
  test_primitive_from_conservative_mhd(dv);
}
