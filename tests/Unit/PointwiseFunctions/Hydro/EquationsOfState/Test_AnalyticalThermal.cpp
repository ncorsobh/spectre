// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "Framework/TestingFramework.hpp"

#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
namespace {
// void check_random_polytrope() {
//   register_derived_classes_with_charm<
//       EquationsOfState::EquationOfState<true, 3>>();
//   const double d_for_size = std::numeric_limits<double>::signaling_NaN();
//   const DataVector dv_for_size(5);
//   TestHelpers::EquationsOfState::check(
//       EquationsOfState::HybridEos<
//           EquationsOfState::PolytropicFluid<IsRelativistic>>{
//           EquationsOfState::PolytropicFluid<IsRelativistic>{100.0, 4.0
//           / 3.0}, 5.0 / 3.0},
//       "HybridEos", "hybrid_polytrope", d_for_size, 100.0, 4.0 / 3.0, 5.0
//       / 3.0);
//   TestHelpers::EquationsOfState::check(
//       EquationsOfState::HybridEos<
//           EquationsOfState::PolytropicFluid<IsRelativistic>>{
//           EquationsOfState::PolytropicFluid<IsRelativistic>{100.0, 4.0
//           / 3.0}, 5.0 / 3.0},
//       "HybridEos", "hybrid_polytrope", dv_for_size, 100.0, 4.0 / 3.0,
//       5.0 / 3.0);
// }
// Matches implementation
double baryon_mass_in_mev_ = 939.57;
double saturation_density_ = 4.34e-4;
double saturation_number_density_in_fm_3 = .16;
namespace EoS = EquationsOfState;
// Comparing to results in the paper is only possible to
// the precision of constants like nuclear saturation density
// (n_sat ~.16 fm^-3)
Approx paper_approx = Approx::custom().epsilon(1e-3);
void check_creation() {
  // Check that the EoS can be created and copied correctly, and that the
  // EoS has sane behavior out of equilibrium.
  EquationsOfState::PolytropicFluid<true> cold_eos{100.0, 2.0};
  EquationsOfState::AnalyticalThermal<EquationsOfState::PolytropicFluid<true>>
      eos{{100.0, 2.0}, 1.5, .1, .1, 1.0, .85};
  TestHelpers::EquationsOfState::test_get_clone(eos);

  EquationsOfState::AnalyticalThermal<EquationsOfState::PolytropicFluid<true>>
      other_eos{{100.0, 3.0}, 1.5, .1, .1, 1.0, .89};
  const auto other_type_eos =
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0};
  CHECK(eos == eos);
  CHECK(eos != other_eos);
  CHECK(eos != other_type_eos);
  // Sanity checks
  const Scalar<double> rho{1.0e-3};
  const Scalar<double> zero_temp{0.0};
  const Scalar<double> temp{0.01};
  const Scalar<double> elec_frac{.1};
  // Out of equilibrium energy must be greater or equal to beta-equalibrium
  // energy
  CHECK(get(cold_eos.specific_internal_energy_from_density(rho)) <=
        get(eos.specific_internal_energy_from_density_and_temperature(
            rho, zero_temp, elec_frac)));
  // energy at nonzero temperature must be greater or equal to energy at zero
  // temp
  CHECK(get(eos.specific_internal_energy_from_density_and_temperature(
            rho, zero_temp, elec_frac)) <=
        get(eos.specific_internal_energy_from_density_and_temperature(
            rho, temp, elec_frac)));
}

void check_compare_to_paper_results() {
  double cold_eos_polytropic_constant = 100.0;
  double cold_eos_polytropic_exponent = 2.0;
  double S_0 = 31.57 / baryon_mass_in_mev_;
  double L = 47.10 / baryon_mass_in_mev_;
  double gamma = .41;
  double n_0 = saturation_density_ * .08 / .16;
  double alpha = .6;
  EquationsOfState::PolytropicFluid<true> cold_eos{
      cold_eos_polytropic_constant, cold_eos_polytropic_exponent};
  // Particular representation of the sfho eos
  EoS::AnalyticalThermal<EoS::PolytropicFluid<true>> eos_sfho_paper{
      cold_eos, S_0, L, gamma, n_0, alpha};

  // Particular values provided by Carolyn Raithel to test against
  const Scalar<double> rho{saturation_density_};
  const Scalar<double> temp{10.0 / baryon_mass_in_mev_};
  const Scalar<double> elec_frac{.25};
  const Scalar<double> small_rho{saturation_density_};

  const double thermal_specific_internal_energy_provided =
      3.593102 / baryon_mass_in_mev_;
  const double out_of_equalibrium_specific_internal_energy_provided =
      1.69207e1 / baryon_mass_in_mev_;
  const double thermal_pressure_provided = 4.587350e-01 / baryon_mass_in_mev_ *
                                           saturation_density_ /
                                           saturation_number_density_in_fm_3;
  const double out_of_equalibrium_pressure_provided =
      4.161009e-1 / baryon_mass_in_mev_ * saturation_density_ /
      saturation_number_density_in_fm_3;

  const auto p = eos_sfho_paper.pressure_from_density_and_temperature(
      small_rho, temp, elec_frac);
  const auto pc = cold_eos.pressure_from_density(small_rho);

  CHECK(get(p) - get(pc) == paper_approx(thermal_pressure_provided +
                                         out_of_equalibrium_pressure_provided));

  const auto eps =
      eos_sfho_paper.specific_internal_energy_from_density_and_temperature(
          small_rho, temp, elec_frac);
  const auto epsc = cold_eos.specific_internal_energy_from_density(small_rho);
  CHECK(get(eps) - get(epsc) ==
        paper_approx(thermal_specific_internal_energy_provided +
                     out_of_equalibrium_specific_internal_energy_provided));
  // Check temperature from energy is computed correctly
  CHECK(get(eos_sfho_paper.temperature_from_density_and_energy(
            small_rho, eps, elec_frac)) == paper_approx(get(temp)));
}

void check_bounds() {
  const auto cold_eos = EquationsOfState::PolytropicFluid<true>{100.0, 1.5};
  const EquationsOfState::AnalyticalThermal<
      EquationsOfState::PolytropicFluid<true>>
      eos{cold_eos, 1.5, .1, .1, 1.0, .89};
  double electron_fraction = .05;
  double rest_mass_density = .005;
  CHECK(0.0 == eos.rest_mass_density_lower_bound());
  CHECK(0.0 == eos.temperature_lower_bound());

  CHECK(0.0 == eos.specific_internal_energy_lower_bound(rest_mass_density,
                                                        electron_fraction));
  CHECK(0.0 == eos.electron_fraction_lower_bound());
  CHECK(0.5 == eos.electron_fraction_upper_bound());

  const double max_double = std::numeric_limits<double>::max();
  CHECK(max_double == eos.rest_mass_density_upper_bound());
  CHECK(max_double == eos.specific_internal_energy_upper_bound(
                          rest_mass_density, electron_fraction));
  CHECK(max_double == eos.temperature_upper_bound());
}
void check_random_polytrope() {
  // Compare against python implementation
  // Relativistic checks
  double cold_eos_polytropic_constant = 100.0;
  double cold_eos_polytropic_exponent = 2.0;
  double S_0 = 31.57 / baryon_mass_in_mev_;
  double L = 47.10 / baryon_mass_in_mev_;
  double gamma = .41;
  double n_0 = saturation_density_ * .08 / .16;
  double alpha = .6;
  DataVector d_for_size{5};
  INFO("Testing get clone...");
  TestHelpers::EquationsOfState::test_get_clone(
      EoS::AnalyticalThermal<EoS::PolytropicFluid<true>>{
          EquationsOfState::PolytropicFluid<true>(cold_eos_polytropic_constant,
                                                  cold_eos_polytropic_exponent),
          S_0, L, gamma, n_0, alpha});
  INFO("Testing against python implementation..");
  TestHelpers::EquationsOfState::check(
      EoS::AnalyticalThermal<EoS::PolytropicFluid<true>>{
          EquationsOfState::PolytropicFluid<true>(cold_eos_polytropic_constant,
                                                  cold_eos_polytropic_exponent),
          S_0, L, gamma, n_0, alpha},
      "AnalyticalThermal", "analytical_thermal_polytrope", d_for_size,
      cold_eos_polytropic_constant, cold_eos_polytropic_exponent, S_0, L, gamma,
      n_0, alpha);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.AnalyticalThermal",
                  "[Unit][EquationsOfState]") {
  register_derived_classes_with_charm<EoS::EquationOfState<true, 3>>();
  register_derived_classes_with_charm<EoS::EquationOfState<true, 2>>();
  register_derived_classes_with_charm<EoS::EquationOfState<true, 1>>();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro/EquationsOfState/"};

  check_creation();
  check_bounds();
  check_compare_to_paper_results();
  check_random_polytrope();
}
