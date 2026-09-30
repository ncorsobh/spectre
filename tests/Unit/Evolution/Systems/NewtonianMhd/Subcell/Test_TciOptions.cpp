// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/NewtonianMhd/Subcell/TciOptions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Serialization/Serialize.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianMhd.Subcell.TciOptions",
                  "[Unit][Evolution]") {
  const auto tci_options_from_opts = TestHelpers::test_option_tag<
      NewtonianMhd::subcell::OptionTags::TciOptions>(
      "MinimumValueOfDensity: 1.0e-18\n"
      "MinimumValueOfPressure: 1.0e-16\n"
      "SafetyFactorForB: 1.0e-12\n"
      "MagneticFieldCutoff: 0.01\n");
  const auto tci_options = serialize_and_deserialize(tci_options_from_opts);
  CHECK(tci_options.minimum_density == 1.0e-18);
  CHECK(tci_options.minimum_pressure == 1.0e-16);
  CHECK(tci_options.safety_factor_for_magnetic_field == 1.0e-12);
  CHECK(tci_options.magnetic_field_cutoff.value() == 0.01);

  const auto no_b_check =
      serialize_and_deserialize(TestHelpers::test_option_tag<
                                NewtonianMhd::subcell::OptionTags::TciOptions>(
          "MinimumValueOfDensity: 1.0e-18\n"
          "MinimumValueOfPressure: 1.0e-16\n"
          "SafetyFactorForB: 1.0e-12\n"
          "MagneticFieldCutoff: DoNotCheckMagneticField\n"));
  CHECK_FALSE(no_b_check.magnetic_field_cutoff.has_value());
}
