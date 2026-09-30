// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>

#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/NewtonianMhd/FiniteDifference/PrimReconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"

namespace {
void test(const std::optional<double> alpha_7,
          const std::optional<double> alpha_9,
          const std::string& alpha_7_option,
          const std::string& alpha_9_option) {
  CAPTURE(3);
  CAPTURE(alpha_7);
  CAPTURE(alpha_9);
  namespace helpers = TestHelpers::NewtonianMhd::fd;
  const NewtonianMhd::fd::PositivityPreservingAdaptiveOrderPrim recons{
      4.0, alpha_7, alpha_9,
      ::fd::reconstruction::FallbackReconstructorType::MonotonisedCentral};
  helpers::test_prim_reconstructor<>(2 * recons.ghost_zone_size() + 1, recons);

  const auto from_options_base = TestHelpers::test_factory_creation<
      NewtonianMhd::fd::Reconstructor,
      NewtonianMhd::fd::OptionTags::Reconstructor>(
      "PositivityPreservingAdaptiveOrderPrim:\n"
      "  Alpha5: 4.0\n"
      "  Alpha7: " +
      alpha_7_option +
      "\n"
      "  Alpha9: " +
      alpha_9_option +
      "\n"
      "  LowOrderReconstructor: MonotonisedCentral\n");
  const auto* const from_options = dynamic_cast<
      const NewtonianMhd::fd::PositivityPreservingAdaptiveOrderPrim*>(
      from_options_base.get());
  REQUIRE(from_options != nullptr);
  CHECK(*from_options == recons);
  CHECK_FALSE(*from_options != recons);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianMhd.Fd.PositivityPreservingAdaptiveOrder",
    "[Unit][Evolution]") {
  test(std::nullopt, std::nullopt, "None", "None");
  test(4.0, std::nullopt, "4.0", "None");
  test(4.0, 4.0, "4.0", "4.0");
}
