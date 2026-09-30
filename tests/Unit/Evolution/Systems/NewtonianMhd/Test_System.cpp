// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.NewtonianMhd.System.Name", "[Unit][Evolution]") {
  CHECK((NewtonianMhd::System<false>::name()) == "NewtonianMhd");
  CHECK((NewtonianMhd::System<true>::name()) == "NewtonianMhd");

  // With the splitting disabled B0 is absent from the time derivative rather
  // than present and zero.
  static_assert(
      not tmpl::list_contains_v<
          typename NewtonianMhd::System<
              false>::compute_volume_time_derivative_terms::temporary_tags,
          NewtonianMhd::Tags::BackgroundMagneticField<>>);
  static_assert(tmpl::list_contains_v<
                typename NewtonianMhd::System<
                    true>::compute_volume_time_derivative_terms::temporary_tags,
                NewtonianMhd::Tags::BackgroundMagneticField<>>);
}
