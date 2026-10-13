// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim>
void test_hydrodynamic_variables() {
  using system = NewtonianMhd::System<Dim>;
  static_assert(not system::evolve_magnetic_field);
  static_assert(system::volume_dim == Dim);

  static_assert(
      std::is_same_v<typename system::flux_variables,
                     tmpl::list<NewtonianMhd::Tags::MassDensityCons,
                                NewtonianMhd::Tags::MomentumDensity<Dim>,
                                NewtonianMhd::Tags::EnergyDensity>>,
      "Evolving without a magnetic field must leave the Euler variables "
      "untouched.");

  static_assert(
      std::is_same_v<typename system::primitive_variables_tag::tags_list,
                     tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                                hydro::Tags::SpatialVelocity<DataVector, Dim>,
                                hydro::Tags::SpecificInternalEnergy<DataVector>,
                                hydro::Tags::Pressure<DataVector>>>);
}

void test_magnetic_variables() {
  using system = NewtonianMhd::System<3, true>;
  static_assert(system::evolve_magnetic_field);
  static_assert(system::volume_dim == 3);

  static_assert(
      std::is_same_v<
          typename system::flux_variables,
          tmpl::list<NewtonianMhd::Tags::MassDensityCons,
                     NewtonianMhd::Tags::MomentumDensity<3>,
                     NewtonianMhd::Tags::EnergyDensity,
                     NewtonianMhd::Tags::MagneticFieldCons<>,
                     NewtonianMhd::Tags::DivergenceCleaningFieldCons>>,
      "The magnetic field and the divergence-cleaning field are appended to "
      "the Euler variables, which keep their order.");

  static_assert(std::is_same_v<
                typename system::primitive_variables_tag::tags_list,
                tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                           hydro::Tags::SpatialVelocity<DataVector, 3>,
                           hydro::Tags::SpecificInternalEnergy<DataVector>,
                           hydro::Tags::Pressure<DataVector>,
                           hydro::Tags::MagneticField<DataVector, 3>,
                           hydro::Tags::DivergenceCleaningField<DataVector>>>);

  // The evolved variables are exactly the flux variables in both cases.
  static_assert(std::is_same_v<typename system::variables_tag::tags_list,
                               typename system::flux_variables>);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.System", "[Unit][Evolution]") {
  CHECK(NewtonianMhd::System<1>::name() == "NewtonianEuler");
  CHECK(NewtonianMhd::System<2>::name() == "NewtonianEuler");
  CHECK(NewtonianMhd::System<3>::name() == "NewtonianEuler");
  CHECK(NewtonianMhd::System<3, true>::name() == "NewtonianMhd");

  test_hydrodynamic_variables<1>();
  test_hydrodynamic_variables<2>();
  test_hydrodynamic_variables<3>();
  test_magnetic_variables();
}
