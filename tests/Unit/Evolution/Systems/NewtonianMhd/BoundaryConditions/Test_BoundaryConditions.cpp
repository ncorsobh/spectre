// Distributed under the MIT License
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/ConductorReflection.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/Reflection.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

struct ConvertIdeal {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::EquationOfState<false, 2>;
  using packed_type = bool;

  static unpacked_container unpack(const packed_container& /*packed*/,
                                   const size_t /*grid_point_index*/) {
    return false;
  }

  [[noreturn]] static void pack(
      const gsl::not_null<packed_container*> /*packed*/,
      const unpacked_container& /*unpacked*/,
      const size_t /*grid_point_index*/) {
    ERROR("Should not be converting an EOS from an unpacked to a packed type");
  }

  static size_t get_size(const packed_container& /*packed*/) { return 1; }
};

template <size_t Dim>
using ghost_function_names = tuples::TaggedTuple<
    helpers::Tags::PythonFunctionForErrorMessage<>,
    helpers::Tags::PythonFunctionName<NewtonianMhd::Tags::MassDensityCons>,
    helpers::Tags::PythonFunctionName<NewtonianMhd::Tags::MomentumDensity<Dim>>,
    helpers::Tags::PythonFunctionName<NewtonianMhd::Tags::EnergyDensity>,
    helpers::Tags::PythonFunctionName<
        NewtonianMhd::Tags::MagneticFieldCons<Dim>>,
    helpers::Tags::PythonFunctionName<
        NewtonianMhd::Tags::DivergenceCleaningFieldCons>,
    helpers::Tags::PythonFunctionName<
        ::Tags::Flux<NewtonianMhd::Tags::MassDensityCons, tmpl::size_t<Dim>,
                     Frame::Inertial>>,
    helpers::Tags::PythonFunctionName<
        ::Tags::Flux<NewtonianMhd::Tags::MomentumDensity<Dim>,
                     tmpl::size_t<Dim>, Frame::Inertial>>,
    helpers::Tags::PythonFunctionName<::Tags::Flux<
        NewtonianMhd::Tags::EnergyDensity, tmpl::size_t<Dim>, Frame::Inertial>>,
    helpers::Tags::PythonFunctionName<
        ::Tags::Flux<NewtonianMhd::Tags::MagneticFieldCons<Dim>,
                     tmpl::size_t<Dim>, Frame::Inertial>>,
    helpers::Tags::PythonFunctionName<
        ::Tags::Flux<NewtonianMhd::Tags::DivergenceCleaningFieldCons,
                     tmpl::size_t<Dim>, Frame::Inertial>>,
    helpers::Tags::PythonFunctionName<
        NewtonianMhd::Tags::BackgroundMagneticField<Dim>>,
    helpers::Tags::PythonFunctionName<
        hydro::Tags::SpatialVelocity<DataVector, Dim>>,
    helpers::Tags::PythonFunctionName<
        hydro::Tags::SpecificInternalEnergy<DataVector>>>;

auto positive_ranges() {
  return tuples::TaggedTuple<
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<hydro::Tags::SpecificInternalEnergy<DataVector>>>{
      std::array{1.0e-2, 1.0}, std::array{1.0e-2, 1.0}};
}

// Only the background-field build is exercised here: with the splitting
// disabled neither the ghost B0 nor the interior B0 exists, so there is no
// B0-related behaviour left to compare against.
template <size_t Dim, typename BoundaryConditionType>
void test_ghost_condition(const std::string& python_module,
                          const std::string& option_string) {
  MAKE_GENERATOR(gen);
  helpers::test_boundary_condition_with_python<
      BoundaryConditionType,
      NewtonianMhd::BoundaryConditions::BoundaryCondition<Dim>,
      NewtonianMhd::System<Dim, true>,
      tmpl::list<NewtonianMhd::BoundaryCorrections::Hll<Dim, true>>>(
      make_not_null(&gen), python_module,
      ghost_function_names<Dim>{
          "error", "mass_density_cons", "momentum_density", "energy_density",
          "magnetic_field_cons", "divergence_cleaning_field_cons",
          "flux_mass_density", "flux_momentum_density", "flux_energy_density",
          "flux_magnetic_field", "flux_divergence_cleaning_field",
          "background_magnetic_field", "velocity", "specific_internal_energy"},
      option_string, Index<Dim - 1>{Dim == 1 ? 1 : 5},
      db::create<
          db::AddSimpleTags<NewtonianMhd::Tags::DivergenceCleaningSpeed>>(1.5),
      positive_ranges());
}

template <size_t Dim>
void test_demand_outgoing_char_speeds() {
  MAKE_GENERATOR(gen);
  helpers::test_boundary_condition_with_python<
      NewtonianMhd::BoundaryConditions::DemandOutgoingCharSpeeds<Dim, true>,
      NewtonianMhd::BoundaryConditions::BoundaryCondition<Dim>,
      NewtonianMhd::System<Dim, true>,
      tmpl::list<NewtonianMhd::BoundaryCorrections::Hll<Dim, true>>,
      tmpl::list<ConvertIdeal>>(
      make_not_null(&gen),
      "Evolution.Systems.NewtonianMhd.BoundaryConditions."
      "DemandOutgoingCharSpeeds",
      tuples::TaggedTuple<helpers::Tags::PythonFunctionForErrorMessage<>>{
          "error"},
      "DemandOutgoingCharSpeeds:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5},
      db::create<db::AddSimpleTags<hydro::Tags::EquationOfState<false, 2>>>(
          EquationsOfState::IdealFluid<false>{1.3}.get_clone()),
      positive_ranges());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.BoundaryConditions", "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test_ghost_condition<3,
                       NewtonianMhd::BoundaryConditions::Reflection<3, true>>(
      "Evolution.Systems.NewtonianMhd.BoundaryConditions.Reflection",
      "Reflection:\n");
  test_ghost_condition<
      3, NewtonianMhd::BoundaryConditions::ConductorReflection<3, true>>(
      "Evolution.Systems.NewtonianMhd.BoundaryConditions.ConductorReflection",
      "ConductorReflection:\n");
  test_demand_outgoing_char_speeds<3>();
}
