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

using ghost_function_names = tuples::TaggedTuple<
    helpers::Tags::PythonFunctionForErrorMessage<>,
    helpers::Tags::PythonFunctionName<NewtonianMhd::Tags::MassDensityCons>,
    helpers::Tags::PythonFunctionName<NewtonianMhd::Tags::MomentumDensity<>>,
    helpers::Tags::PythonFunctionName<NewtonianMhd::Tags::EnergyDensity>,
    helpers::Tags::PythonFunctionName<NewtonianMhd::Tags::MagneticFieldCons<>>,
    helpers::Tags::PythonFunctionName<
        NewtonianMhd::Tags::DivergenceCleaningFieldCons>,
    helpers::Tags::PythonFunctionName<::Tags::Flux<
        NewtonianMhd::Tags::MassDensityCons, tmpl::size_t<3>, Frame::Inertial>>,
    helpers::Tags::PythonFunctionName<
        ::Tags::Flux<NewtonianMhd::Tags::MomentumDensity<>, tmpl::size_t<3>,
                     Frame::Inertial>>,
    helpers::Tags::PythonFunctionName<::Tags::Flux<
        NewtonianMhd::Tags::EnergyDensity, tmpl::size_t<3>, Frame::Inertial>>,
    helpers::Tags::PythonFunctionName<
        ::Tags::Flux<NewtonianMhd::Tags::MagneticFieldCons<>, tmpl::size_t<3>,
                     Frame::Inertial>>,
    helpers::Tags::PythonFunctionName<
        ::Tags::Flux<NewtonianMhd::Tags::DivergenceCleaningFieldCons,
                     tmpl::size_t<3>, Frame::Inertial>>,
    helpers::Tags::PythonFunctionName<
        NewtonianMhd::Tags::BackgroundMagneticField<>>,
    helpers::Tags::PythonFunctionName<
        hydro::Tags::SpatialVelocity<DataVector, 3>>,
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
template <typename BoundaryConditionType>
void test_ghost_condition(const std::string& python_module,
                          const std::string& option_string) {
  MAKE_GENERATOR(gen);
  helpers::test_boundary_condition_with_python<
      BoundaryConditionType,
      NewtonianMhd::BoundaryConditions::BoundaryCondition,
      NewtonianMhd::System<true>,
      tmpl::list<NewtonianMhd::BoundaryCorrections::Hll<true>>>(
      make_not_null(&gen), python_module,
      ghost_function_names{
          "error", "mass_density_cons", "momentum_density", "energy_density",
          "magnetic_field_cons", "divergence_cleaning_field_cons",
          "flux_mass_density", "flux_momentum_density", "flux_energy_density",
          "flux_magnetic_field", "flux_divergence_cleaning_field",
          "background_magnetic_field", "velocity", "specific_internal_energy"},
      option_string, Index<3 - 1>{3 == 1 ? 1 : 5},
      db::create<
          db::AddSimpleTags<NewtonianMhd::Tags::DivergenceCleaningSpeed>>(1.5),
      positive_ranges());
}

void test_demand_outgoing_char_speeds() {
  MAKE_GENERATOR(gen);
  helpers::test_boundary_condition_with_python<
      NewtonianMhd::BoundaryConditions::DemandOutgoingCharSpeeds<true>,
      NewtonianMhd::BoundaryConditions::BoundaryCondition,
      NewtonianMhd::System<true>,
      tmpl::list<NewtonianMhd::BoundaryCorrections::Hll<true>>,
      tmpl::list<ConvertIdeal>>(
      make_not_null(&gen),
      "Evolution.Systems.NewtonianMhd.BoundaryConditions."
      "DemandOutgoingCharSpeeds",
      tuples::TaggedTuple<helpers::Tags::PythonFunctionForErrorMessage<>>{
          "error"},
      "DemandOutgoingCharSpeeds:\n", Index<3 - 1>{3 == 1 ? 1 : 5},
      db::create<db::AddSimpleTags<hydro::Tags::EquationOfState<false, 2>>>(
          EquationsOfState::IdealFluid<false>{1.3}.get_clone()),
      positive_ranges());
}

// The free-slip condition requires B0 to be tangent to the boundary. The
// python comparison above cannot produce such a B0, since it draws every
// interior tag at random, so the two branches are checked directly here.
void test_reflection_requires_tangent_background_field() {
  const size_t num_points = 5;
  const DataVector one{num_points, 1.0};

  tnsr::i<DataVector, 3, Frame::Inertial> normal_covector{num_points, 0.0};
  get<0>(normal_covector) = one;

  Scalar<DataVector> mass_density_cons{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> momentum_density{num_points};
  Scalar<DataVector> energy_density{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field_cons{num_points};
  Scalar<DataVector> divergence_cleaning_field_cons{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_mass_density{num_points};
  tnsr::IJ<DataVector, 3, Frame::Inertial> flux_momentum_density{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_energy_density{num_points};
  tnsr::IJ<DataVector, 3, Frame::Inertial> flux_magnetic_field{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_divergence_cleaning_field{
      num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> ghost_background{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> velocity{num_points};
  Scalar<DataVector> specific_internal_energy{num_points};

  const Scalar<DataVector> interior_mass_density{DataVector{num_points, 1.2}};
  const Scalar<DataVector> interior_pressure{DataVector{num_points, 0.8}};
  const Scalar<DataVector> interior_specific_internal_energy{
      DataVector{num_points, 1.0}};
  const Scalar<DataVector> interior_divergence_cleaning_field{
      DataVector{num_points, 0.3}};
  tnsr::I<DataVector, 3, Frame::Inertial> interior_velocity{num_points, 0.0};
  get<1>(interior_velocity) = 0.4 * one;
  tnsr::I<DataVector, 3, Frame::Inertial> interior_magnetic_field{num_points,
                                                                  0.0};
  get<2>(interior_magnetic_field) = 0.5 * one;

  const auto call =
      [&](const tnsr::I<DataVector, 3, Frame::Inertial>& background,
          const bool no_slip) {
        NewtonianMhd::BoundaryConditions::detail::reflection_dg_ghost<true>(
            make_not_null(&mass_density_cons), make_not_null(&momentum_density),
            make_not_null(&energy_density), make_not_null(&magnetic_field_cons),
            make_not_null(&divergence_cleaning_field_cons),
            make_not_null(&flux_mass_density),
            make_not_null(&flux_momentum_density),
            make_not_null(&flux_energy_density),
            make_not_null(&flux_magnetic_field),
            make_not_null(&flux_divergence_cleaning_field),
            make_not_null(&ghost_background), make_not_null(&velocity),
            make_not_null(&specific_internal_energy), std::nullopt,
            normal_covector, interior_magnetic_field,
            interior_divergence_cleaning_field, interior_mass_density,
            interior_velocity, interior_specific_internal_energy,
            interior_pressure, 1.5, no_slip, background);
      };

  // Tangent to the x-normal boundary: free slip is admissible.
  tnsr::I<DataVector, 3, Frame::Inertial> tangent_background{num_points, 0.0};
  get<1>(tangent_background) = 0.7 * one;
  call(tangent_background, false);
  CHECK_ITERABLE_APPROX(get<1>(velocity), get<1>(interior_velocity));
  CHECK_ITERABLE_APPROX(get(divergence_cleaning_field_cons),
                        -get(interior_divergence_cleaning_field));
  CHECK_ITERABLE_APPROX(get<1>(ghost_background), get<1>(tangent_background));

  // Threaded by normal flux: free slip must be rejected, no slip accepted.
  tnsr::I<DataVector, 3, Frame::Inertial> normal_background{num_points, 0.0};
  get<0>(normal_background) = 0.7 * one;
  CHECK_THROWS_WITH(
      call(normal_background, false),
      Catch::Matchers::ContainsSubstring("tangent to the boundary"));
  call(normal_background, true);
  const DataVector expected_no_slip_velocity{num_points, -0.4};
  CHECK_ITERABLE_APPROX(get<1>(velocity), expected_no_slip_velocity);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.BoundaryConditions", "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test_ghost_condition<
      NewtonianMhd::BoundaryConditions::ConductorReflection<true>>(
      "Evolution.Systems.NewtonianMhd.BoundaryConditions.ConductorReflection",
      "ConductorReflection:\n");
  test_demand_outgoing_char_speeds();
  test_reflection_requires_tangent_background_field();
}
