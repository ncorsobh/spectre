// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim, bool UseBackgroundMagneticField>
void test(const gsl::not_null<std::mt19937*> gen,
          const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  using Fluxes = NewtonianMhd::ComputeFluxes<Dim, UseBackgroundMagneticField>;
  const size_t num_pts = 5;
  const double divergence_cleaning_speed = 1.3;

  // The cleaning speed is an argument tag but a double rather than a field, so
  // the variables are listed explicitly instead of taken from argument_tags.
  using field_tags = tmpl::append<
      typename Fluxes::return_tags,
      tmpl::list<NewtonianMhd::Tags::MomentumDensity<Dim>,
                 NewtonianMhd::Tags::EnergyDensity,
                 NewtonianMhd::Tags::MagneticFieldCons<Dim>,
                 NewtonianMhd::Tags::DivergenceCleaningFieldCons,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::Pressure<DataVector>>,
      NewtonianMhd::background_magnetic_field_tag_list<
          NewtonianMhd::Tags::BackgroundMagneticField<Dim>,
          UseBackgroundMagneticField>>;

  auto vars =
      make_with_random_values<Variables<field_tags>>(gen, dist, num_pts);

  Variables<typename Fluxes::return_tags> expected_fluxes{num_pts};
  const auto apply_fluxes = [&expected_fluxes, &vars, &divergence_cleaning_speed](
                                const auto&... background_magnetic_field) {
    Fluxes::apply(
        make_not_null(
            &get<::Tags::Flux<NewtonianMhd::Tags::MassDensityCons,
                              tmpl::size_t<Dim>, Frame::Inertial>>(
                expected_fluxes)),
        make_not_null(
            &get<::Tags::Flux<NewtonianMhd::Tags::MomentumDensity<Dim>,
                              tmpl::size_t<Dim>, Frame::Inertial>>(
                expected_fluxes)),
        make_not_null(
            &get<::Tags::Flux<NewtonianMhd::Tags::EnergyDensity,
                              tmpl::size_t<Dim>, Frame::Inertial>>(
                expected_fluxes)),
        make_not_null(
            &get<::Tags::Flux<NewtonianMhd::Tags::MagneticFieldCons<Dim>,
                              tmpl::size_t<Dim>, Frame::Inertial>>(
                expected_fluxes)),
        make_not_null(
            &get<::Tags::Flux<NewtonianMhd::Tags::DivergenceCleaningFieldCons,
                              tmpl::size_t<Dim>, Frame::Inertial>>(
                expected_fluxes)),
        get<NewtonianMhd::Tags::MomentumDensity<Dim>>(vars),
        get<NewtonianMhd::Tags::EnergyDensity>(vars),
        get<NewtonianMhd::Tags::MagneticFieldCons<Dim>>(vars),
        get<NewtonianMhd::Tags::DivergenceCleaningFieldCons>(vars),
        get<hydro::Tags::SpatialVelocity<DataVector, Dim>>(vars),
        get<hydro::Tags::Pressure<DataVector>>(vars), divergence_cleaning_speed,
        background_magnetic_field...);
  };
  if constexpr (UseBackgroundMagneticField) {
    apply_fluxes(get<NewtonianMhd::Tags::BackgroundMagneticField<Dim>>(vars));
  } else {
    apply_fluxes();
  }

  NewtonianMhd::subcell::compute_fluxes<Dim, UseBackgroundMagneticField>(
      make_not_null(&vars), divergence_cleaning_speed);

  tmpl::for_each<typename Fluxes::return_tags>(
      [&expected_fluxes, &vars](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(vars), get<tag>(expected_fluxes));
      });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianMhd.Subcell.ComputeFluxes",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  test<1, false>(make_not_null(&gen), make_not_null(&dist));
  test<2, false>(make_not_null(&gen), make_not_null(&dist));
  test<3, false>(make_not_null(&gen), make_not_null(&dist));
  test<1, true>(make_not_null(&gen), make_not_null(&dist));
  test<2, true>(make_not_null(&gen), make_not_null(&dist));
  test<3, true>(make_not_null(&gen), make_not_null(&dist));
}
