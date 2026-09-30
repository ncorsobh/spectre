// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/NewtonianMhd/Initialization/BackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t Dim = 3;
using BackgroundField = NewtonianMhd::Tags::BackgroundMagneticFieldVolume<Dim>;
using MagneticField = hydro::Tags::MagneticField<DataVector, Dim>;
using primitive_variables_tag =
    typename NewtonianMhd::System<Dim, true>::primitive_variables_tag;

NewtonianMhd::Solutions::AlfvenWave make_initial_data() {
  return NewtonianMhd::Solutions::AlfvenWave{
      {{1.0, 2.0, -1.0}}, 1.5, 0.8, 1.1, 0.3, 5.0 / 3.0};
}

tnsr::I<DataVector, Dim, Frame::Inertial> sample_coordinates() {
  tnsr::I<DataVector, Dim, Frame::Inertial> coords{3_st};
  get<0>(coords) = DataVector{2.0, -3.5, 0.0};
  get<1>(coords) = DataVector{4.0, 1.0, -2.5};
  get<2>(coords) = DataVector{-1.0, 2.5, 5.0};
  return coords;
}

void test_sets_the_background_from_the_initial_data() {
  const auto coords = sample_coordinates();
  const auto initial_data = make_initial_data();

  auto box = db::create<db::AddSimpleTags<
      BackgroundField, domain::Tags::Coordinates<Dim, Frame::Inertial>,
      evolution::initial_data::Tags::InitialData>>(
      typename BackgroundField::type{}, coords,
      std::unique_ptr<evolution::initial_data::InitialData>{
          initial_data.get_clone()});

  db::mutate_apply<NewtonianMhd::Initialization::BackgroundMagneticField<Dim>>(
      make_not_null(&box));

  const auto expected = get<BackgroundField>(
      initial_data.variables(coords, 0.0, tmpl::list<BackgroundField>{}));
  CHECK_ITERABLE_APPROX(db::get<BackgroundField>(box), expected);
}

// The evolved field must become the perturbation, leaving the total field
// recoverable as B0 + B1.
void test_subtracts_the_background_from_the_primitives() {
  const auto coords = sample_coordinates();
  const auto initial_data = make_initial_data();
  const auto background = get<BackgroundField>(
      initial_data.variables(coords, 0.0, tmpl::list<BackgroundField>{}));

  typename primitive_variables_tag::type prims{get<0>(coords).size(), 0.0};
  auto& total_field = get<MagneticField>(prims);
  get<0>(total_field) = DataVector{0.3, -1.2, 2.0};
  get<1>(total_field) = DataVector{-0.6, 0.9, 0.1};
  get<2>(total_field) = DataVector{1.1, 0.4, -0.8};
  const auto expected_total_field = total_field;

  auto box =
      db::create<db::AddSimpleTags<primitive_variables_tag, BackgroundField>>(
          prims, background);
  db::mutate_apply<
      NewtonianMhd::Initialization::SubtractBackgroundMagneticField<Dim>>(
      make_not_null(&box));

  const auto& perturbation =
      get<MagneticField>(db::get<primitive_variables_tag>(box));
  for (size_t i = 0; i < Dim; ++i) {
    CAPTURE(i);
    const DataVector recovered = perturbation.get(i) + background.get(i);
    CHECK_ITERABLE_APPROX(recovered, expected_total_field.get(i));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianMhd.Initialization.BackgroundMagneticField",
                  "[Unit][Evolution]") {
  test_sets_the_background_from_the_initial_data();
  test_subtracts_the_background_from_the_primitives();
}
