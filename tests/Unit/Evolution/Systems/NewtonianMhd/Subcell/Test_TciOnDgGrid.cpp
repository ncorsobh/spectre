// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TciOptions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace {
enum class TestThis : uint8_t {
  AllGood,
  SmallDensity,
  SmallPressure,
  PerssonDensity,
  PerssonEnergyDensity,
  RdmpMassDensity,
  RdmpEnergyDensity,
  PerssonMagneticField,
  MagneticEnergyTooLarge
};

void test(const TestThis test_this) {
  using MassDensityCons = NewtonianMhd::Tags::MassDensityCons;
  using EnergyDensity = NewtonianMhd::Tags::EnergyDensity;
  using MomentumDensity = NewtonianMhd::Tags::MomentumDensity<>;
  using MagneticFieldCons = NewtonianMhd::Tags::MagneticFieldCons<>;
  using DivergenceCleaningFieldCons =
      NewtonianMhd::Tags::DivergenceCleaningFieldCons;

  using MassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
  using SpecificInternalEnergy =
      hydro::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;

  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  using cons_tags = tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity,
                               MagneticFieldCons, DivergenceCleaningFieldCons>;
  using ConsVars = Variables<cons_tags>;
  using prim_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure,
                 MagneticField, DivergenceCleaningField>;
  using PrimVars = Variables<prim_tags>;

  const double persson_exponent = 4.0;
  const double adiabatic_index = 5.0 / 3.0;
  const double uniform_energy_density = 2.0;
  PrimVars dg_prims{dg_mesh.number_of_grid_points(), 1.0e-7};
  std::unique_ptr<EquationsOfState::EquationOfState<false, 2>> eos =
      std::make_unique<EquationsOfState::IdealFluid<false>>(adiabatic_index);
  if (test_this == TestThis::SmallDensity) {
    get(get<MassDensity>(dg_prims))[dg_mesh.number_of_grid_points() / 2] =
        0.1 * 1.0e-18;
  } else if (test_this == TestThis::SmallPressure) {
    get(get<Pressure>(dg_prims))[dg_mesh.number_of_grid_points() / 2] =
        0.1 * 1.0e-18;
  } else if (test_this == TestThis::PerssonDensity) {
    get(get<MassDensity>(dg_prims))[dg_mesh.number_of_grid_points() / 2] =
        1.0e18;
  } else if (test_this == TestThis::PerssonEnergyDensity) {
    get(get<Pressure>(dg_prims))[dg_mesh.number_of_grid_points() / 2] = 1.0e18;
  } else if (test_this == TestThis::PerssonMagneticField) {
    // A current sheet: |B| is sharp but the gas pressure compensates so that
    // the total energy density stays uniform. The mass and energy density
    // checks therefore cannot see it, leaving only the |B| Persson check.
    get(get<MassDensity>(dg_prims)) = 1.0;
    get<0>(get<MagneticField>(dg_prims)) = 1.0;
    get<0>(get<MagneticField>(dg_prims))[dg_mesh.number_of_grid_points() / 2] =
        1.5;
    get(get<Pressure>(dg_prims)) =
        (adiabatic_index - 1.0) *
        (uniform_energy_density -
         0.5 * get(dot_product(get<MagneticField>(dg_prims),
                               get<MagneticField>(dg_prims))));
  } else if (test_this == TestThis::MagneticEnergyTooLarge) {
    // Uniform, so no Persson or RDMP check fires, and small enough that the
    // recovered pressure stays positive: only the |B|^2 bound catches it.
    get<0>(get<MagneticField>(dg_prims)) = 1.0e-2;
  }

  get<SpecificInternalEnergy>(dg_prims) =
      eos->specific_internal_energy_from_density_and_pressure(
          get<MassDensity>(dg_prims), get<Pressure>(dg_prims));

  const evolution::dg::subcell::SubcellOptions subcell_options{
      persson_exponent,
      1_st,
      1.0e-18,
      1.0e-4,
      false,
      false,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
      false,
      fd::DerivativeOrder::Two,
      1,
      1,
      1};

  // The bound |B|^2 <= 2(1 - eps_B) e degenerates into the positive-pressure
  // check as eps_B -> 0, so the case testing it uses a loose safety factor.
  const NewtonianMhd::subcell::TciOptions tci_options{
      1.0e-18, 1.0e-18,
      test_this == TestThis::MagneticEnergyTooLarge ? 0.5 : 1.0e-12, 1.0e-4};

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Variables<cons_tags>, ::Tags::Variables<prim_tags>,
      ::domain::Tags::Mesh<3>, ::evolution::dg::subcell::Tags::Mesh<3>,
      hydro::Tags::EquationOfState<false, 2>,
      evolution::dg::subcell::Tags::SubcellOptions<3>,
      NewtonianMhd::subcell::Tags::TciOptions,
      evolution::dg::subcell::Tags::DataForRdmpTci>>(
      ConsVars{dg_mesh.number_of_grid_points()}, dg_prims, dg_mesh,
      subcell_mesh, std::move(eos), subcell_options, tci_options,
      evolution::dg::subcell::RdmpTciData{});
  db::mutate_apply<NewtonianMhd::ConservativeFromPrimitive>(
      make_not_null(&box));

  // Set the RDMP TCI past data.
  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData past_rdmp_tci_data{
      .max_variables_values = {max(max(get(get<MassDensityCons>(box))),
                                   max(evolution::dg::subcell::fd::project(
                                       get(get<MassDensityCons>(box)), dg_mesh,
                                       subcell_mesh.extents()))),
                               max(max(get(get<EnergyDensity>(box))),
                                   max(evolution::dg::subcell::fd::project(
                                       get(get<EnergyDensity>(box)), dg_mesh,
                                       subcell_mesh.extents())))},
      .min_variables_values = {min(min(get(get<MassDensityCons>(box))),
                                   min(evolution::dg::subcell::fd::project(
                                       get(get<MassDensityCons>(box)), dg_mesh,
                                       subcell_mesh.extents()))),
                               min(min(get(get<EnergyDensity>(box))),
                                   min(evolution::dg::subcell::fd::project(
                                       get(get<EnergyDensity>(box)), dg_mesh,
                                       subcell_mesh.extents())))}};

  const evolution::dg::subcell::RdmpTciData expected_rdmp_tci_data =
      past_rdmp_tci_data;

  // Modify past data if we are expected an RDMP TCI failure.
  db::mutate<evolution::dg::subcell::Tags::DataForRdmpTci>(
      [&past_rdmp_tci_data, &test_this](const auto rdmp_tci_data_ptr) {
        *rdmp_tci_data_ptr = past_rdmp_tci_data;
        if (test_this == TestThis::RdmpMassDensity) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[0] *= 1.01;
        } else if (test_this == TestThis::RdmpEnergyDensity) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[1] *= 1.01;
        }
      },
      make_not_null(&box));

  const bool element_stays_on_dg = false;
  const std::tuple<int, evolution::dg::subcell::RdmpTciData> result =
      db::mutate_apply<NewtonianMhd::subcell::TciOnDgGrid>(
          make_not_null(&box), persson_exponent, element_stays_on_dg);

  CHECK_ITERABLE_APPROX(get<1>(result).max_variables_values,
                        expected_rdmp_tci_data.max_variables_values);
  CHECK_ITERABLE_APPROX(get<1>(result).min_variables_values,
                        expected_rdmp_tci_data.min_variables_values);
  // The status code says which check fired, so each case pins its own.
  const int expected_status = [&test_this]() {
    switch (test_this) {
      case TestThis::AllGood:
        return 0;
      case TestThis::SmallDensity:
        return -1;
      case TestThis::SmallPressure:
        return -2;
      case TestThis::MagneticEnergyTooLarge:
        return -3;
      case TestThis::PerssonDensity:
        return -4;
      case TestThis::PerssonEnergyDensity:
        return -5;
      case TestThis::PerssonMagneticField:
        return -6;
      case TestThis::RdmpMassDensity:
        return -7;
      case TestThis::RdmpEnergyDensity:
        return -8;
      default:
        ERROR("Unhandled TestThis");
    }
  }();
  CHECK(std::get<0>(result) == expected_status);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianMhd.Subcell.TciOnDgGrid",
                  "[Unit][Evolution]") {
  for (const auto test_this :
       {TestThis::AllGood, TestThis::SmallDensity, TestThis::SmallPressure,
        TestThis::PerssonDensity, TestThis::PerssonEnergyDensity,
        TestThis::RdmpMassDensity, TestThis::RdmpEnergyDensity,
        TestThis::PerssonMagneticField, TestThis::MagneticEnergyTooLarge}) {
    test(test_this);
  }
}
