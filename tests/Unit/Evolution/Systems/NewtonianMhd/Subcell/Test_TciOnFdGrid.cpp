// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TciOnFdGrid.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TciOptions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace {
enum class TestThis {
  AllGood,
  SmallDensitySubcell,
  SmallPressureSubcell,
  PerssonDensity,
  PerssonEnergyDensity,
  RdmpMassDensity,
  RdmpEnergyDensity,
  PerssonMagneticField,
  MagneticEnergyTooLarge
};

template <size_t Dim>
void test(const TestThis test_this) {
  CAPTURE(Dim);
  CAPTURE(test_this);
  using MassDensityCons = NewtonianMhd::Tags::MassDensityCons;
  using EnergyDensity = NewtonianMhd::Tags::EnergyDensity;
  using MomentumDensity = NewtonianMhd::Tags::MomentumDensity<Dim>;
  using MagneticFieldCons = NewtonianMhd::Tags::MagneticFieldCons<Dim>;
  using DivergenceCleaningFieldCons =
      NewtonianMhd::Tags::DivergenceCleaningFieldCons;

  using MassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      hydro::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, Dim>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;

  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

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
  std::unique_ptr<EquationsOfState::EquationOfState<false, 2>> eos =
      std::make_unique<EquationsOfState::IdealFluid<false>>(adiabatic_index);
  ConsVars subcell_cons{subcell_mesh.number_of_grid_points()};
  PrimVars subcell_prim{subcell_mesh.number_of_grid_points(), 1.0e-7};

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

  if (test_this == TestThis::PerssonEnergyDensity) {
    get(get<Pressure>(subcell_prim))[subcell_mesh.number_of_grid_points() / 2] =
        1.0;
  } else if (test_this == TestThis::SmallDensitySubcell) {
    get(get<MassDensity>(
        subcell_prim))[subcell_mesh.number_of_grid_points() / 2] =
        0.1 * 1.0e-18;
  } else if (test_this == TestThis::SmallPressureSubcell) {
    get(get<Pressure>(subcell_prim))[subcell_mesh.number_of_grid_points() / 2] =
        0.1 * 1.0e-18;
  } else if (test_this == TestThis::PerssonDensity) {
    get(get<MassDensity>(subcell_prim))[dg_mesh.number_of_grid_points() / 2] =
        1.0e-6;
  } else if (test_this == TestThis::PerssonMagneticField) {
    // A current sheet: |B| is sharp but the gas pressure compensates so that
    // the total energy density stays uniform. The mass and energy density
    // checks therefore cannot see it, leaving only the |B| Persson check.
    get(get<MassDensity>(subcell_prim)) = 1.0;
    get<0>(get<MagneticField>(subcell_prim)) = 1.0;
    get<0>(get<MagneticField>(
        subcell_prim))[subcell_mesh.number_of_grid_points() / 2] = 1.5;
    get(get<Pressure>(subcell_prim)) =
        (adiabatic_index - 1.0) *
        (uniform_energy_density -
         0.5 * get(dot_product(get<MagneticField>(subcell_prim),
                               get<MagneticField>(subcell_prim))));
  } else if (test_this == TestThis::MagneticEnergyTooLarge) {
    // Uniform, so no Persson or RDMP check fires, and small enough that the
    // recovered pressure stays positive: only the |B|^2 bound catches it.
    get<0>(get<MagneticField>(subcell_prim)) = 1.0e-2;
  }

  get<SpecificInternalEnergy>(subcell_prim) =
      eos->specific_internal_energy_from_density_and_pressure(
          get<MassDensity>(subcell_prim), get<Pressure>(subcell_prim));

  // The bound |B|^2 <= 2(1 - eps_B) e degenerates into the positive-pressure
  // check as eps_B -> 0, so the case testing it uses a loose safety factor.
  const NewtonianMhd::subcell::TciOptions tci_options{
      1.0e-18, 1.0e-18,
      test_this == TestThis::MagneticEnergyTooLarge ? 0.5 : 1.0e-12, 1.0e-4};

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Variables<cons_tags>, ::Tags::Variables<prim_tags>,
      ::domain::Tags::Mesh<Dim>, ::evolution::dg::subcell::Tags::Mesh<Dim>,
      hydro::Tags::EquationOfState<false, 2>,
      evolution::dg::subcell::Tags::SubcellOptions<Dim>,
      NewtonianMhd::subcell::Tags::TciOptions,
      evolution::dg::subcell::Tags::DataForRdmpTci>>(
      subcell_cons, subcell_prim, dg_mesh, subcell_mesh, std::move(eos),
      subcell_options, tci_options, evolution::dg::subcell::RdmpTciData{});
  db::mutate_apply<NewtonianMhd::ConservativeFromPrimitive<Dim>>(
      make_not_null(&box));

  // Set the RDMP TCI past data.
  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData past_rdmp_tci_data{};

  past_rdmp_tci_data.max_variables_values = DataVector{
      max(max(get(db::get<MassDensityCons>(box))),
          max(evolution::dg::subcell::fd::reconstruct(
              get(db::get<MassDensityCons>(box)), dg_mesh,
              subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim))),
      max(max(get(db::get<EnergyDensity>(box))),
          max(evolution::dg::subcell::fd::reconstruct(
              get(db::get<EnergyDensity>(box)), dg_mesh, subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim)))};
  past_rdmp_tci_data.min_variables_values = DataVector{
      min(min(get(db::get<MassDensityCons>(box))),
          min(evolution::dg::subcell::fd::reconstruct(
              get(db::get<MassDensityCons>(box)), dg_mesh,
              subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim))),
      min(min(get(db::get<EnergyDensity>(box))),
          min(evolution::dg::subcell::fd::reconstruct(
              get(db::get<EnergyDensity>(box)), dg_mesh, subcell_mesh.extents(),
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim)))};

  evolution::dg::subcell::RdmpTciData expected_rdmp_tci_data{};
  expected_rdmp_tci_data.max_variables_values =
      DataVector{max(get(db::get<MassDensityCons>(box))),
                 max(get(db::get<EnergyDensity>(box)))};
  expected_rdmp_tci_data.min_variables_values =
      DataVector{min(get(db::get<MassDensityCons>(box))),
                 min(get(db::get<EnergyDensity>(box)))};

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

  const auto result = db::mutate_apply<NewtonianMhd::subcell::TciOnFdGrid<Dim>>(
      make_not_null(&box), persson_exponent, false);
  CHECK(get<1>(result) == expected_rdmp_tci_data);

  if (test_this == TestThis::AllGood) {
    CHECK_FALSE(std::get<0>(result));
  } else {
    CHECK(std::get<0>(result));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianMhd.Subcell.TciOnFdGrid",
                  "[Unit][Evolution]") {
  for (const auto test_this :
       {TestThis::AllGood, TestThis::SmallDensitySubcell,
        TestThis::SmallPressureSubcell, TestThis::PerssonDensity,
        TestThis::PerssonEnergyDensity, TestThis::RdmpMassDensity,
        TestThis::RdmpEnergyDensity, TestThis::PerssonMagneticField,
        TestThis::MagneticEnergyTooLarge}) {
    test<1>(test_this);
    test<2>(test_this);
    test<3>(test_this);
  }
}
