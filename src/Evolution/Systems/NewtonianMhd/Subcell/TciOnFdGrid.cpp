// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Subcell/TciOnFdGrid.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/Systems/NewtonianMhd/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::subcell {
std::tuple<bool, evolution::dg::subcell::RdmpTciData> TciOnFdGrid::apply(
    const gsl::not_null<Variables<
        tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure,
                   MagneticField, DivergenceCleaningField>>*>
        subcell_grid_prim_vars,
    const Variables<tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity,
                               MagneticFieldCons, DivergenceCleaningFieldCons>>&
        subcell_vars,
    const EquationsOfState::EquationOfState<false, 2>& eos,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
    const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
    const evolution::dg::subcell::SubcellOptions& subcell_options,
    const TciOptions& tci_options, const double persson_exponent,
    const bool need_rdmp_data_only) {
  const Scalar<DataVector>& subcell_mass_density =
      get<MassDensityCons>(subcell_vars);
  const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_momentum_density =
      get<MomentumDensity>(subcell_vars);
  const Scalar<DataVector>& subcell_energy_density =
      get<EnergyDensity>(subcell_vars);
  const auto dg_vars = evolution::dg::subcell::fd::reconstruct(
      subcell_vars, dg_mesh, subcell_mesh.extents(),
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);
  const Scalar<DataVector>& dg_mass_density = get<MassDensityCons>(dg_vars);
  const tnsr::I<DataVector, 3, Frame::Inertial>& dg_momentum_density =
      get<MomentumDensity>(dg_vars);
  const Scalar<DataVector>& dg_energy_density = get<EnergyDensity>(dg_vars);

  NewtonianMhd::PrimitiveFromConservative::apply(
      make_not_null(&get<MassDensity>(*subcell_grid_prim_vars)),
      make_not_null(&get<Velocity>(*subcell_grid_prim_vars)),
      make_not_null(&get<SpecificInternalEnergy>(*subcell_grid_prim_vars)),
      make_not_null(&get<Pressure>(*subcell_grid_prim_vars)),
      make_not_null(&get<MagneticField>(*subcell_grid_prim_vars)),
      make_not_null(&get<DivergenceCleaningField>(*subcell_grid_prim_vars)),
      subcell_mass_density, subcell_momentum_density, subcell_energy_density,
      get<MagneticFieldCons>(subcell_vars),
      get<DivergenceCleaningFieldCons>(subcell_vars), eos);
  Variables<tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure,
                       MagneticField, DivergenceCleaningField>>
      dg_grid_prim_vars{get(dg_energy_density).size()};
  NewtonianMhd::PrimitiveFromConservative::apply(
      make_not_null(&get<MassDensity>(dg_grid_prim_vars)),
      make_not_null(&get<Velocity>(dg_grid_prim_vars)),
      make_not_null(&get<SpecificInternalEnergy>(dg_grid_prim_vars)),
      make_not_null(&get<Pressure>(dg_grid_prim_vars)),
      make_not_null(&get<MagneticField>(dg_grid_prim_vars)),
      make_not_null(&get<DivergenceCleaningField>(dg_grid_prim_vars)),
      dg_mass_density, dg_momentum_density, dg_energy_density,
      get<MagneticFieldCons>(dg_vars),
      get<DivergenceCleaningFieldCons>(dg_vars), eos);

  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{
      {max(get(subcell_mass_density)), max(get(subcell_energy_density))},
      {min(get(subcell_mass_density)), min(get(subcell_energy_density))}};

  const evolution::dg::subcell::RdmpTciData rdmp_tci_data_for_check{
      {max(rdmp_tci_data.max_variables_values[0], max(get(dg_mass_density))),
       max(rdmp_tci_data.max_variables_values[1], max(get(dg_energy_density)))},
      {min(rdmp_tci_data.min_variables_values[0], min(get(dg_mass_density))),
       min(rdmp_tci_data.min_variables_values[1],
           min(get(dg_energy_density)))}};

  if (need_rdmp_data_only) {
    return {false, rdmp_tci_data};
  }

  // The internal energy recovered from the conserved variables is
  // (e - |B|^2/2)/rho - v^2/2, so a cell whose magnetic energy approaches the
  // total energy is about to produce a negative internal energy.
  const Scalar<DataVector> subcell_magnetic_field_squared =
      dot_product(get<MagneticFieldCons>(subcell_vars),
                  get<MagneticFieldCons>(subcell_vars));
  const bool magnetic_energy_too_large =
      max(get(subcell_magnetic_field_squared) -
          2.0 * (1.0 - tci_options.safety_factor_for_magnetic_field) *
              get(subcell_energy_density)) > 0.0;

  bool cell_is_troubled =
      evolution::dg::subcell::rdmp_tci(
          rdmp_tci_data_for_check.max_variables_values,
          rdmp_tci_data_for_check.min_variables_values,
          past_rdmp_tci_data.max_variables_values,
          past_rdmp_tci_data.min_variables_values,
          subcell_options.rdmp_delta0(), subcell_options.rdmp_epsilon()) or
      min(min(get(subcell_mass_density)), min(get(dg_mass_density))) <
          tci_options.minimum_density or
      min(min(get(get<Pressure>(*subcell_grid_prim_vars))),
          min(get(get<Pressure>(dg_grid_prim_vars)))) <
          tci_options.minimum_pressure or
      magnetic_energy_too_large or
      evolution::dg::subcell::persson_tci(
          dg_mass_density, dg_mesh, persson_exponent,
          subcell_options.persson_num_highest_modes()) or
      evolution::dg::subcell::persson_tci(
          dg_energy_density, dg_mesh, persson_exponent,
          subcell_options.persson_num_highest_modes());

  // An element may only return to DG if the reconstructed magnetic field is
  // smooth there too, so |B| is checked on the DG grid as it is above.
  if (not cell_is_troubled and tci_options.magnetic_field_cutoff.has_value()) {
    const Scalar<DataVector> dg_magnetic_field_magnitude{sqrt(get(dot_product(
        get<MagneticFieldCons>(dg_vars), get<MagneticFieldCons>(dg_vars))))};
    cell_is_troubled =
        max(get(dg_magnetic_field_magnitude)) >
            tci_options.magnetic_field_cutoff.value() and
        evolution::dg::subcell::persson_tci(
            dg_magnetic_field_magnitude, dg_mesh, persson_exponent,
            subcell_options.persson_num_highest_modes());
  }

  return {cell_is_troubled, std::move(rdmp_tci_data)};
}

#define INSTANTIATION(r, data) INSTANTIATION(~, ~)
#undef INSTANTIATION
}  // namespace NewtonianMhd::subcell
