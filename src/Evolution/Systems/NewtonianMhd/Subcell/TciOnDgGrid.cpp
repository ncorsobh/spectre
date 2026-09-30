// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Subcell/TciOnDgGrid.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/Systems/NewtonianMhd/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::subcell {
std::tuple<int, evolution::dg::subcell::RdmpTciData> TciOnDgGrid::apply(
    const gsl::not_null<Variables<
        tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure,
                   MagneticField, DivergenceCleaningField>>*>
        dg_prim_vars,
    const Variables<tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity,
                               MagneticFieldCons, DivergenceCleaningFieldCons>>&
        dg_vars,
    const EquationsOfState::EquationOfState<false, 2>& eos,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
    const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
    const evolution::dg::subcell::SubcellOptions& subcell_options,
    const TciOptions& tci_options, const double persson_exponent,
    [[maybe_unused]] const bool element_stays_on_dg) {
  const Variables<tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity,
                             MagneticFieldCons, DivergenceCleaningFieldCons>>
      subcell_vars = evolution::dg::subcell::fd::project(
          dg_vars, dg_mesh, subcell_mesh.extents());
  const Scalar<DataVector>& mass_density = get<MassDensityCons>(dg_vars);
  const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density =
      get<MomentumDensity>(dg_vars);
  const Scalar<DataVector>& energy_density = get<EnergyDensity>(dg_vars);

  const Scalar<DataVector>& subcell_mass_density =
      get<MassDensityCons>(subcell_vars);
  const Scalar<DataVector>& subcell_energy_density =
      get<EnergyDensity>(subcell_vars);

  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{
      {max(max(get(mass_density)), max(get(subcell_mass_density))),
       max(max(get(energy_density)), max(get(subcell_energy_density)))},
      {min(min(get(mass_density)), min(get(subcell_mass_density))),
       min(min(get(energy_density)), min(get(subcell_energy_density)))}};

  NewtonianMhd::PrimitiveFromConservative::apply(
      make_not_null(&get<MassDensity>(*dg_prim_vars)),
      make_not_null(&get<Velocity>(*dg_prim_vars)),
      make_not_null(&get<SpecificInternalEnergy>(*dg_prim_vars)),
      make_not_null(&get<Pressure>(*dg_prim_vars)),
      make_not_null(&get<MagneticField>(*dg_prim_vars)),
      make_not_null(&get<DivergenceCleaningField>(*dg_prim_vars)), mass_density,
      momentum_density, energy_density, get<MagneticFieldCons>(dg_vars),
      get<DivergenceCleaningFieldCons>(dg_vars), eos);

  if (min(get(mass_density)) < tci_options.minimum_density) {
    return {-1, std::move(rdmp_tci_data)};
  }
  if (min(get(get<Pressure>(*dg_prim_vars))) < tci_options.minimum_pressure) {
    return {-2, std::move(rdmp_tci_data)};
  }

  // The internal energy recovered on the subcells is
  // (e - |B|^2/2)/rho - v^2/2, so a cell whose magnetic energy approaches the
  // total energy is about to produce a negative internal energy.
  const Scalar<DataVector> magnetic_field_squared = dot_product(
      get<MagneticFieldCons>(dg_vars), get<MagneticFieldCons>(dg_vars));
  if (max(get(magnetic_field_squared) -
          2.0 * (1.0 - tci_options.safety_factor_for_magnetic_field) *
              get(energy_density)) > 0.0) {
    return {-3, std::move(rdmp_tci_data)};
  }

  if (evolution::dg::subcell::persson_tci(
          mass_density, dg_mesh, persson_exponent,
          subcell_options.persson_num_highest_modes())) {
    return {-4, std::move(rdmp_tci_data)};
  }
  if (evolution::dg::subcell::persson_tci(
          energy_density, dg_mesh, persson_exponent,
          subcell_options.persson_num_highest_modes())) {
    return {-5, std::move(rdmp_tci_data)};
  }

  // Sharp magnetic structure can occur where the fluid variables are smooth,
  // so |B| gets its own Persson check. The cutoff keeps regions with no
  // appreciable field, where |B| is noise-dominated, from tripping it.
  if (tci_options.magnetic_field_cutoff.has_value()) {
    const Scalar<DataVector> magnetic_field_magnitude{
        sqrt(get(magnetic_field_squared))};
    if (max(get(magnetic_field_magnitude)) >
            tci_options.magnetic_field_cutoff.value() and
        evolution::dg::subcell::persson_tci(
            magnetic_field_magnitude, dg_mesh, persson_exponent,
            subcell_options.persson_num_highest_modes())) {
      return {-6, std::move(rdmp_tci_data)};
    }
  }

  if (const int rdmp_tci_status = evolution::dg::subcell::rdmp_tci(
          rdmp_tci_data.max_variables_values,
          rdmp_tci_data.min_variables_values,
          past_rdmp_tci_data.max_variables_values,
          past_rdmp_tci_data.min_variables_values,
          subcell_options.rdmp_delta0(), subcell_options.rdmp_epsilon());
      rdmp_tci_status != 0) {
    return {-(6 + rdmp_tci_status), std::move(rdmp_tci_data)};
  }

  return {0, std::move(rdmp_tci_data)};
}

#define INSTANTIATION(r, data) INSTANTIATION(~, ~)
#undef INSTANTIATION
}  // namespace NewtonianMhd::subcell
