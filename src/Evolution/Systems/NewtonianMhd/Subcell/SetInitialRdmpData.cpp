// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Subcell/SetInitialRdmpData.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace NewtonianMhd::subcell {
void SetInitialRdmpData::apply(
    const gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
    const Variables<tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity,
                               MagneticFieldCons, DivergenceCleaningFieldCons>>&
        vars,
    const evolution::dg::subcell::ActiveGrid active_grid,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh) {
  if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
    const Scalar<DataVector>& mass_density = get<MassDensityCons>(vars);
    const Scalar<DataVector>& energy_density = get<EnergyDensity>(vars);
    *rdmp_tci_data = {.max_variables_values = {max(get(mass_density)),
                                               max(get(energy_density))},
                      .min_variables_values = {min(get(mass_density)),
                                               min(get(energy_density))}};
  } else {
    const Scalar<DataVector>& dg_mass_density = get<MassDensityCons>(vars);
    const Scalar<DataVector>& dg_energy_density = get<EnergyDensity>(vars);
    const auto subcell_mass_density = evolution::dg::subcell::fd::project(
        get(dg_mass_density), dg_mesh, subcell_mesh.extents());
    const auto subcell_energy_density = evolution::dg::subcell::fd::project(
        get(dg_energy_density), dg_mesh, subcell_mesh.extents());

    using std::max;
    using std::min;
    *rdmp_tci_data = {
        .max_variables_values = {max(max(get(dg_mass_density)),
                                     max(subcell_mass_density)),
                                 max(max(get(dg_energy_density)),
                                     max(subcell_energy_density))},
        .min_variables_values = {
            min(min(get(dg_mass_density)), min(subcell_mass_density)),
            min(min(get(dg_energy_density)), min(subcell_energy_density))}};
  }
}

#define INSTANTIATION(r, data) INSTANTIATION(~, ~)
#undef INSTANTIATION
}  // namespace NewtonianMhd::subcell
