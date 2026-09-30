// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TciOptions.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename TagsList>
class Variables;
/// \endcond

namespace NewtonianMhd::subcell {
/*!
 * \brief Troubled-cell indicator applied to the FD solution, deciding whether
 * the element may return to DG.
 *
 * Reconstructs the conserved variables to the DG grid, computes the primitive
 * variables on both grids, and returns which check, if any, flagged the
 * element. The checks run in the order below and the first to fire returns:
 *
 * - `+1` the minimum mass density fell below
 *   `TciOptions::MinimumValueOfDensity`
 * - `+2` the minimum pressure fell below `TciOptions::MinimumValueOfPressure`
 * - `+3` \f$|B|^2 > 2(1 - \epsilon_B)e\f$ somewhere
 * - `+4` the Persson TCI flagged the reconstructed mass density
 * - `+5` the Persson TCI flagged the reconstructed energy density
 * - `+6` the Persson TCI flagged the reconstructed \f$|B|\f$
 * - `+(6 + n)` the RDMP TCI flagged variable `n`
 * - `0` the element may return to DG
 */
class TciOnFdGrid {
 private:
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

 public:
  using return_tags = tmpl::list<::Tags::Variables<
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure,
                 MagneticField, DivergenceCleaningField>>>;
  using argument_tags =
      tmpl::list<::Tags::Variables<tmpl::list<MassDensityCons, MomentumDensity,
                                              EnergyDensity, MagneticFieldCons,
                                              DivergenceCleaningFieldCons>>,
                 hydro::Tags::EquationOfState<false, 2>, domain::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::Mesh<3>,
                 evolution::dg::subcell::Tags::DataForRdmpTci,
                 evolution::dg::subcell::Tags::SubcellOptions<3>,
                 NewtonianMhd::subcell::Tags::TciOptions>;

  static std::tuple<int, evolution::dg::subcell::RdmpTciData> apply(
      gsl::not_null<Variables<
          tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure,
                     MagneticField, DivergenceCleaningField>>*>
          subcell_grid_prim_vars,
      const Variables<tmpl::list<MassDensityCons, MomentumDensity,
                                 EnergyDensity, MagneticFieldCons,
                                 DivergenceCleaningFieldCons>>& subcell_vars,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
      const evolution::dg::subcell::SubcellOptions& subcell_options,
      const TciOptions& tci_options, double persson_exponent,
      bool need_rdmp_data_only);
};
}  // namespace NewtonianMhd::subcell
