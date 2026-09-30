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
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TciOptions.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename T>
class Variables;
/// \endcond

namespace NewtonianMhd::subcell {
/*!
 * \brief Troubled-cell indicator applied to the DG solution.
 *
 * Computes the primitive variables on the DG grid, mutating them in the
 * DataBox, and returns which check, if any, flagged the element. The checks
 * run in the order below and the first to fire returns:
 *
 * - `-1` the minimum mass density fell below
 *   `TciOptions::MinimumValueOfDensity`
 * - `-2` the minimum pressure fell below `TciOptions::MinimumValueOfPressure`
 * - `-3` \f$|B|^2 > 2(1 - \epsilon_B)e\f$ somewhere, so the internal energy
 *   recovered from the conserved variables is about to go negative
 * - `-4` the Persson TCI flagged the mass density
 * - `-5` the Persson TCI flagged the energy density. Applying it to both
 *   densities is what flags contact discontinuities.
 * - `-6` the Persson TCI flagged \f$|B|\f$. Sharp magnetic structure such as
 *   a current sheet can occur where the fluid variables are smooth, so it gets
 *   its own check; the element is skipped when the largest \f$|B|\f$ is below
 *   `TciOptions::MagneticFieldCutoff`, where the field is noise-dominated.
 * - `-(6 + n)` the RDMP TCI flagged variable `n`
 * - `0` the element is not troubled
 */
class TciOnDgGrid {
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
          dg_prim_vars,
      const Variables<
          tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity,
                     MagneticFieldCons, DivergenceCleaningFieldCons>>& dg_vars,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
      const evolution::dg::subcell::SubcellOptions& subcell_options,
      const TciOptions& tci_options, double persson_exponent,
      bool element_stays_on_dg);
};
}  // namespace NewtonianMhd::subcell
