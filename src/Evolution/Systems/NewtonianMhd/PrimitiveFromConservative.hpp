// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianMhd/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace NewtonianMhd {

/*!
 * \brief Compute the primitive variables from the conservative variables for
 * Newtonian MHD.
 *
 * \f{align*}
 *   \rho &= \rho_{\rm cons} \\
 *   v^i &= (\rho v)^i / \rho \\
 *   \epsilon &= \left(e - \tfrac{1}{2}|B_1|^2\right)/\rho
 *              - \tfrac{1}{2} v^2 \\
 *   B_1^i &= B_{1,{\rm cons}}^i \\
 *   \psi &= \psi_{\rm cons}
 * \f}
 *
 * Pressure is then obtained from the equation of state, \f$P =
 * P(\rho,\epsilon)\f$. No root-finding is required in the Newtonian limit.
 */
template <size_t Dim>
struct PrimitiveFromConservative {
  using return_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::MagneticField<DataVector, Dim>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

  using argument_tags =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity, Tags::MagneticFieldCons<Dim>,
                 Tags::DivergenceCleaningFieldCons,
                 hydro::Tags::EquationOfState<false, 2>>;

  template <size_t ThermodynamicDim>
  static void apply(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<tnsr::I<DataVector, Dim>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, Dim>& magnetic_field_cons,
      const Scalar<DataVector>& divergence_cleaning_field_cons,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state);
};
}  // namespace NewtonianMhd
