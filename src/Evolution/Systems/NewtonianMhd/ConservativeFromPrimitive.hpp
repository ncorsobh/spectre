// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianMhd/TagsDeclarations.hpp"
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
 * \brief Compute the conservative variables from the primitive variables for
 * Newtonian MHD.
 *
 * \f{align*}
 *   \rho_{\rm cons} &= \rho \\
 *   (\rho v)^i &= \rho v^i \\
 *   e &= \rho \left(\tfrac{1}{2} v^2 + \epsilon\right) + \tfrac{1}{2} |B_1|^2
 * \\
 *   B_{1,{\rm cons}}^i &= B_1^i \\
 *   \psi_{\rm cons} &= \psi
 * \f}
 *
 * The perturbation magnetic field \f$B_1^i\f$ and GLM cleaning field \f$\psi\f$
 * are identical between primitive and conservative representations.  The static
 * background field \f$B_0\f$ does not enter the energy variable directly
 * because we evolve only the perturbation energy.
 */
template <size_t Dim>
struct ConservativeFromPrimitive {
  using return_tags =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity, Tags::MagneticFieldCons<Dim>,
                 Tags::DivergenceCleaningFieldCons>;

  using argument_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::MagneticField<DataVector, Dim>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      gsl::not_null<tnsr::I<DataVector, Dim>*> magnetic_field_cons,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,
      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& specific_internal_energy,
      const tnsr::I<DataVector, Dim>& magnetic_field,
      const Scalar<DataVector>& divergence_cleaning_field);
};
}  // namespace NewtonianMhd
