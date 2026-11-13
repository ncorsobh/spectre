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
 * \brief Compute the conservative variables from the primitive variables.
 *
 * \f{align*}
 * S^i &= \rho v^i \\
 * e &= \dfrac{1}{2}\rho v^2 + \rho\epsilon + \dfrac{1}{2}B^2
 * \f}
 *
 * where \f$S^i\f$ is the momentum density, \f$e\f$ is the energy density,
 * \f$\rho\f$ is the mass density, \f$v^i\f$ is the velocity, \f$v^2\f$ is its
 * magnitude squared, \f$\epsilon\f$ is the specific internal energy and
 * \f$B^i\f$ is the magnetic field. In addition, this method returns the mass
 * density as a conservative.
 *
 * The magnetic term in \f$e\f$ is present only when the magnetic field is
 * evolved, in which case the magnetic field and the divergence-cleaning field
 * are also returned as conservatives; they are numerically identical to their
 * primitive counterparts.
 */
template <size_t Dim, bool EvolveMagneticField = true>
struct ConservativeFromPrimitive {
  static_assert(Dim == 3 or not EvolveMagneticField,
                "Magnetic field evolution is only supported at Dim == 3 since "
                "the vector dimensionality is tied to that of the system.");

  using return_tags = tmpl::append<
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity>,
      tmpl::conditional_t<EvolveMagneticField,
                          tmpl::list<Tags::MagneticFieldCons<>,
                                     Tags::DivergenceCleaningFieldCons>,
                          tmpl::list<>>>;

  using argument_tags = tmpl::append<
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>>,
      tmpl::conditional_t<
          EvolveMagneticField,
          tmpl::list<hydro::Tags::MagneticField<DataVector, 3>,
                     hydro::Tags::DivergenceCleaningField<DataVector>>,
          tmpl::list<>>>;

  static void apply(gsl::not_null<Scalar<DataVector>*> mass_density_cons,
                    gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
                    gsl::not_null<Scalar<DataVector>*> energy_density,
                    const Scalar<DataVector>& mass_density,
                    const tnsr::I<DataVector, Dim>& velocity,
                    const Scalar<DataVector>& specific_internal_energy);

  static void apply(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      gsl::not_null<tnsr::I<DataVector, 3>*> magnetic_field_cons,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,
      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& specific_internal_energy,
      const tnsr::I<DataVector, 3>& magnetic_field,
      const Scalar<DataVector>& divergence_cleaning_field);
};
}  // namespace NewtonianMhd
