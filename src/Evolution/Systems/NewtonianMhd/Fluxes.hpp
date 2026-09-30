// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
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

namespace detail {
/// \brief Compute the fluxes of the Newtonian MHD conservative variables.
///
/// Called internally by `ComputeFluxes::apply` and by
/// `TimeDerivativeTerms::apply`. The temporary `magnetic_pressure` holds
/// \f$p_{\rm mag} = |B_1|^2/2 + B_0\cdot B_1\f$ on output, dropping the
/// \f$B_0\f$ term when the splitting is disabled.
template <bool UseBackgroundMagneticField = false>
void fluxes_impl(
    gsl::not_null<tnsr::I<DataVector, 3>*> mass_density_cons_flux,
    gsl::not_null<tnsr::IJ<DataVector, 3>*> momentum_density_flux,
    gsl::not_null<tnsr::I<DataVector, 3>*> energy_density_flux,
    gsl::not_null<tnsr::IJ<DataVector, 3>*> magnetic_field_flux,
    gsl::not_null<tnsr::I<DataVector, 3>*> divergence_cleaning_field_flux,
    gsl::not_null<Scalar<DataVector>*> magnetic_pressure,
    const tnsr::I<DataVector, 3>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, 3>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field,
    const tnsr::I<DataVector, 3>& velocity, const Scalar<DataVector>& pressure,
    double divergence_cleaning_speed,
    BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
        background_magnetic_field);
}  // namespace detail

/*!
 * \brief Compute the fluxes of the Newtonian MHD conservative variables with
 * an optional static background magnetic field \f$B_0\f$.
 *
 * Only the perturbation field \f$B_1\f$ is evolved.  Define
 * \f$B_{\rm tot}^i = B_0^i + B_1^i\f$ and
 * \f$p_{\rm tot} = P + \tfrac{1}{2}|B_1|^2 + B_0\cdot B_1\f$.
 *
 * \f{align*}
 *   F^j(\rho) &= (\rho v)^j \\
 *   F^j(\rho v^i) &= \rho v^i v^j + p_{\rm tot} \delta^{ij}
 *                    - B_{\rm tot}^j B_1^i - B_0^i B_1^j \\
 *   F^j(e) &= (e + p_{\rm tot}) v^j - B_{\rm tot}^j (v \cdot B_1) \\
 *   F^j(B_1^i) &= v^j B_{\rm tot}^i - B_{\rm tot}^j v^i + \delta^{ij}\psi \\
 *   F^j(\psi) &= c_h^2 B_1^j
 * \f}
 *
 * The \f$B_0\f$-only pieces of the standard MHD flux cancel identically in the
 * divergence because \f$B_0\f$ is curl-free and divergence-free by
 * construction, so they are not included here.
 *
 * With `UseBackgroundMagneticField == false` every \f$B_0\f$ term above is
 * dropped at compile time and \f$B_0\f$ is not an argument tag at all, so these
 * become the standard MHD fluxes with no residual cost.
 *
 * \note For the rank-2 fluxes the *first* index is the flux direction, as
 * `divergence` and `normal_dot_flux` require: `flux.get(j, i)` is
 * \f$F^j\f$ of the \f$i\f$th component. The momentum flux is symmetric so the
 * distinction is invisible there, but the induction flux is antisymmetric and
 * transposing it flips the sign of the induction term.
 */
template <bool UseBackgroundMagneticField = false>
struct ComputeFluxes {
  using return_tags = tmpl::list<
      ::Tags::Flux<Tags::MassDensityCons, tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Flux<Tags::MomentumDensity<>, tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Flux<Tags::EnergyDensity, tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Flux<Tags::MagneticFieldCons<>, tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Flux<Tags::DivergenceCleaningFieldCons, tmpl::size_t<3>,
                   Frame::Inertial>>;

  // The background field is last so that omitting it simply shortens the
  // argument list.
  using argument_tags = tmpl::append<
      tmpl::list<Tags::MomentumDensity<>, Tags::EnergyDensity,
                 Tags::MagneticFieldCons<>, Tags::DivergenceCleaningFieldCons,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::Pressure<DataVector>,
                 Tags::DivergenceCleaningSpeed>,
      background_magnetic_field_tag_list<Tags::BackgroundMagneticField<>,
                                         UseBackgroundMagneticField>>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, 3>*> mass_density_cons_flux,
      gsl::not_null<tnsr::IJ<DataVector, 3>*> momentum_density_flux,
      gsl::not_null<tnsr::I<DataVector, 3>*> energy_density_flux,
      gsl::not_null<tnsr::IJ<DataVector, 3>*> magnetic_field_flux,
      gsl::not_null<tnsr::I<DataVector, 3>*> divergence_cleaning_field_flux,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, 3>& magnetic_field,
      const Scalar<DataVector>& divergence_cleaning_field,
      const tnsr::I<DataVector, 3>& velocity,
      const Scalar<DataVector>& pressure, double divergence_cleaning_speed,
      BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
          background_magnetic_field = {});
};

}  // namespace NewtonianMhd
