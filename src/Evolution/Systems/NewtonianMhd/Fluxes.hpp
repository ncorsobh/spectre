// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
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

namespace detail {
/// \brief Compute the fluxes of the Newtonian MHD conservative variables.
///
/// Called internally by `ComputeFluxes::apply` and by
/// `TimeDerivativeTerms::apply`. The temporary `magnetic_pressure` holds
/// \f$p_{\rm mag} = |B_1|^2/2 + B_0\cdot B_1\f$ on output.
template <size_t Dim>
void fluxes_impl(
    gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_cons_flux,
    gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
    gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,
    gsl::not_null<tnsr::IJ<DataVector, Dim>*> magnetic_field_flux,
    gsl::not_null<tnsr::I<DataVector, Dim>*> divergence_cleaning_field_flux,
    gsl::not_null<Scalar<DataVector>*> magnetic_pressure,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, Dim>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, Dim>& background_magnetic_field,
    double glm_cleaning_speed);
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
 */
template <size_t Dim>
struct ComputeFluxes {
  using return_tags = tmpl::list<
      ::Tags::Flux<Tags::MassDensityCons, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Tags::MomentumDensity<Dim>, tmpl::size_t<Dim>,
                   Frame::Inertial>,
      ::Tags::Flux<Tags::EnergyDensity, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Tags::MagneticField<Dim>, tmpl::size_t<Dim>,
                   Frame::Inertial>,
      ::Tags::Flux<Tags::DivergenceCleaningField, tmpl::size_t<Dim>,
                   Frame::Inertial>>;

  using argument_tags =
      tmpl::list<Tags::MomentumDensity<Dim>, Tags::EnergyDensity,
                 Tags::MagneticField<Dim>, Tags::DivergenceCleaningField,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::Pressure<DataVector>,
                 Tags::BackgroundMagneticField<Dim>, Tags::GlmCleaningSpeed>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_cons_flux,
      gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
      gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,
      gsl::not_null<tnsr::IJ<DataVector, Dim>*> magnetic_field_flux,
      gsl::not_null<tnsr::I<DataVector, Dim>*> divergence_cleaning_field_flux,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, Dim>& magnetic_field,
      const Scalar<DataVector>& divergence_cleaning_field,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& pressure,
      const tnsr::I<DataVector, Dim>& background_magnetic_field,
      double glm_cleaning_speed);
};

}  // namespace NewtonianMhd
