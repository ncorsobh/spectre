// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeDerivativeDecisions.hpp"
#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/Source.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace NewtonianMhd {
namespace detail {
/// Shared body of `TimeDerivativeTerms::apply`, see there for the physics.
template <bool UseBackgroundMagneticField = false>
void time_derivative_impl(
    gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_mass_density,
    gsl::not_null<tnsr::I<DataVector, 3>*> non_flux_terms_dt_momentum_density,
    gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_energy_density,
    gsl::not_null<tnsr::I<DataVector, 3>*> non_flux_terms_dt_magnetic_field,
    gsl::not_null<Scalar<DataVector>*>
        non_flux_terms_dt_divergence_cleaning_field,
    gsl::not_null<tnsr::I<DataVector, 3>*> mass_density_cons_flux,
    gsl::not_null<tnsr::IJ<DataVector, 3>*> momentum_density_flux,
    gsl::not_null<tnsr::I<DataVector, 3>*> energy_density_flux,
    gsl::not_null<tnsr::IJ<DataVector, 3>*> magnetic_field_flux,
    gsl::not_null<tnsr::I<DataVector, 3>*> divergence_cleaning_field_flux,
    gsl::not_null<Scalar<DataVector>*> magnetic_pressure,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, 3>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, 3>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field,
    const tnsr::I<DataVector, 3>& velocity, const Scalar<DataVector>& pressure,
    double divergence_cleaning_speed, double constraint_damping_parameter,
    const EquationsOfState::EquationOfState<false, 2>& eos,
    const tnsr::I<DataVector, 3>& coords, double time,
    const Sources::Source<UseBackgroundMagneticField>& source,
    BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
        background_magnetic_field);
}  // namespace detail

/*!
 * \brief Compute the time derivative of the conserved variables for the
 * Newtonian MHD system.
 *
 * The only non-flux term produced here is the GLM constraint damping
 *
 * \f{align*}
 *   \partial_t \psi \mathrel{+}= -\alpha c_h \psi ,
 * \f}
 *
 * applied everywhere in the domain.  The user-specified volume source term is
 * evaluated afterwards and adds to all five equations.
 *
 * With `UseBackgroundMagneticField == true` the stored \f$B_0\f$ is copied into
 * a temporary so that the DG machinery can project it onto element faces for
 * the boundary corrections and boundary conditions, which can only see evolved
 * variables, fluxes and time-derivative temporaries. With the splitting
 * disabled neither the temporary nor the copy exists.
 */
template <bool UseBackgroundMagneticField = false>
struct TimeDerivativeTerms;

/// \cond
template <>
struct TimeDerivativeTerms<false> {
 private:
  struct MagneticPressure : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  using temporary_tags = tmpl::list<MagneticPressure>;
  using argument_tags = tmpl::list<
      Tags::MassDensityCons, Tags::MomentumDensity<>, Tags::EnergyDensity,
      Tags::MagneticFieldCons<>, Tags::DivergenceCleaningFieldCons,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::Pressure<DataVector>, Tags::DivergenceCleaningSpeed,
      Tags::ConstraintDampingParameter, hydro::Tags::EquationOfState<false, 2>,
      domain::Tags::Coordinates<3, Frame::Inertial>, ::Tags::Time,
      NewtonianMhd::Tags::SourceTerm<false>>;

  static evolution::dg::TimeDerivativeDecisions<3> apply(
      const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_mass_density,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          non_flux_terms_dt_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_energy_density,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          non_flux_terms_dt_magnetic_field,
      const gsl::not_null<Scalar<DataVector>*>
          non_flux_terms_dt_divergence_cleaning_field,

      const gsl::not_null<tnsr::I<DataVector, 3>*> mass_density_cons_flux,
      const gsl::not_null<tnsr::IJ<DataVector, 3>*> momentum_density_flux,
      const gsl::not_null<tnsr::I<DataVector, 3>*> energy_density_flux,
      const gsl::not_null<tnsr::IJ<DataVector, 3>*> magnetic_field_flux,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          divergence_cleaning_field_flux,

      const gsl::not_null<Scalar<DataVector>*> magnetic_pressure,

      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, 3>& magnetic_field,
      const Scalar<DataVector>& divergence_cleaning_field,
      const tnsr::I<DataVector, 3>& velocity,
      const Scalar<DataVector>& pressure,
      const double divergence_cleaning_speed,
      const double constraint_damping_parameter,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const tnsr::I<DataVector, 3>& coords, const double time,
      const Sources::Source<>& source) {
    detail::time_derivative_impl<false>(
        non_flux_terms_dt_mass_density, non_flux_terms_dt_momentum_density,
        non_flux_terms_dt_energy_density, non_flux_terms_dt_magnetic_field,
        non_flux_terms_dt_divergence_cleaning_field, mass_density_cons_flux,
        momentum_density_flux, energy_density_flux, magnetic_field_flux,
        divergence_cleaning_field_flux, magnetic_pressure, mass_density_cons,
        momentum_density, energy_density, magnetic_field,
        divergence_cleaning_field, velocity, pressure,
        divergence_cleaning_speed, constraint_damping_parameter, eos, coords,
        time, source, {});
    return {true};
  }
};

template <>
struct TimeDerivativeTerms<true> {
 private:
  struct MagneticPressure : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  using temporary_tags =
      tmpl::list<MagneticPressure, Tags::BackgroundMagneticField<>>;
  using argument_tags = tmpl::list<
      Tags::MassDensityCons, Tags::MomentumDensity<>, Tags::EnergyDensity,
      Tags::MagneticFieldCons<>, Tags::DivergenceCleaningFieldCons,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::Pressure<DataVector>, Tags::DivergenceCleaningSpeed,
      Tags::ConstraintDampingParameter, hydro::Tags::EquationOfState<false, 2>,
      domain::Tags::Coordinates<3, Frame::Inertial>, ::Tags::Time,
      NewtonianMhd::Tags::SourceTerm<true>,
      Tags::BackgroundMagneticFieldVolume<>>;

  static evolution::dg::TimeDerivativeDecisions<3> apply(
      const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_mass_density,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          non_flux_terms_dt_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_energy_density,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          non_flux_terms_dt_magnetic_field,
      const gsl::not_null<Scalar<DataVector>*>
          non_flux_terms_dt_divergence_cleaning_field,

      const gsl::not_null<tnsr::I<DataVector, 3>*> mass_density_cons_flux,
      const gsl::not_null<tnsr::IJ<DataVector, 3>*> momentum_density_flux,
      const gsl::not_null<tnsr::I<DataVector, 3>*> energy_density_flux,
      const gsl::not_null<tnsr::IJ<DataVector, 3>*> magnetic_field_flux,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          divergence_cleaning_field_flux,

      const gsl::not_null<Scalar<DataVector>*> magnetic_pressure,
      const gsl::not_null<tnsr::I<DataVector, 3>*> background_magnetic_field,

      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, 3>& magnetic_field,
      const Scalar<DataVector>& divergence_cleaning_field,
      const tnsr::I<DataVector, 3>& velocity,
      const Scalar<DataVector>& pressure,
      const double divergence_cleaning_speed,
      const double constraint_damping_parameter,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const tnsr::I<DataVector, 3>& coords, const double time,
      const Sources::Source<true>& source,
      const tnsr::I<DataVector, 3>& background_magnetic_field_volume) {
    // The DG machinery can only project evolved variables, fluxes and
    // time-derivative temporaries onto faces, so the stored B0 is copied here.
    *background_magnetic_field = background_magnetic_field_volume;
    detail::time_derivative_impl<true>(
        non_flux_terms_dt_mass_density, non_flux_terms_dt_momentum_density,
        non_flux_terms_dt_energy_density, non_flux_terms_dt_magnetic_field,
        non_flux_terms_dt_divergence_cleaning_field, mass_density_cons_flux,
        momentum_density_flux, energy_density_flux, magnetic_field_flux,
        divergence_cleaning_field_flux, magnetic_pressure, mass_density_cons,
        momentum_density, energy_density, magnetic_field,
        divergence_cleaning_field, velocity, pressure,
        divergence_cleaning_speed, constraint_damping_parameter, eos, coords,
        time, source, *background_magnetic_field);
    return {true};
  }
};
/// \endcond

}  // namespace NewtonianMhd
