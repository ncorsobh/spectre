// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/TimeDerivativeTerms.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd::detail {

template <size_t Dim, bool UseBackgroundMagneticField>
void time_derivative_impl(
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        non_flux_terms_dt_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_energy_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        non_flux_terms_dt_magnetic_field,
    const gsl::not_null<Scalar<DataVector>*>
        non_flux_terms_dt_divergence_cleaning_field,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_cons_flux,
    const gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,
    const gsl::not_null<tnsr::IJ<DataVector, Dim>*> magnetic_field_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        divergence_cleaning_field_flux,
    const gsl::not_null<Scalar<DataVector>*> magnetic_pressure,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, Dim>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& pressure, const double divergence_cleaning_speed,
    const double constraint_damping_parameter,
    const EquationsOfState::EquationOfState<false, 2>& eos,
    const tnsr::I<DataVector, Dim>& coords, const double time,
    const Sources::Source<Dim, UseBackgroundMagneticField>& source,
    const BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
        background_magnetic_field) {
  fluxes_impl<Dim, UseBackgroundMagneticField>(
      mass_density_cons_flux, momentum_density_flux, energy_density_flux,
      magnetic_field_flux, divergence_cleaning_field_flux, magnetic_pressure,
      momentum_density, energy_density, magnetic_field,
      divergence_cleaning_field, velocity, pressure, divergence_cleaning_speed,
      background_magnetic_field);

  get(*non_flux_terms_dt_mass_density) = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    non_flux_terms_dt_momentum_density->get(i) = 0.0;
    non_flux_terms_dt_magnetic_field->get(i) = 0.0;
  }
  get(*non_flux_terms_dt_energy_density) = 0.0;
  get(*non_flux_terms_dt_divergence_cleaning_field) =
      -constraint_damping_parameter * divergence_cleaning_speed *
      get(divergence_cleaning_field);

  const auto eos_2d = eos.promote_to_2d_eos();
  source(non_flux_terms_dt_mass_density, non_flux_terms_dt_momentum_density,
         non_flux_terms_dt_energy_density, non_flux_terms_dt_magnetic_field,
         non_flux_terms_dt_divergence_cleaning_field, mass_density_cons,
         momentum_density, energy_density, magnetic_field,
         divergence_cleaning_field, velocity, pressure,
         background_magnetic_field, *eos_2d, coords, time);
}

}  // namespace NewtonianMhd::detail

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define USE_BG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                 \
  template void                                                              \
  NewtonianMhd::detail::time_derivative_impl<DIM(data), USE_BG(data)>(       \
      gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_mass_density,     \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*>                         \
          non_flux_terms_dt_momentum_density,                                \
      gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_energy_density,   \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*>                         \
          non_flux_terms_dt_magnetic_field,                                  \
      gsl::not_null<Scalar<DataVector>*>                                     \
          non_flux_terms_dt_divergence_cleaning_field,                       \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*> mass_density_cons_flux, \
      gsl::not_null<tnsr::IJ<DataVector, DIM(data)>*> momentum_density_flux, \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*> energy_density_flux,    \
      gsl::not_null<tnsr::IJ<DataVector, DIM(data)>*> magnetic_field_flux,   \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*>                         \
          divergence_cleaning_field_flux,                                    \
      gsl::not_null<Scalar<DataVector>*> magnetic_pressure,                  \
      const Scalar<DataVector>& mass_density_cons,                           \
      const tnsr::I<DataVector, DIM(data)>& momentum_density,                \
      const Scalar<DataVector>& energy_density,                              \
      const tnsr::I<DataVector, DIM(data)>& magnetic_field,                  \
      const Scalar<DataVector>& divergence_cleaning_field,                   \
      const tnsr::I<DataVector, DIM(data)>& velocity,                        \
      const Scalar<DataVector>& pressure, double divergence_cleaning_speed,  \
      double constraint_damping_parameter,                                   \
      const EquationsOfState::EquationOfState<false, 2>& eos,                \
      const tnsr::I<DataVector, DIM(data)>& coords, double time,             \
      const NewtonianMhd::Sources::Source<DIM(data), USE_BG(data)>& source,                \
      NewtonianMhd::BackgroundMagneticFieldArgument<DIM(data), USE_BG(data)> \
          background_magnetic_field);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (true, false))

#undef DIM
#undef USE_BG
#undef INSTANTIATE
