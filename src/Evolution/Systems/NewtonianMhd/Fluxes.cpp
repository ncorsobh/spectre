// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd {
namespace detail {

template <size_t Dim, bool UseBackgroundMagneticField>
void fluxes_impl(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_cons_flux,
    const gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,
    const gsl::not_null<tnsr::IJ<DataVector, Dim>*> magnetic_field_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        divergence_cleaning_field_flux,
    const gsl::not_null<Scalar<DataVector>*> magnetic_pressure,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, Dim>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& pressure, const double divergence_cleaning_speed,
    const BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
        background_magnetic_field) {
  // Magnetic pressure contribution to the total pressure (excludes P):
  //   p_mag = |B1|^2 / 2 + B0 . B1
  get(*magnetic_pressure) =
      0.5 * get(dot_product(magnetic_field, magnetic_field));
  if constexpr (UseBackgroundMagneticField) {
    get(*magnetic_pressure) +=
        get(dot_product(background_magnetic_field, magnetic_field));
  }

  // The velocity - magnetic field contraction that appears in the energy flux.
  const Scalar<DataVector> v_dot_b1 = dot_product(velocity, magnetic_field);

  for (size_t i = 0; i < Dim; ++i) {
    // Mass flux:  F^i(rho) = (rho v)^i = momentum_density^i
    mass_density_cons_flux->get(i) = momentum_density.get(i);
  }

  // Momentum flux tensor: flux[j][i] = F^j(rho v^i)
  //   = rho v^i v^j + p_tot delta^{ij} - B_total^j B1^i - B0^i B1^j
  for (size_t j = 0; j < Dim; ++j) {
    for (size_t i = 0; i < Dim; ++i) {
      momentum_density_flux->get(j, i) =
          momentum_density.get(i) * velocity.get(j) -
          magnetic_field.get(i) * magnetic_field.get(j);
      if constexpr (UseBackgroundMagneticField) {
        momentum_density_flux->get(j, i) -=
            magnetic_field.get(i) * background_magnetic_field.get(j) +
            background_magnetic_field.get(i) * magnetic_field.get(j);
      }
    }
    momentum_density_flux->get(j, j) += get(pressure) + get(*magnetic_pressure);
  }

  // Energy flux:
  //   F^j(e) = (e + p_tot) v^j - B_total^j (v . B1) - B0^j psi
  // The last term is the GLM contribution: e differs from the total energy by
  // B0.B1, so F(e) picks up -B0_i F^j(B1^i), whose delta^{ij} psi part is the
  // only piece not already accounted for by the B0 terms above.
  for (size_t j = 0; j < Dim; ++j) {
    energy_density_flux->get(j) =
        (get(energy_density) + get(pressure) + get(*magnetic_pressure)) *
            velocity.get(j) -
        magnetic_field.get(j) * get(v_dot_b1);
    if constexpr (UseBackgroundMagneticField) {
      energy_density_flux->get(j) -=
          background_magnetic_field.get(j) *
          (get(v_dot_b1) + get(divergence_cleaning_field));
    }
  }

  // Magnetic-field flux tensor: flux[j][i] = F^j(B1^i)
  //   = v^j B_total^i - B_total^j v^i + delta^{ij} psi
  // This one is antisymmetric, so the index order matters: the flux direction
  // must come first for `divergence` and `normal_dot_flux`.
  for (size_t j = 0; j < Dim; ++j) {
    for (size_t i = 0; i < Dim; ++i) {
      magnetic_field_flux->get(j, i) = velocity.get(j) * magnetic_field.get(i) -
                                       magnetic_field.get(j) * velocity.get(i);
      if constexpr (UseBackgroundMagneticField) {
        magnetic_field_flux->get(j, i) +=
            velocity.get(j) * background_magnetic_field.get(i) -
            background_magnetic_field.get(j) * velocity.get(i);
      }
    }
    magnetic_field_flux->get(j, j) += get(divergence_cleaning_field);
  }

  // GLM divergence-cleaning-field flux: F^j(psi) = c_h^2 B1^j
  const double c_h_squared =
      divergence_cleaning_speed * divergence_cleaning_speed;
  for (size_t j = 0; j < Dim; ++j) {
    divergence_cleaning_field_flux->get(j) =
        c_h_squared * magnetic_field.get(j);
  }
}

}  // namespace detail

template <size_t Dim, bool UseBackgroundMagneticField>
void ComputeFluxes<Dim, UseBackgroundMagneticField>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_cons_flux,
    const gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,
    const gsl::not_null<tnsr::IJ<DataVector, Dim>*> magnetic_field_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        divergence_cleaning_field_flux,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, Dim>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& pressure, const double divergence_cleaning_speed,
    const BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
        background_magnetic_field) {
  Scalar<DataVector> magnetic_pressure{get<0>(momentum_density).size()};
  detail::fluxes_impl<Dim, UseBackgroundMagneticField>(
      mass_density_cons_flux, momentum_density_flux, energy_density_flux,
      magnetic_field_flux, divergence_cleaning_field_flux,
      make_not_null(&magnetic_pressure), momentum_density, energy_density,
      magnetic_field, divergence_cleaning_field, velocity, pressure,
      divergence_cleaning_speed, background_magnetic_field);
}

}  // namespace NewtonianMhd

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define USE_BG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                 \
  template void NewtonianMhd::detail::fluxes_impl<DIM(data), USE_BG(data)>(  \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*> mass_density_cons_flux, \
      gsl::not_null<tnsr::IJ<DataVector, DIM(data)>*> momentum_density_flux, \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*> energy_density_flux,    \
      gsl::not_null<tnsr::IJ<DataVector, DIM(data)>*> magnetic_field_flux,   \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*>                         \
          divergence_cleaning_field_flux,                                    \
      gsl::not_null<Scalar<DataVector>*> magnetic_pressure,                  \
      const tnsr::I<DataVector, DIM(data)>& momentum_density,                \
      const Scalar<DataVector>& energy_density,                              \
      const tnsr::I<DataVector, DIM(data)>& magnetic_field,                  \
      const Scalar<DataVector>& divergence_cleaning_field,                   \
      const tnsr::I<DataVector, DIM(data)>& velocity,                        \
      const Scalar<DataVector>& pressure, double divergence_cleaning_speed,  \
      NewtonianMhd::BackgroundMagneticFieldArgument<DIM(data), USE_BG(data)> \
          background_magnetic_field);                                        \
  template struct NewtonianMhd::ComputeFluxes<DIM(data), USE_BG(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (true, false))

#undef DIM
#undef USE_BG
#undef INSTANTIATE
