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

template <size_t Dim>
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
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, Dim>& background_magnetic_field,
    const double glm_cleaning_speed) {
  // Magnetic pressure contribution to the total pressure (excludes P):
  //   p_mag = |B1|^2 / 2 + B0 . B1
  get(*magnetic_pressure) =
      0.5 * get(dot_product(magnetic_field, magnetic_field)) +
      get(dot_product(background_magnetic_field, magnetic_field));

  // Total scalar pressure appearing on the diagonal of the momentum flux and
  // in the energy flux:
  //   p_tot = P + p_mag = P + |B1|^2 / 2 + B0 . B1
  // The velocity - magnetic field contraction that appears in the energy flux:
  //   v . B1
  const Scalar<DataVector> v_dot_b1 = dot_product(velocity, magnetic_field);

  for (size_t i = 0; i < Dim; ++i) {
    // Mass flux:  F^i(rho) = (rho v)^i = momentum_density^i
    mass_density_cons_flux->get(i) = momentum_density.get(i);
  }

  for (size_t i = 0; i < Dim; ++i) {
    // Momentum flux tensor: flux[i][j] = F^j(rho v^i)
    //   = rho v^i v^j + p_tot delta^{ij}
    //     - B_total^j B1^i - B0^i B1^j
    // Expanding B_total = B0 + B1:
    //   = momentum^i * v^j
    //     - B1^i (B0^j + B1^j)
    //     - B0^i B1^j
    for (size_t j = 0; j < Dim; ++j) {
      momentum_density_flux->get(i, j) =
          momentum_density.get(i) * velocity.get(j) -
          magnetic_field.get(i) *
              (background_magnetic_field.get(j) + magnetic_field.get(j)) -
          background_magnetic_field.get(i) * magnetic_field.get(j);
    }
    // Add p_tot on the diagonal
    momentum_density_flux->get(i, i) += get(pressure) + get(*magnetic_pressure);
  }

  // Energy flux:
  //   F^j(e) = (e + p_tot) v^j - B_total^j (v . B1)
  for (size_t j = 0; j < Dim; ++j) {
    energy_density_flux->get(j) =
        (get(energy_density) + get(pressure) + get(*magnetic_pressure)) *
            velocity.get(j) -
        (background_magnetic_field.get(j) + magnetic_field.get(j)) *
            get(v_dot_b1);
  }

  // Magnetic-field flux tensor: flux[i][j] = F^j(B1^i)
  //   = v^j B_total^i - B_total^j v^i + delta^{ij} psi
  for (size_t i = 0; i < Dim; ++i) {
    const double bg_i_scalar = 0.0;  // unused placeholder, silences warnings
    (void)bg_i_scalar;
    for (size_t j = 0; j < Dim; ++j) {
      magnetic_field_flux->get(i, j) =
          velocity.get(j) *
              (background_magnetic_field.get(i) + magnetic_field.get(i)) -
          (background_magnetic_field.get(j) + magnetic_field.get(j)) *
              velocity.get(i);
    }
    magnetic_field_flux->get(i, i) += get(divergence_cleaning_field);
  }

  // GLM divergence-cleaning-field flux: F^j(psi) = c_h^2 B1^j
  const double c_h_squared = glm_cleaning_speed * glm_cleaning_speed;
  for (size_t j = 0; j < Dim; ++j) {
    divergence_cleaning_field_flux->get(j) =
        c_h_squared * magnetic_field.get(j);
  }
}

}  // namespace detail

template <size_t Dim>
void ComputeFluxes<Dim>::apply(
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
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, Dim>& background_magnetic_field,
    const double glm_cleaning_speed) {
  Scalar<DataVector> magnetic_pressure{get<0>(momentum_density).size()};
  detail::fluxes_impl(
      mass_density_cons_flux, momentum_density_flux, energy_density_flux,
      magnetic_field_flux, divergence_cleaning_field_flux,
      make_not_null(&magnetic_pressure), momentum_density, energy_density,
      magnetic_field, divergence_cleaning_field, velocity, pressure,
      background_magnetic_field, glm_cleaning_speed);
}

}  // namespace NewtonianMhd

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template void NewtonianMhd::detail::fluxes_impl<DIM(data)>(                \
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
      const Scalar<DataVector>& pressure,                                    \
      const tnsr::I<DataVector, DIM(data)>& background_magnetic_field,       \
      double glm_cleaning_speed);                                            \
  template class NewtonianMhd::ComputeFluxes<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
