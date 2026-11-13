// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd {

template <size_t Dim, bool EvolveMagneticField>
void ConservativeFromPrimitive<Dim, EvolveMagneticField>::apply(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& specific_internal_energy) {
  get(*mass_density_cons) = get(mass_density);

  for (size_t i = 0; i < Dim; ++i) {
    momentum_density->get(i) = get(mass_density) * velocity.get(i);
  }

  get(*energy_density) =
      get(mass_density) * (0.5 * get(dot_product(velocity, velocity)) +
                           get(specific_internal_energy));
}

template <size_t Dim, bool EvolveMagneticField>
void ConservativeFromPrimitive<Dim, EvolveMagneticField>::apply(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3>*> magnetic_field_cons,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& specific_internal_energy,
    const tnsr::I<DataVector, 3>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field) {
  apply(mass_density_cons, momentum_density, energy_density, mass_density,
        velocity, specific_internal_energy);

  get(*energy_density) +=
      0.5 * get(dot_product(magnetic_field, magnetic_field));

  for (size_t i = 0; i < 3; ++i) {
    magnetic_field_cons->get(i) = magnetic_field.get(i);
  }
  get(*divergence_cleaning_field_cons) = get(divergence_cleaning_field);
}

}  // namespace NewtonianMhd

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define EVOLVE_B(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                         \
  template struct NewtonianMhd::ConservativeFromPrimitive<DIM(data), \
                                                          EVOLVE_B(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (false))
GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (true))

#undef DIM
#undef EVOLVE_B
#undef INSTANTIATE
