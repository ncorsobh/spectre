// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd {

void ConservativeFromPrimitive::apply(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3>*> magnetic_field_cons,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, 3>& velocity,
    const Scalar<DataVector>& specific_internal_energy,
    const tnsr::I<DataVector, 3>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field) {
  get(*mass_density_cons) = get(mass_density);

  for (size_t i = 0; i < 3; ++i) {
    momentum_density->get(i) = get(mass_density) * velocity.get(i);
  }

  get(*energy_density) =
      get(mass_density) * (0.5 * get(dot_product(velocity, velocity)) +
                           get(specific_internal_energy)) +
      0.5 * get(dot_product(magnetic_field, magnetic_field));

  for (size_t i = 0; i < 3; ++i) {
    magnetic_field_cons->get(i) = magnetic_field.get(i);
  }
  get(*divergence_cleaning_field_cons) = get(divergence_cleaning_field);
}

}  // namespace NewtonianMhd

#define INSTANTIATE(_, data)

INSTANTIATE(~, ~)

#undef INSTANTIATE
