// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/PrimitiveFromConservative.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd {

template <size_t Dim, bool EvolveMagneticField>
template <size_t ThermodynamicDim>
void PrimitiveFromConservative<Dim, EvolveMagneticField>::apply(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) {
  get(*mass_density) = get(mass_density_cons);

  // store inverse mass density here in order to save allocation
  get(*specific_internal_energy) = 1.0 / get(mass_density_cons);
  for (size_t i = 0; i < Dim; ++i) {
    velocity->get(i) = momentum_density.get(i) * get(*specific_internal_energy);
  }

  get(*specific_internal_energy) *= get(energy_density);
  get(*specific_internal_energy) -=
      (0.5 * get(dot_product(*velocity, *velocity)));

  if constexpr (ThermodynamicDim == 1) {
    *pressure = equation_of_state.pressure_from_density(mass_density_cons);
  } else if constexpr (ThermodynamicDim == 2) {
    *pressure = equation_of_state.pressure_from_density_and_energy(
        mass_density_cons, *specific_internal_energy);
  }
}

template <size_t Dim, bool EvolveMagneticField>
template <size_t ThermodynamicDim>
void PrimitiveFromConservative<Dim, EvolveMagneticField>::apply(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<tnsr::I<DataVector, 3>*> magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, 3>& magnetic_field_cons,
    const Scalar<DataVector>& divergence_cleaning_field_cons,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) {
  get(*mass_density) = get(mass_density_cons);

  for (size_t i = 0; i < 3; ++i) {
    magnetic_field->get(i) = magnetic_field_cons.get(i);
  }
  get(*divergence_cleaning_field) = get(divergence_cleaning_field_cons);

  // store inverse mass density here in order to save allocation
  get(*specific_internal_energy) = 1.0 / get(mass_density_cons);
  for (size_t i = 0; i < Dim; ++i) {
    velocity->get(i) = momentum_density.get(i) * get(*specific_internal_energy);
  }

  // the magnetic energy is not internal, so remove it before dividing
  get(*specific_internal_energy) *=
      (get(energy_density) -
       0.5 * get(dot_product(magnetic_field_cons, magnetic_field_cons)));
  get(*specific_internal_energy) -=
      (0.5 * get(dot_product(*velocity, *velocity)));

  if constexpr (ThermodynamicDim == 1) {
    *pressure = equation_of_state.pressure_from_density(mass_density_cons);
  } else if constexpr (ThermodynamicDim == 2) {
    *pressure = equation_of_state.pressure_from_density_and_energy(
        mass_density_cons, *specific_internal_energy);
  }
}

}  // namespace NewtonianMhd

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define EVOLVE_B(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                       \
  template struct NewtonianMhd::PrimitiveFromConservative<DIM(data), \
                                                          EVOLVE_B(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (false))
GENERATE_INSTANTIATIONS(INSTANTIATION, (3), (true))

#undef INSTANTIATION

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(_, data)                                               \
  template void                                                              \
  NewtonianMhd::PrimitiveFromConservative<DIM(data), EVOLVE_B(data)>::apply< \
      THERMO_DIM(data)>(                                                     \
      const gsl::not_null<Scalar<DataVector>*> mass_density,                 \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*> velocity,         \
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,     \
      const gsl::not_null<Scalar<DataVector>*> pressure,                     \
      const Scalar<DataVector>& mass_density_cons,                           \
      const tnsr::I<DataVector, DIM(data)>& momentum_density,                \
      const Scalar<DataVector>& energy_density,                              \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>&      \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (false), (1, 2))
GENERATE_INSTANTIATIONS(INSTANTIATION, (3), (true), (1, 2))

#undef INSTANTIATION

#define INSTANTIATION(_, data)                                               \
  template void                                                              \
  NewtonianMhd::PrimitiveFromConservative<DIM(data), EVOLVE_B(data)>::apply< \
      THERMO_DIM(data)>(                                                     \
      const gsl::not_null<Scalar<DataVector>*> mass_density,                 \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*> velocity,         \
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,     \
      const gsl::not_null<Scalar<DataVector>*> pressure,                     \
      const gsl::not_null<tnsr::I<DataVector, 3>*> magnetic_field,           \
      const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,    \
      const Scalar<DataVector>& mass_density_cons,                           \
      const tnsr::I<DataVector, DIM(data)>& momentum_density,                \
      const Scalar<DataVector>& energy_density,                              \
      const tnsr::I<DataVector, 3>& magnetic_field_cons,                     \
      const Scalar<DataVector>& divergence_cleaning_field_cons,              \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>&      \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (3), (true), (1, 2))

#undef INSTANTIATION
#undef THERMO_DIM
#undef EVOLVE_B
#undef DIM
