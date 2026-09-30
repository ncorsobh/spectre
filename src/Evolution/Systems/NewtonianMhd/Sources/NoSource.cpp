// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Sources/NoSource.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd::Sources {
template <bool UseBackgroundMagneticField>
NoSource<UseBackgroundMagneticField>::NoSource(CkMigrateMessage* msg)
    : Source<UseBackgroundMagneticField>{msg} {}

template <bool UseBackgroundMagneticField>
void NoSource<UseBackgroundMagneticField>::pup(PUP::er& p) {
  Source<UseBackgroundMagneticField>::pup(p);
}

template <bool UseBackgroundMagneticField>
auto NoSource<UseBackgroundMagneticField>::get_clone() const
    -> std::unique_ptr<Source<UseBackgroundMagneticField>> {
  return std::make_unique<NoSource<UseBackgroundMagneticField>>(*this);
}

template <bool UseBackgroundMagneticField>
void NoSource<UseBackgroundMagneticField>::operator()(
    const gsl::not_null<Scalar<DataVector>*> /*source_mass_density_cons*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*> /*source_momentum_density*/,
    const gsl::not_null<Scalar<DataVector>*> /*source_energy_density*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*> /*source_magnetic_field*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*source_divergence_cleaning_field*/,
    const Scalar<DataVector>& /*mass_density_cons*/,
    const tnsr::I<DataVector, 3>& /*momentum_density*/,
    const Scalar<DataVector>& /*energy_density*/,
    const tnsr::I<DataVector, 3>& /*magnetic_field*/,
    const Scalar<DataVector>& /*divergence_cleaning_field*/,
    const tnsr::I<DataVector, 3>& /*velocity*/,
    const Scalar<DataVector>& /*pressure*/,
    BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
    /*background_magnetic_field*/,
    const EquationsOfState::EquationOfState<false, 2>& /*eos*/,
    const tnsr::I<DataVector, 3>& /*coords*/, const double /*time*/) const {}

template <bool UseBackgroundMagneticField>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID NoSource<UseBackgroundMagneticField>::my_PUP_ID = 0;

#define USE_BG(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data) template class NoSource<USE_BG(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (true, false))

#undef INSTANTIATION
#undef USE_BG
}  // namespace NewtonianMhd::Sources
