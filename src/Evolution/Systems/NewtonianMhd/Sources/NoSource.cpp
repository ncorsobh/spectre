// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Sources/NoSource.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd::Sources {
template <size_t Dim, bool UseBackgroundMagneticField>
NoSource<Dim, UseBackgroundMagneticField>::NoSource(CkMigrateMessage* msg)
    : Source<Dim, UseBackgroundMagneticField>{msg} {}

template <size_t Dim, bool UseBackgroundMagneticField>
void NoSource<Dim, UseBackgroundMagneticField>::pup(PUP::er& p) {
  Source<Dim, UseBackgroundMagneticField>::pup(p);
}

template <size_t Dim, bool UseBackgroundMagneticField>
auto NoSource<Dim, UseBackgroundMagneticField>::get_clone() const
    -> std::unique_ptr<Source<Dim, UseBackgroundMagneticField>> {
  return std::make_unique<NoSource<Dim, UseBackgroundMagneticField>>(*this);
}

template <size_t Dim, bool UseBackgroundMagneticField>
void NoSource<Dim, UseBackgroundMagneticField>::operator()(
    const gsl::not_null<Scalar<DataVector>*> /*source_mass_density_cons*/,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> /*source_momentum_density*/,
    const gsl::not_null<Scalar<DataVector>*> /*source_energy_density*/,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> /*source_magnetic_field*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*source_divergence_cleaning_field*/,
    const Scalar<DataVector>& /*mass_density_cons*/,
    const tnsr::I<DataVector, Dim>& /*momentum_density*/,
    const Scalar<DataVector>& /*energy_density*/,
    const tnsr::I<DataVector, Dim>& /*magnetic_field*/,
    const Scalar<DataVector>& /*divergence_cleaning_field*/,
    const tnsr::I<DataVector, Dim>& /*velocity*/,
    const Scalar<DataVector>& /*pressure*/,
    BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
    /*background_magnetic_field*/,
    const EquationsOfState::EquationOfState<false, 2>& /*eos*/,
    const tnsr::I<DataVector, Dim>& /*coords*/, const double /*time*/) const {}

template <size_t Dim, bool UseBackgroundMagneticField>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID NoSource<Dim, UseBackgroundMagneticField>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define USE_BG(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data) template class NoSource<DIM(data), USE_BG(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (true, false))

#undef INSTANTIATION
#undef USE_BG
#undef DIM
}  // namespace NewtonianMhd::Sources
