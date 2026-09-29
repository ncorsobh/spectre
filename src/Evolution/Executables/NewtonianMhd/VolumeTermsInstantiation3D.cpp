// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Executables/NewtonianMhd/VolumeTermsInstantiation.tpp"

// With the background-field splitting enabled B0 is the last argument tag of
// `TimeDerivativeTerms`, so it simply extends the argument list.
#define BACKGROUND_MAGNETIC_FIELD_ARG_3D \
  , const tnsr::I<DataVector, 3>& background_magnetic_field_volume

namespace evolution::dg::Actions::detail {
VOLUME_TERMS_INSTANTIATION(3, false, )
VOLUME_TERMS_INSTANTIATION(3, true, BACKGROUND_MAGNETIC_FIELD_ARG_3D)
}  // namespace evolution::dg::Actions::detail

#undef BACKGROUND_MAGNETIC_FIELD_ARG_3D
