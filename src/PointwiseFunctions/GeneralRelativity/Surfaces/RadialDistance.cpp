// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/RadialDistance.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace StrahlkorperGr {
template <typename Frame>
void radial_distance(const gsl::not_null<Scalar<DataVector>*> radial_distance,
                     const Strahlkorper<Frame>& strahlkorper_a,
                     const Strahlkorper<Frame>& strahlkorper_b) {
  if (strahlkorper_a.expansion_center() != strahlkorper_b.expansion_center()) {
    ERROR(
        "Currently computing the radial distance between two Strahlkorpers "
        "is only supported if they have the same centers, but the "
        "strahlkorpers provided have centers "
        << strahlkorper_a.expansion_center() << " and "
        << strahlkorper_b.expansion_center());
  }
  get(*radial_distance)
      .destructive_resize(strahlkorper_a.ylm_spherepack().physical_size());
  if (strahlkorper_a.l_max() == strahlkorper_b.l_max() and
      strahlkorper_a.m_max() == strahlkorper_b.m_max()) {
    get(*radial_distance) = get(StrahlkorperFunctions::radius(strahlkorper_a)) -
                            get(StrahlkorperFunctions::radius(strahlkorper_b));
  } else if (strahlkorper_a.l_max() > strahlkorper_b.l_max() or
             (strahlkorper_a.l_max() == strahlkorper_b.l_max() and
              strahlkorper_a.m_max() > strahlkorper_b.m_max())) {
    get(*radial_distance) =
        get(StrahlkorperFunctions::radius(strahlkorper_a)) -
        get(StrahlkorperFunctions::radius(Strahlkorper<Frame>(
            strahlkorper_a.l_max(), strahlkorper_a.m_max(), strahlkorper_b)));
  } else {
    get(*radial_distance) =
        -get(StrahlkorperFunctions::radius(strahlkorper_b)) +
        get(StrahlkorperFunctions::radius(Strahlkorper<Frame>(
            strahlkorper_b.l_max(), strahlkorper_b.m_max(), strahlkorper_a)));
  };
}
}  // namespace StrahlkorperGr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                    \
  template void StrahlkorperGr::radial_distance<FRAME(data)>(   \
      const gsl::not_null<Scalar<DataVector>*> radial_distance, \
      const Strahlkorper<FRAME(data)>& strahlkorper_a,          \
      const Strahlkorper<FRAME(data)>& strahlkorper_b);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME