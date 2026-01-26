// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/WeylElectric.hpp"

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace hydro {
template <typename DataType>
tnsr::ii<DataType, 3> weyl_electric(
    const tnsr::ii<DataType, 3>& vacuum_weyl_electric,
    const tnsr::AA<DataType, 3>& stress_energy,
    const tnsr::aa<DataType, 3>& ricci_tensor,
    const Scalar<DataType>& ricci_scalar,
    const tnsr::AA<DataType, 3>& inverse_spacetime_metric,
    const tnsr::aa<DataType, 3>& induced_spatial_metric) {
  tnsr::ii<DataType, 3> result{get_size(get<0, 0>(stress_energy))};
  weyl_electric(make_not_null(&result), vacuum_weyl_electric, stress_energy,
                ricci_tensor, ricci_scalar, inverse_spacetime_metric,
                induced_spatial_metric);
  return result;
}

template <typename DataType>
void weyl_electric(const gsl::not_null<tnsr::ii<DataType, 3>*> weyl_electric,
                   const tnsr::ii<DataType, 3>& vacuum_weyl_electric,
                   const tnsr::AA<DataType, 3>& stress_energy,
                   const tnsr::aa<DataType, 3>& ricci_tensor,
                   const Scalar<DataType>& ricci_scalar,
                   const tnsr::AA<DataType, 3>& inverse_spacetime_metric,
                   const tnsr::aa<DataType, 3>& induced_spatial_metric) {
  set_number_of_grid_points(weyl_electric, stress_energy);
  auto spacetime_weyl_electric_from_matter =
      make_with_value<tnsr::aa<DataType, 3>>(get<0, 0>(stress_energy), 0.0);

  auto induced_spatial_metric_aB =
      make_with_value<tnsr::aB<DataType, 3>>(get<0, 0>(stress_energy), 0.0);
  ::tenex::evaluate<ti::a, ti::B>(make_not_null(&induced_spatial_metric_aB),
                                  inverse_spacetime_metric(ti::B, ti::C) *
                                      induced_spatial_metric(ti::a, ti::c));

  ::tenex::evaluate<ti::a, ti::b>(
      make_not_null(&spacetime_weyl_electric_from_matter),
      -0.5 *
              (induced_spatial_metric_aB(ti::a, ti::C) *
                   induced_spatial_metric_aB(ti::b, ti::D) +
               induced_spatial_metric(ti::a, ti::b) *
                   inverse_spacetime_metric(ti::D, ti::E) *
                   induced_spatial_metric_aB(ti::e, ti::C)) *
              ricci_tensor(ti::c, ti::d) +
          induced_spatial_metric(ti::a, ti::b) * ricci_scalar() / 3.);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      weyl_electric->get(i, j) =
          vacuum_weyl_electric.get(i, j) +
          spacetime_weyl_electric_from_matter.get(i + 1, j + 1);
    }
  }
}
}  // namespace hydro

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template tnsr::ii<DTYPE(data), 3> hydro::weyl_electric(           \
      const tnsr::ii<DTYPE(data), 3>& vacuum_weyl_electric,         \
      const tnsr::AA<DTYPE(data), 3>& stress_energy,                \
      const tnsr::aa<DTYPE(data), 3>& ricci_tensor,                 \
      const Scalar<DTYPE(data)>& ricci_scalar,                      \
      const tnsr::AA<DTYPE(data), 3>& inverse_spacetime_metric,     \
      const tnsr::aa<DTYPE(data), 3>& induced_spatial_metric);      \
  template void hydro::weyl_electric(                               \
      const gsl::not_null<tnsr::ii<DTYPE(data), 3>*> weyl_electric, \
      const tnsr::ii<DTYPE(data), 3>& vacuum_weyl_electric,         \
      const tnsr::AA<DTYPE(data), 3>& stress_energy,                \
      const tnsr::aa<DTYPE(data), 3>& ricci_tensor,                 \
      const Scalar<DTYPE(data)>& ricci_scalar,                      \
      const tnsr::AA<DTYPE(data), 3>& inverse_spacetime_metric,     \
      const tnsr::aa<DTYPE(data), 3>& induced_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
