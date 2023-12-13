// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/LandauLifshitzPseudotensor.hpp"

#include <iostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/CombineSpacetimeView.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeDerivativeOfGothG.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
template <typename DataType, size_t SpatialDim, typename Frame>
void landau_lifshitz_pseudotensor(
    gsl::not_null<tnsr::AA<DataType, SpatialDim, Frame>*> ll_pseudotensor,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::a<DataType, SpatialDim, Frame>& da_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric) {
  set_number_of_grid_points(ll_pseudotensor, inverse_spacetime_metric);

  tnsr::aBB<DataType, SpatialDim, Frame> da_goth_g{};
  spacetime_deriv_of_goth_g(make_not_null(&da_goth_g), inverse_spacetime_metric,
                            da_spacetime_metric, lapse, da_lapse,
                            sqrt_det_spatial_metric, da_det_spatial_metric);

  tnsr::A<DataType, SpatialDim, Frame> div_goth_g;
  tenex::evaluate<ti::A>(make_not_null(&div_goth_g),
                         da_goth_g(ti::b, ti::A, ti::B));

  tnsr::a<DataType, SpatialDim, Frame> da_trace_goth_g;
  tenex::evaluate<ti::a>(
      make_not_null(&da_trace_goth_g),
      spacetime_metric(ti::b, ti::c) * da_goth_g(ti::a, ti::B, ti::C));

  tnsr::A<DataType, SpatialDim, Frame> dA_trace_goth_g;
  tenex::evaluate<ti::A>(
      make_not_null(&dA_trace_goth_g),
      inverse_spacetime_metric(ti::A, ti::B) * da_trace_goth_g(ti::b));

  tnsr::abC<DataType, SpatialDim, Frame> da_goth_g_bC;
  tenex::evaluate<ti::a, ti::b, ti::C>(
      make_not_null(&da_goth_g_bC),
      spacetime_metric(ti::b, ti::d) * da_goth_g(ti::a, ti::D, ti::C));

  tnsr::AbC<DataType, SpatialDim, Frame> dA_goth_g_bC;
  tenex::evaluate<ti::A, ti::b, ti::C>(make_not_null(&dA_goth_g_bC),
                                       inverse_spacetime_metric(ti::A, ti::D) *
                                           da_goth_g_bC(ti::d, ti::b, ti::C));

  tnsr::ABC<DataType, SpatialDim, Frame> dA_goth_g;
  tenex::evaluate<ti::A, ti::B, ti::C>(
      make_not_null(&dA_goth_g),
      inverse_spacetime_metric(ti::A, ti::D) * da_goth_g(ti::d, ti::B, ti::C));

  tnsr::AB<DataType, SpatialDim, Frame> da_goth_g_product;
  tenex::evaluate<ti::A, ti::B>(
      make_not_null(&da_goth_g_product),
      da_goth_g_bC(ti::c, ti::d, ti::A) * dA_goth_g(ti::B, ti::D, ti::C));

  tenex::evaluate<ti::A, ti::B>(
      ll_pseudotensor,
      (1. / (16. * M_PI)) *
          (da_goth_g(ti::c, ti::A, ti::B) * div_goth_g(ti::C) -
           div_goth_g(ti::A) * div_goth_g(ti::B) +
           0.5 * inverse_spacetime_metric(ti::A, ti::B) *
               da_goth_g_bC(ti::c, ti::d, ti::E) *
               da_goth_g(ti::e, ti::C, ti::D) -
           (da_goth_g_product(ti::A, ti::B) + da_goth_g_product(ti::B, ti::A)) +
           da_goth_g_bC(ti::c, ti::d, ti::A) * dA_goth_g(ti::C, ti::B, ti::D) +
           0.5 * dA_goth_g_bC(ti::A, ti::c, ti::D) *
               dA_goth_g_bC(ti::B, ti::d, ti::C) -
           0.25 * inverse_spacetime_metric(ti::A, ti::B) *
               da_goth_g_bC(ti::c, ti::d, ti::E) *
               dA_goth_g_bC(ti::C, ti::e, ti::D) -
           0.25 * dA_trace_goth_g(ti::A) * dA_trace_goth_g(ti::B) +
           0.125 * inverse_spacetime_metric(ti::A, ti::B) *
               da_trace_goth_g(ti::c) * dA_trace_goth_g(ti::C)));
}

template <typename DataType, size_t SpatialDim, typename Frame>
void landau_lifshitz_pseudotensor(
    gsl::not_null<tnsr::AA<DataType, SpatialDim, Frame>*> ll_pseudotensor,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric) {
  tnsr::a<DataType, SpatialDim, Frame> da_lapse{};
  combine_spacetime_view<SpatialDim, UpLo::Lo, Frame>(make_not_null(&da_lapse),
                                                      dt_lapse, deriv_lapse);

  landau_lifshitz_pseudotensor(ll_pseudotensor, spacetime_metric,
                               inverse_spacetime_metric, da_spacetime_metric,
                               lapse, da_lapse, sqrt_det_spatial_metric,
                               da_det_spatial_metric);
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::AA<DataType, SpatialDim, Frame> landau_lifshitz_pseudotensor(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::a<DataType, SpatialDim, Frame>& da_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric) {
  tnsr::AA<DataType, SpatialDim, Frame> ll_pseudotensor{};
  landau_lifshitz_pseudotensor(make_not_null(&ll_pseudotensor),
                               spacetime_metric, inverse_spacetime_metric,
                               da_spacetime_metric, lapse, da_lapse,
                               sqrt_det_spatial_metric, da_det_spatial_metric);
  return ll_pseudotensor;
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::AA<DataType, SpatialDim, Frame> landau_lifshitz_pseudotensor(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric) {
  tnsr::AA<DataType, SpatialDim, Frame> ll_pseudotensor{};
  landau_lifshitz_pseudotensor(
      make_not_null(&ll_pseudotensor), spacetime_metric,
      inverse_spacetime_metric, da_spacetime_metric, lapse, dt_lapse,
      deriv_lapse, sqrt_det_spatial_metric, da_det_spatial_metric);
  return ll_pseudotensor;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template void gr::landau_lifshitz_pseudotensor(                            \
      const gsl::not_null<tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>*>    \
          ll_pseudotensor,                                                   \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& spacetime_metric, \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data)>&                  \
          da_spacetime_metric,                                               \
      const Scalar<DTYPE(data)>& lapse,                                      \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& da_lapse,          \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                    \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                    \
          da_det_spatial_metric);                                            \
  template void gr::landau_lifshitz_pseudotensor(                            \
      const gsl::not_null<tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>*>    \
          ll_pseudotensor,                                                   \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& spacetime_metric, \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data)>&                  \
          da_spacetime_metric,                                               \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse, \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,       \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                    \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                    \
          da_det_spatial_metric);                                            \
  template tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>                     \
  gr::landau_lifshitz_pseudotensor(                                          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& spacetime_metric, \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data)>&                  \
          da_spacetime_metric,                                               \
      const Scalar<DTYPE(data)>& lapse,                                      \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& da_lapse,          \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                    \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                    \
          da_det_spatial_metric);                                            \
  template tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>                     \
  gr::landau_lifshitz_pseudotensor(                                          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& spacetime_metric, \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data)>&                  \
          da_spacetime_metric,                                               \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse, \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,       \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                    \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                    \
          da_det_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
