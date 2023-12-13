// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the product of the negative determinant of the spacetime
 * metric and the Landau-Lifshitz pseudotensor,
 * \f$ (-g) t_\mathrm{L-L}^{ab} \f$, as defined in \cite LandauLifshitz1962.
 *
 * \details Computes the product of the negative determinant of the spacetime
 * metric and the Landau-Lifshitz pseudotensor,
 * \f$ (-g) t_\mathrm{L-L}^{ab} \f$, as defined in \cite LandauLifshitz1962.
 * Using \f$ \mathfrak{g}^{ab}\equiv (-g)^{1/2} g^{ab} \f$, the Landau-Lifshitz
 * pseudotensor is defined as the following,
 * \f{align}{
 *   (-g) t_\mathrm{L-L}^{ab} =& \frac{1}{16\pi}
 *   \left{\partial_c \mathfrak{g}^{ab} \partial_d \mathfrak{g}^{cd}
 *         - \partial_c \mathfrak{g}^{ac} \partial_d \mathfrak{g}^{bd}
 *         +  \frac{1}{2} g^{ab} g_{cd}
 *              \partial_e \mathfrak{g}^{cf} \partial_f \mathfrak{g}^{ed} \\
 *       & - (g^{ac} g_{df}
 *              \partial_e \mathfrak{g}^{bf} \partial_c \mathfrak{g}^{de}
 *           + g^{bc} g_{df}
 *              \partial_e \mathfrak{g}^{af} \partial_c \mathfrak{g}^{de})
 *         + g_{cd} g^{fe}
 *              \partial_f \mathfrak{g}^{ac} \partial_e \mathfrak{g}^{bd} \\
 *       & + \frac{1}{8} (2 g^{ac} g^{bd} - g^{ab} g^{cd})
 *                       (2 g_{fe} g_{gh} - g_{eg} g_{fh})
 *             \partial_c \mathfrak{g}^{fh} \partial_d \mathfrak{g}^{eg}\right}.
 * \f}
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void landau_lifshitz_pseudotensor(
    gsl::not_null<tnsr::AA<DataType, SpatialDim, Frame>*> ll_pseudotensor,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::a<DataType, SpatialDim, Frame>& da_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
void landau_lifshitz_pseudotensor(
    gsl::not_null<tnsr::AA<DataType, SpatialDim, Frame>*> ll_pseudotensor,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::AA<DataType, SpatialDim, Frame> landau_lifshitz_pseudotensor(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::a<DataType, SpatialDim, Frame>& da_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::AA<DataType, SpatialDim, Frame> landau_lifshitz_pseudotensor(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric);
/// @}
}  // namespace gr
