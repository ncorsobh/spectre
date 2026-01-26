// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Declares function templates to calculate the Ricci tensor

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace hydro {

/// @{
/*!
 * \brief Computes Ricci tensor in hydro by trace-reversing the stress-energy
 * tensor.
 *
 * \details From the Einstein Equations, trace-reversal of the stress-energy
 * tensor yields \f$R_{ab} = 8\pi(T_{ab} - \frac{1}{2}g_{ab}T)\f$ where
 * \f$T = g^{ab}T_{ab}\f$ is the stress-energy trace.
 */
template <typename DataType>
void ricci_tensor(gsl::not_null<tnsr::aa<DataType, 3>*> result,
                  const tnsr::AA<DataType, 3>& stress_energy,
                  const tnsr::aa<DataType, 3>& spacetime_metric);

template <typename DataType>
tnsr::aa<DataType, 3> ricci_tensor(
    const tnsr::AA<DataType, 3>& stress_energy,
    const tnsr::aa<DataType, 3>& spacetime_metric);
/// @}

namespace Tags {
/// Compute item for the spacetime (4D) Ricci tensor \f$R_{ab}\f$ in hydro,
/// computed by trace-reversing the stress-energy tensor.
///
/// Can be retrieved using `hydro::Tags::Ricci`
template <typename DataType>
struct RicciCompute : Ricci<DataType, 3>, db::ComputeTag {
  using argument_tags = tmpl::list<hydro::Tags::StressEnergy<DataType, 3>,
                                   gr::Tags::SpacetimeMetric<DataType, 3>>;

  using return_type = tnsr::aa<DataType, 3>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::aa<DataType, 3>*>, const tnsr::AA<DataType, 3>&,
      const tnsr::aa<DataType, 3>&)>(&ricci_tensor);

  using base = Ricci<DataType, 3>;
};

/// Computes the spacetime (4D) Ricci scalar using the spacetime Ricci tensor
/// and the inverse spacetime metric.
///
/// Can be retrieved using `hydro::Tags::RicciScalar`
template <typename DataType>
struct RicciScalarCompute : RicciScalar<DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<hydro::Tags::Ricci<DataType, 3>,
                 gr::Tags::InverseSpacetimeMetric<DataType, 3>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const tnsr::aa<DataType, 3>&,
      const tnsr::AA<DataType, 3>&)>(&gr::ricci_scalar);

  using base = RicciScalar<DataType>;
};
}  // namespace Tags
}  // namespace hydro
