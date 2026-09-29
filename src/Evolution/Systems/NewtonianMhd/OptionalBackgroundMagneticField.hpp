// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace NewtonianMhd {
/*!
 * \brief Stands in for \f$B_0\f$ where the background-field splitting is
 * disabled at compile time.
 *
 * Every \f$B_0\f$ code path is guarded by `if constexpr
 * (UseBackgroundMagneticField)`, so with the splitting disabled none of that
 * arithmetic is compiled and no \f$B_0\f$ is stored, projected to faces or
 * copied. Passing this empty type in place of the tensor keeps a single
 * signature for the shared implementations rather than duplicating them.
 */
struct NoBackgroundMagneticField {};

/// The type used to pass \f$B_0\f$ into the shared implementations: the field
/// itself when the splitting is enabled, and `NoBackgroundMagneticField`
/// otherwise.
template <size_t Dim, bool UseBackgroundMagneticField = false>
using BackgroundMagneticFieldArgument =
    tmpl::conditional_t<UseBackgroundMagneticField,
                        const tnsr::I<DataVector, Dim, Frame::Inertial>&,
                        NoBackgroundMagneticField>;

/// The type used to return the ghost \f$B_0\f$ from a boundary condition:
/// a pointer to the field when the splitting is enabled, nothing otherwise.
template <size_t Dim, bool UseBackgroundMagneticField = false>
using BackgroundMagneticFieldOutput = tmpl::conditional_t<
    UseBackgroundMagneticField,
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>,
    NoBackgroundMagneticField>;

/// The tag list holding \f$B_0\f$, empty when the splitting is disabled.
template <typename Tag, bool UseBackgroundMagneticField>
using background_magnetic_field_tag_list =
    tmpl::conditional_t<UseBackgroundMagneticField, tmpl::list<Tag>,
                        tmpl::list<>>;
}  // namespace NewtonianMhd
