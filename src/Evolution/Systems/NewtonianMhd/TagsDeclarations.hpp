// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
namespace NewtonianMhd {
namespace Tags {

struct MassDensityCons;
template <size_t Dim, typename Fr = Frame::Inertial>
struct MomentumDensity;
struct EnergyDensity;

template <size_t Dim, typename Fr = Frame::Inertial>
struct MagneticFieldCons;
struct DivergenceCleaningFieldCons;

template <size_t Dim, typename Fr = Frame::Inertial>
struct BackgroundMagneticFieldVolume;
template <size_t Dim, typename Fr = Frame::Inertial>
struct BackgroundMagneticField;

struct DivergenceCleaningSpeed;
struct ConstraintDampingParameter;

template <size_t Dim>
struct CharacteristicSpeeds;

template <size_t Dim, bool UseBackgroundMagneticField>
struct SourceTerm;
}  // namespace Tags
}  // namespace NewtonianMhd
/// \endcond
