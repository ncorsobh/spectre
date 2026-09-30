// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
namespace NewtonianMhd {
namespace Tags {

struct MassDensityCons;
template <typename Fr = Frame::Inertial>
struct MomentumDensity;
struct EnergyDensity;

template <typename Fr = Frame::Inertial>
struct MagneticFieldCons;
struct DivergenceCleaningFieldCons;

template <typename Fr = Frame::Inertial>
struct BackgroundMagneticFieldVolume;
template <typename Fr = Frame::Inertial>
struct BackgroundMagneticField;

struct DivergenceCleaningSpeed;
struct ConstraintDampingParameter;

struct CharacteristicSpeeds;

template <bool UseBackgroundMagneticField>
struct SourceTerm;
}  // namespace Tags
}  // namespace NewtonianMhd
/// \endcond
