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
struct BackgroundMagneticField;

struct GlmCleaningSpeed;
struct GlmConstraintDampingFactor;

template <size_t Dim>
struct CharacteristicSpeeds;

template <size_t Dim>
struct SourceTerm;
}  // namespace Tags
}  // namespace NewtonianMhd
/// \endcond
