// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace NewtonianMhd {

/*!
 * \brief The fast magnetosonic speed
 *
 * \f{align*}
 *   c_f = \sqrt{c_s^2 + \frac{|B_{\rm tot}|^2}{\rho}}
 * \f}
 *
 * where \f$B_{\rm tot} = B_0 + B_1\f$ and \f$c_s\f$ is the (Newtonian) sound
 * speed.  This is the *approximate* Dedner-type upper bound on the fast wave
 * speed which is commonly used in HLL-type solvers for Newtonian MHD.
 */
template <size_t Dim>
void fast_magnetosonic_speed(
    gsl::not_null<Scalar<DataVector>*> fast_speed,
    const Scalar<DataVector>& mass_density,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::I<DataVector, Dim>& magnetic_field,
    const tnsr::I<DataVector, Dim>& background_magnetic_field);

/*!
 * \brief Characteristic speeds of the Newtonian MHD system.
 *
 * The nine physical speeds in 3D are, ordered by increasing wave speed,
 * \f$-c_h, v_n - c_f, v_n - c_A, v_n - c_{s,\rm slow}, v_n\f$ (entropy),
 * \f$v_n + c_{s,\rm slow}, v_n + c_A, v_n + c_f, +c_h\f$.  For CFL and HLL
 * wave-speed estimates only the outermost speeds are used, so this routine
 * populates a \f$2\,{\rm Dim} + 3\f$-element array whose extreme entries are
 * \f$\pm c_h\f$ and \f$v_n \pm c_f\f$; interior entries are filled with
 * \f$v_n\f$ or \f$v_n \pm c_A\f$ (Alfvén speed) as placeholders.
 */
template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 2 * Dim + 3>*> char_speeds,
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::I<DataVector, Dim>& magnetic_field,
    const tnsr::I<DataVector, Dim>& background_magnetic_field,
    const tnsr::i<DataVector, Dim>& normal, double glm_cleaning_speed);

template <size_t Dim>
std::array<DataVector, 2 * Dim + 3> characteristic_speeds(
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::I<DataVector, Dim>& magnetic_field,
    const tnsr::I<DataVector, Dim>& background_magnetic_field,
    const tnsr::i<DataVector, Dim>& normal, double glm_cleaning_speed);

namespace Tags {

/// The (scalar) fast magnetosonic speed for MHD.
struct FastMagnetosonicSpeed : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The sound speed squared.
struct SoundSpeedSquared : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The scalar largest characteristic speed used for CFL control.
struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};

/// Compute the largest characteristic speed used for CFL control.
///
/// \f$c_{\max} = \max(|v| + c_f, c_h)\f$
template <size_t Dim>
struct ComputeLargestCharacteristicSpeed : LargestCharacteristicSpeed,
                                           db::ComputeTag {
  using argument_tags =
      tmpl::list<hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 FastMagnetosonicSpeed, NewtonianMhd::Tags::GlmCleaningSpeed>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(gsl::not_null<double*> speed,
                       const tnsr::I<DataVector, Dim>& velocity,
                       const Scalar<DataVector>& fast_speed,
                       double glm_cleaning_speed) {
    *speed = std::max(max(get(magnitude(velocity)) + get(fast_speed)),
                      glm_cleaning_speed);
  }
};
}  // namespace Tags
}  // namespace NewtonianMhd
