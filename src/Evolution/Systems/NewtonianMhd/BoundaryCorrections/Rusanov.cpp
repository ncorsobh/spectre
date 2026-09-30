// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Rusanov.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianMhd/Characteristics.hpp"
#include "Evolution/Systems/NewtonianMhd/SoundSpeedSquared.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd::BoundaryCorrections {
template <bool UseBackgroundMagneticField>
Rusanov<UseBackgroundMagneticField>::Rusanov(CkMigrateMessage* msg)
    : BoundaryCorrection(msg) {}

template <bool UseBackgroundMagneticField>
std::unique_ptr<evolution::BoundaryCorrection>
Rusanov<UseBackgroundMagneticField>::get_clone() const {
  return std::make_unique<Rusanov>(*this);
}

template <bool UseBackgroundMagneticField>
void Rusanov<UseBackgroundMagneticField>::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
}

template <bool UseBackgroundMagneticField>
double Rusanov<UseBackgroundMagneticField>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_mass_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> packaged_energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> packaged_divergence_cleaning_field,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_flux_mass_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_momentum_density,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_flux_energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_magnetic_field,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_flux_divergence_cleaning_field,
    const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,

    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field,

    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_mass_density,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_momentum_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_energy_density,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_magnetic_field,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        flux_divergence_cleaning_field,

    const tnsr::I<DataVector, 3, Frame::Inertial>& velocity,
    const Scalar<DataVector>& specific_internal_energy,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<false, 2>& equation_of_state,
    const double divergence_cleaning_speed) const {
  // Selected by an empty `dg_package_data_temporary_tags`, so that B0 is
  // never projected onto element faces when the splitting is disabled.
  if constexpr (UseBackgroundMagneticField) {
    ERROR(
        "Called the boundary correction overload that takes no background "
        "magnetic field, but the background-field splitting is enabled.");
  } else {
    return dg_package_data(
        packaged_mass_density, packaged_momentum_density,
        packaged_energy_density, packaged_magnetic_field,
        packaged_divergence_cleaning_field,
        packaged_normal_dot_flux_mass_density,
        packaged_normal_dot_flux_momentum_density,
        packaged_normal_dot_flux_energy_density,
        packaged_normal_dot_flux_magnetic_field,
        packaged_normal_dot_flux_divergence_cleaning_field,
        packaged_abs_char_speed, mass_density, momentum_density, energy_density,
        magnetic_field, divergence_cleaning_field, flux_mass_density,
        flux_momentum_density, flux_energy_density, flux_magnetic_field,
        flux_divergence_cleaning_field, {}, velocity, specific_internal_energy,
        normal_covector, mesh_velocity, normal_dot_mesh_velocity,
        equation_of_state, divergence_cleaning_speed);
  }
}

template <bool UseBackgroundMagneticField>
double Rusanov<UseBackgroundMagneticField>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_mass_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> packaged_energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> packaged_divergence_cleaning_field,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_flux_mass_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_momentum_density,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_flux_energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_magnetic_field,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_flux_divergence_cleaning_field,
    const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,

    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& divergence_cleaning_field,

    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_mass_density,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_momentum_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_energy_density,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_magnetic_field,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        flux_divergence_cleaning_field,

    const BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
        background_magnetic_field,

    const tnsr::I<DataVector, 3, Frame::Inertial>& velocity,
    const Scalar<DataVector>& specific_internal_energy,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<false, 2>& equation_of_state,
    const double divergence_cleaning_speed) const {
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>>>
      buffer{get(mass_density).size()};
  auto& sound_speed_sq = get<::Tags::TempScalar<0>>(buffer);
  auto& fast_speed = get<::Tags::TempScalar<1>>(buffer);
  auto& normal_dot_velocity = get<::Tags::TempScalar<2>>(buffer);

  sound_speed_squared(make_not_null(&sound_speed_sq), mass_density,
                      specific_internal_energy, equation_of_state);
  fast_magnetosonic_speed<UseBackgroundMagneticField>(
      make_not_null(&fast_speed), mass_density, sound_speed_sq, magnetic_field,
      background_magnetic_field);
  dot_product(make_not_null(&normal_dot_velocity), velocity, normal_covector);
  if (normal_dot_mesh_velocity.has_value()) {
    get(normal_dot_velocity) -= get(*normal_dot_mesh_velocity);
  }

  // The GLM waves travel at +/- c_h regardless of the fluid state, so they can
  // dominate the fluid speeds.
  get(*packaged_abs_char_speed) =
      max(abs(get(normal_dot_velocity)) + get(fast_speed),
          divergence_cleaning_speed);

  *packaged_mass_density = mass_density;
  *packaged_momentum_density = momentum_density;
  *packaged_energy_density = energy_density;
  *packaged_magnetic_field = magnetic_field;
  *packaged_divergence_cleaning_field = divergence_cleaning_field;

  normal_dot_flux(packaged_normal_dot_flux_mass_density, normal_covector,
                  flux_mass_density);
  normal_dot_flux(packaged_normal_dot_flux_momentum_density, normal_covector,
                  flux_momentum_density);
  normal_dot_flux(packaged_normal_dot_flux_energy_density, normal_covector,
                  flux_energy_density);
  normal_dot_flux(packaged_normal_dot_flux_magnetic_field, normal_covector,
                  flux_magnetic_field);
  normal_dot_flux(packaged_normal_dot_flux_divergence_cleaning_field,
                  normal_covector, flux_divergence_cleaning_field);

  return max(get(*packaged_abs_char_speed));
}

template <bool UseBackgroundMagneticField>
void Rusanov<UseBackgroundMagneticField>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_mass_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_magnetic_field,
    const gsl::not_null<Scalar<DataVector>*>
        boundary_correction_divergence_cleaning_field,
    const Scalar<DataVector>& mass_density_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density_int,
    const Scalar<DataVector>& energy_density_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field_int,
    const Scalar<DataVector>& divergence_cleaning_field_int,
    const Scalar<DataVector>& normal_dot_flux_mass_density_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        normal_dot_flux_momentum_density_int,
    const Scalar<DataVector>& normal_dot_flux_energy_density_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        normal_dot_flux_magnetic_field_int,
    const Scalar<DataVector>& normal_dot_flux_divergence_cleaning_field_int,
    const Scalar<DataVector>& abs_char_speed_int,
    const Scalar<DataVector>& mass_density_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density_ext,
    const Scalar<DataVector>& energy_density_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field_ext,
    const Scalar<DataVector>& divergence_cleaning_field_ext,
    const Scalar<DataVector>& normal_dot_flux_mass_density_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        normal_dot_flux_momentum_density_ext,
    const Scalar<DataVector>& normal_dot_flux_energy_density_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        normal_dot_flux_magnetic_field_ext,
    const Scalar<DataVector>& normal_dot_flux_divergence_cleaning_field_ext,
    const Scalar<DataVector>& abs_char_speed_ext,
    const dg::Formulation dg_formulation) const {
  const DataVector max_abs_char_speed =
      max(get(abs_char_speed_int), get(abs_char_speed_ext));

  const auto scalar_correction =
      [&max_abs_char_speed, &dg_formulation](
          const gsl::not_null<Scalar<DataVector>*> correction,
          const Scalar<DataVector>& normal_dot_flux_int,
          const Scalar<DataVector>& normal_dot_flux_ext,
          const Scalar<DataVector>& var_int,
          const Scalar<DataVector>& var_ext) {
        if (dg_formulation == dg::Formulation::WeakInertial) {
          get(*correction) =
              0.5 * (get(normal_dot_flux_int) - get(normal_dot_flux_ext));
        } else {
          get(*correction) =
              -0.5 * (get(normal_dot_flux_int) + get(normal_dot_flux_ext));
        }
        get(*correction) -=
            0.5 * max_abs_char_speed * (get(var_ext) - get(var_int));
      };
  const auto vector_correction =
      [&max_abs_char_speed, &dg_formulation](
          const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
              correction,
          const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_int,
          const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_ext,
          const tnsr::I<DataVector, 3, Frame::Inertial>& var_int,
          const tnsr::I<DataVector, 3, Frame::Inertial>& var_ext) {
        for (size_t i = 0; i < 3; ++i) {
          if (dg_formulation == dg::Formulation::WeakInertial) {
            correction->get(i) =
                0.5 * (normal_dot_flux_int.get(i) - normal_dot_flux_ext.get(i));
          } else {
            correction->get(i) = -0.5 * (normal_dot_flux_int.get(i) +
                                         normal_dot_flux_ext.get(i));
          }
          correction->get(i) -=
              0.5 * max_abs_char_speed * (var_ext.get(i) - var_int.get(i));
        }
      };

  scalar_correction(
      boundary_correction_mass_density, normal_dot_flux_mass_density_int,
      normal_dot_flux_mass_density_ext, mass_density_int, mass_density_ext);
  vector_correction(boundary_correction_momentum_density,
                    normal_dot_flux_momentum_density_int,
                    normal_dot_flux_momentum_density_ext, momentum_density_int,
                    momentum_density_ext);
  scalar_correction(boundary_correction_energy_density,
                    normal_dot_flux_energy_density_int,
                    normal_dot_flux_energy_density_ext, energy_density_int,
                    energy_density_ext);
  vector_correction(boundary_correction_magnetic_field,
                    normal_dot_flux_magnetic_field_int,
                    normal_dot_flux_magnetic_field_ext, magnetic_field_int,
                    magnetic_field_ext);
  scalar_correction(boundary_correction_divergence_cleaning_field,
                    normal_dot_flux_divergence_cleaning_field_int,
                    normal_dot_flux_divergence_cleaning_field_ext,
                    divergence_cleaning_field_int,
                    divergence_cleaning_field_ext);
}

template <bool UseBackgroundMagneticField>
// NOLINTNEXTLINE
PUP::able::PUP_ID Rusanov<UseBackgroundMagneticField>::my_PUP_ID = 0;

#define USE_BG(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class Rusanov<USE_BG(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (true, false))

#undef INSTANTIATION
#undef USE_BG
}  // namespace NewtonianMhd::BoundaryCorrections
