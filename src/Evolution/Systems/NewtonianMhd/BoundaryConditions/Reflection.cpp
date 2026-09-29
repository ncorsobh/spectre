// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/Reflection.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::BoundaryConditions {
namespace detail {
template <size_t Dim, bool UseBackgroundMagneticField>
void reflection_dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        magnetic_field_cons,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_mass_density,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_momentum_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_energy_density,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_magnetic_field,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_divergence_cleaning_field,
    const BackgroundMagneticFieldOutput<Dim, UseBackgroundMagneticField>
        background_magnetic_field,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_pressure,
    const double divergence_cleaning_speed, const bool no_slip,
    const BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
        interior_background_magnetic_field) {
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{
      get(interior_mass_density).size()};
  auto& normal_dot_velocity = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_velocity),
              outward_directed_normal_covector, interior_velocity);

  if (no_slip) {
    for (size_t i = 0; i < Dim; ++i) {
      velocity->get(i) = -interior_velocity.get(i);
    }
  } else {
    for (size_t i = 0; i < Dim; ++i) {
      velocity->get(i) = interior_velocity.get(i) -
                         2.0 * get(normal_dot_velocity) *
                             outward_directed_normal_covector.get(i);
    }
  }
  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<1>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());
    if (no_slip) {
      for (size_t i = 0; i < Dim; ++i) {
        velocity->get(i) += 2.0 * face_mesh_velocity.value().get(i);
      }
    } else {
      for (size_t i = 0; i < Dim; ++i) {
        velocity->get(i) += 2.0 * get(normal_dot_mesh_velocity) *
                            outward_directed_normal_covector.get(i);
      }
    }
  }

  auto& normal_dot_magnetic_field = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_magnetic_field),
              outward_directed_normal_covector, interior_magnetic_field);
  tnsr::I<DataVector, Dim, Frame::Inertial> ghost_magnetic_field{
      get(interior_mass_density).size()};
  for (size_t i = 0; i < Dim; ++i) {
    ghost_magnetic_field.get(i) = interior_magnetic_field.get(i) -
                                  2.0 * get(normal_dot_magnetic_field) *
                                      outward_directed_normal_covector.get(i);
  }
  Scalar<DataVector> ghost_divergence_cleaning_field{
      -get(interior_divergence_cleaning_field)};

  *specific_internal_energy = interior_specific_internal_energy;
  if constexpr (UseBackgroundMagneticField) {
    // B0 is smooth and continuous across the boundary, so the exterior value is
    // the interior one.
    *background_magnetic_field = interior_background_magnetic_field;
  }

  ConservativeFromPrimitive<Dim>::apply(
      mass_density_cons, momentum_density, energy_density, magnetic_field_cons,
      divergence_cleaning_field_cons, interior_mass_density, *velocity,
      interior_specific_internal_energy, ghost_magnetic_field,
      ghost_divergence_cleaning_field);
  ComputeFluxes<Dim, UseBackgroundMagneticField>::apply(
      flux_mass_density, flux_momentum_density, flux_energy_density,
      flux_magnetic_field, flux_divergence_cleaning_field, *momentum_density,
      *energy_density, *magnetic_field_cons, *divergence_cleaning_field_cons,
      *velocity, interior_pressure, divergence_cleaning_speed,
      interior_background_magnetic_field);
}
}  // namespace detail

template <size_t Dim, bool UseBackgroundMagneticField>
Reflection<Dim, UseBackgroundMagneticField>::Reflection(
    CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim, bool UseBackgroundMagneticField>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Reflection<Dim, UseBackgroundMagneticField>::get_clone() const {
  return std::make_unique<Reflection>(*this);
}

template <size_t Dim, bool UseBackgroundMagneticField>
void Reflection<Dim, UseBackgroundMagneticField>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim, bool UseBackgroundMagneticField>
std::optional<std::string>
Reflection<Dim, UseBackgroundMagneticField>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        magnetic_field_cons,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,

    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_mass_density,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_momentum_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_energy_density,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_magnetic_field,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_divergence_cleaning_field,

    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        outward_directed_normal_covector,

    const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_pressure,
    const double divergence_cleaning_speed) const {
  // Selected by an empty `dg_package_data_temporary_tags` on the boundary
  // correction, so that B0 is never projected onto element faces when the
  // splitting is disabled.
  if constexpr (UseBackgroundMagneticField) {
    ERROR(
        "Called the boundary condition overload that takes no background "
        "magnetic field, but the background-field splitting is enabled.");
  } else {
    return dg_ghost(mass_density_cons, momentum_density, energy_density,
                    magnetic_field_cons, divergence_cleaning_field_cons,
                    flux_mass_density, flux_momentum_density,
                    flux_energy_density, flux_magnetic_field,
                    flux_divergence_cleaning_field, {}, velocity,
                    specific_internal_energy, face_mesh_velocity,
                    outward_directed_normal_covector, interior_magnetic_field,
                    interior_divergence_cleaning_field, interior_mass_density,
                    interior_velocity, interior_specific_internal_energy,
                    interior_pressure, {}, divergence_cleaning_speed);
  }
}

template <size_t Dim, bool UseBackgroundMagneticField>
std::optional<std::string>
Reflection<Dim, UseBackgroundMagneticField>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        magnetic_field_cons,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,

    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_mass_density,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_momentum_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_energy_density,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_magnetic_field,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_divergence_cleaning_field,

    const BackgroundMagneticFieldOutput<Dim, UseBackgroundMagneticField>
        background_magnetic_field,

    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        outward_directed_normal_covector,

    const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_pressure,
    const BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
        interior_background_magnetic_field,
    const double divergence_cleaning_speed) const {
  detail::reflection_dg_ghost<Dim, UseBackgroundMagneticField>(
      mass_density_cons, momentum_density, energy_density, magnetic_field_cons,
      divergence_cleaning_field_cons, flux_mass_density, flux_momentum_density,
      flux_energy_density, flux_magnetic_field, flux_divergence_cleaning_field,
      background_magnetic_field, velocity, specific_internal_energy,
      face_mesh_velocity, outward_directed_normal_covector,
      interior_magnetic_field, interior_divergence_cleaning_field,
      interior_mass_density, interior_velocity,
      interior_specific_internal_energy, interior_pressure,
      divergence_cleaning_speed, false, interior_background_magnetic_field);
  return {};
}

template <size_t Dim, bool UseBackgroundMagneticField>
// NOLINTNEXTLINE
PUP::able::PUP_ID Reflection<Dim, UseBackgroundMagneticField>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define USE_BG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                               \
  template class Reflection<DIM(data), USE_BG(data)>;                        \
  template void detail::reflection_dg_ghost<DIM(data), USE_BG(data)>(        \
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,                  \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>        \
          momentum_density,                                                  \
      gsl::not_null<Scalar<DataVector>*> energy_density,                     \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>        \
          magnetic_field_cons,                                               \
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,     \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>        \
          flux_mass_density,                                                 \
      gsl::not_null<tnsr::IJ<DataVector, DIM(data), Frame::Inertial>*>       \
          flux_momentum_density,                                             \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>        \
          flux_energy_density,                                               \
      gsl::not_null<tnsr::IJ<DataVector, DIM(data), Frame::Inertial>*>       \
          flux_magnetic_field,                                               \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>        \
          flux_divergence_cleaning_field,                                    \
      NewtonianMhd::BackgroundMagneticFieldOutput<DIM(data), USE_BG(data)>   \
          background_magnetic_field,                                         \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>        \
          velocity,                                                          \
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,           \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&  \
          face_mesh_velocity,                                                \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                 \
          outward_directed_normal_covector,                                  \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                 \
          interior_magnetic_field,                                           \
      const Scalar<DataVector>& interior_divergence_cleaning_field,          \
      const Scalar<DataVector>& interior_mass_density,                       \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                 \
          interior_velocity,                                                 \
      const Scalar<DataVector>& interior_specific_internal_energy,           \
      const Scalar<DataVector>& interior_pressure,                           \
      double divergence_cleaning_speed, bool no_slip,                        \
      NewtonianMhd::BackgroundMagneticFieldArgument<DIM(data), USE_BG(data)> \
          interior_background_magnetic_field);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (true, false))

#undef INSTANTIATION
#undef USE_BG
#undef DIM
}  // namespace NewtonianMhd::BoundaryConditions
