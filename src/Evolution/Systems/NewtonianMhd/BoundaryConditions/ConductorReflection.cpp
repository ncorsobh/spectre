// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/ConductorReflection.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/Reflection.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd::BoundaryConditions {
template <size_t Dim, bool UseBackgroundMagneticField>
ConductorReflection<Dim, UseBackgroundMagneticField>::ConductorReflection(
    CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim, bool UseBackgroundMagneticField>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
ConductorReflection<Dim, UseBackgroundMagneticField>::get_clone() const {
  return std::make_unique<ConductorReflection>(*this);
}

template <size_t Dim, bool UseBackgroundMagneticField>
void ConductorReflection<Dim, UseBackgroundMagneticField>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim, bool UseBackgroundMagneticField>
std::optional<std::string>
ConductorReflection<Dim, UseBackgroundMagneticField>::dg_ghost(
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
ConductorReflection<Dim, UseBackgroundMagneticField>::dg_ghost(
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
      divergence_cleaning_speed, true, interior_background_magnetic_field);
  return {};
}

template <size_t Dim, bool UseBackgroundMagneticField>
// NOLINTNEXTLINE
PUP::able::PUP_ID
    ConductorReflection<Dim, UseBackgroundMagneticField>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define USE_BG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data) \
  template class ConductorReflection<DIM(data), USE_BG(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (true, false))

#undef INSTANTIATION
#undef USE_BG
#undef DIM
}  // namespace NewtonianMhd::BoundaryConditions
