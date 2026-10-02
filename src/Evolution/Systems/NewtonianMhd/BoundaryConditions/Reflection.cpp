// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/Reflection.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::BoundaryConditions {
namespace detail {
template <bool UseBackgroundMagneticField>
void reflection_dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field_cons,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        flux_mass_density,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
        flux_momentum_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        flux_energy_density,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
        flux_magnetic_field,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        flux_divergence_cleaning_field,
    const BackgroundMagneticFieldOutput<UseBackgroundMagneticField>
        background_magnetic_field,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_pressure,
    const double divergence_cleaning_speed, const bool no_slip,
    const BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
        interior_background_magnetic_field) {
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{
      get(interior_mass_density).size()};
  auto& normal_dot_velocity = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_velocity),
              outward_directed_normal_covector, interior_velocity);

  if (no_slip) {
    for (size_t i = 0; i < 3; ++i) {
      velocity->get(i) = -interior_velocity.get(i);
    }
  } else {
    for (size_t i = 0; i < 3; ++i) {
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
      for (size_t i = 0; i < 3; ++i) {
        velocity->get(i) += 2.0 * face_mesh_velocity.value().get(i);
      }
    } else {
      for (size_t i = 0; i < 3; ++i) {
        velocity->get(i) += 2.0 * get(normal_dot_mesh_velocity) *
                            outward_directed_normal_covector.get(i);
      }
    }
  }

  auto& normal_dot_magnetic_field = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_magnetic_field),
              outward_directed_normal_covector, interior_magnetic_field);
  tnsr::I<DataVector, 3, Frame::Inertial> ghost_magnetic_field{
      get(interior_mass_density).size()};
  for (size_t i = 0; i < 3; ++i) {
    ghost_magnetic_field.get(i) = interior_magnetic_field.get(i) -
                                  2.0 * get(normal_dot_magnetic_field) *
                                      outward_directed_normal_covector.get(i);
  }
  const Scalar<DataVector> ghost_divergence_cleaning_field{
      -get(interior_divergence_cleaning_field)};

  *specific_internal_energy = interior_specific_internal_energy;
  if constexpr (UseBackgroundMagneticField) {
    // B0 is smooth and continuous across the boundary, so the exterior value is
    // the interior one.
    *background_magnetic_field = interior_background_magnetic_field;

    if (not no_slip) {
      // Free slip leaves the tangential velocity arbitrary, so the perfect
      // conductor condition B_n (n x v_t) = 0 can only hold if the total
      // normal field vanishes. The reflection already removes B_1^i n_i, so
      // what is left to check is B_0^i n_i.
      const Scalar<DataVector> normal_dot_background = dot_product(
          outward_directed_normal_covector, interior_background_magnetic_field);
      const double background_magnitude =
          max(get(magnitude(interior_background_magnetic_field)));
      if (max(abs(get(normal_dot_background))) >
          1.0e-12 * std::max(background_magnitude, 1.0)) {
        ERROR(
            "The Reflection boundary condition requires the background "
            "magnetic field to be tangent to the boundary, but max|B_0^i n_i| "
            "is "
            << max(abs(get(normal_dot_background)))
            << " against a background field of magnitude "
            << background_magnitude
            << ". A boundary threaded by normal magnetic flux needs the "
               "tangential velocity to vanish as well; use "
               "ConductorReflection there.");
      }
    }
  }

  ConservativeFromPrimitive::apply(
      mass_density_cons, momentum_density, energy_density, magnetic_field_cons,
      divergence_cleaning_field_cons, interior_mass_density, *velocity,
      interior_specific_internal_energy, ghost_magnetic_field,
      ghost_divergence_cleaning_field);
  ComputeFluxes<UseBackgroundMagneticField>::apply(
      flux_mass_density, flux_momentum_density, flux_energy_density,
      flux_magnetic_field, flux_divergence_cleaning_field, *momentum_density,
      *energy_density, *magnetic_field_cons, *divergence_cleaning_field_cons,
      *velocity, interior_pressure, divergence_cleaning_speed,
      interior_background_magnetic_field);
}
void reflection_fd_ghost(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    const Direction<3>& direction, const Mesh<3>& subcell_mesh,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_pressure,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const size_t ghost_zone_size, const bool no_slip) {
  const size_t dim_direction = direction.dimension();
  const auto subcell_extents = subcell_mesh.extents();

  using MassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;
  using prim_tags = tmpl::list<MassDensity, Velocity, Pressure, MagneticField,
                               DivergenceCleaningField>;

  const size_t num_face_pts =
      subcell_extents.slice_away(dim_direction).product();
  Variables<prim_tags> outermost_prim_vars{num_face_pts};

  const auto get_boundary_val = [&direction,
                                 &subcell_extents](const auto& volume_tensor) {
    return evolution::dg::subcell::slice_tensor_for_subcell(
        volume_tensor, subcell_extents, 1, direction, {});
  };

  get<MassDensity>(outermost_prim_vars) =
      get_boundary_val(interior_mass_density);
  get<Pressure>(outermost_prim_vars) = get_boundary_val(interior_pressure);
  // Anti-symmetric, matching the DG ghost state: this is what makes the
  // interface value of psi, and hence of B^i n_i, vanish.
  get(get<DivergenceCleaningField>(outermost_prim_vars)) =
      -get(get_boundary_val(interior_divergence_cleaning_field));

  const auto boundary_velocity = get_boundary_val(interior_velocity);
  const auto boundary_magnetic_field =
      get_boundary_val(interior_magnetic_field);
  for (size_t i = 0; i < 3; ++i) {
    const bool flip_velocity = no_slip or i == dim_direction;
    get<Velocity>(outermost_prim_vars).get(i) =
        (flip_velocity ? -1.0 : 1.0) * boundary_velocity.get(i);
    get<MagneticField>(outermost_prim_vars).get(i) =
        (i == dim_direction ? -1.0 : 1.0) * boundary_magnetic_field.get(i);
  }

  Index<3> ghost_data_extents = subcell_extents;
  ghost_data_extents[dim_direction] = ghost_zone_size;
  Variables<prim_tags> ghost_prim_vars{ghost_data_extents.product(), 0.0};
  for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
    add_slice_to_data(make_not_null(&ghost_prim_vars), outermost_prim_vars,
                      ghost_data_extents, dim_direction, i_ghost);
  }

  *mass_density = get<MassDensity>(ghost_prim_vars);
  *velocity = get<Velocity>(ghost_prim_vars);
  *pressure = get<Pressure>(ghost_prim_vars);
  *magnetic_field = get<MagneticField>(ghost_prim_vars);
  *divergence_cleaning_field = get<DivergenceCleaningField>(ghost_prim_vars);
}
}  // namespace detail

template <bool UseBackgroundMagneticField>
Reflection<UseBackgroundMagneticField>::Reflection(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

template <bool UseBackgroundMagneticField>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Reflection<UseBackgroundMagneticField>::get_clone() const {
  return std::make_unique<Reflection>(*this);
}

template <bool UseBackgroundMagneticField>
void Reflection<UseBackgroundMagneticField>::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
}

template <bool UseBackgroundMagneticField>
std::optional<std::string> Reflection<UseBackgroundMagneticField>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field_cons,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        flux_mass_density,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
        flux_momentum_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        flux_energy_density,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
        flux_magnetic_field,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        flux_divergence_cleaning_field,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,

    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,
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

template <bool UseBackgroundMagneticField>
std::optional<std::string> Reflection<UseBackgroundMagneticField>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field_cons,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        flux_mass_density,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
        flux_momentum_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        flux_energy_density,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
        flux_magnetic_field,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        flux_divergence_cleaning_field,

    const BackgroundMagneticFieldOutput<UseBackgroundMagneticField>
        background_magnetic_field,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,

    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_pressure,
    const BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
        interior_background_magnetic_field,
    const double divergence_cleaning_speed) const {
  detail::reflection_dg_ghost<UseBackgroundMagneticField>(
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

template <bool UseBackgroundMagneticField>
// NOLINTNEXTLINE
PUP::able::PUP_ID Reflection<UseBackgroundMagneticField>::my_PUP_ID = 0;

template <bool UseBackgroundMagneticField>
void Reflection<UseBackgroundMagneticField>::fd_ghost(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    const Direction<3>& direction, const Mesh<3>& subcell_mesh,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_pressure,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const fd::Reconstructor& reconstructor) const {
  detail::reflection_fd_ghost(
      mass_density, velocity, pressure, magnetic_field,
      divergence_cleaning_field, direction, subcell_mesh, interior_mass_density,
      interior_velocity, interior_pressure, interior_magnetic_field,
      interior_divergence_cleaning_field, reconstructor.ghost_zone_size(),
      false);
}

#define USE_BG(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                \
  template class Reflection<USE_BG(data)>;                                    \
  template void detail::reflection_dg_ghost<USE_BG(data)>(                    \
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,                   \
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>                 \
          momentum_density,                                                   \
      gsl::not_null<Scalar<DataVector>*> energy_density,                      \
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>                 \
          magnetic_field_cons,                                                \
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,      \
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>                 \
          flux_mass_density,                                                  \
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>                \
          flux_momentum_density,                                              \
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>                 \
          flux_energy_density,                                                \
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>                \
          flux_magnetic_field,                                                \
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>                 \
          flux_divergence_cleaning_field,                                     \
      NewtonianMhd::BackgroundMagneticFieldOutput<USE_BG(data)>               \
          background_magnetic_field,                                          \
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,       \
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,            \
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&           \
          face_mesh_velocity,                                                 \
      const tnsr::i<DataVector, 3, Frame::Inertial>&                          \
          outward_directed_normal_covector,                                   \
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field, \
      const Scalar<DataVector>& interior_divergence_cleaning_field,           \
      const Scalar<DataVector>& interior_mass_density,                        \
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,       \
      const Scalar<DataVector>& interior_specific_internal_energy,            \
      const Scalar<DataVector>& interior_pressure,                            \
      double divergence_cleaning_speed, bool no_slip,                         \
      NewtonianMhd::BackgroundMagneticFieldArgument<USE_BG(data)>             \
          interior_background_magnetic_field);

GENERATE_INSTANTIATIONS(INSTANTIATION, (true, false))

#undef INSTANTIATION
#undef USE_BG

}  // namespace NewtonianMhd::BoundaryConditions
