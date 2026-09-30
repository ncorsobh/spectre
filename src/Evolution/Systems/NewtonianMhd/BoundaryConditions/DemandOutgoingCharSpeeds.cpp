// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/Systems/NewtonianMhd/Characteristics.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/SoundSpeedSquared.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::BoundaryConditions {
template <size_t Dim, bool UseBackgroundMagneticField>
DemandOutgoingCharSpeeds<Dim, UseBackgroundMagneticField>::
    DemandOutgoingCharSpeeds(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim, bool UseBackgroundMagneticField>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DemandOutgoingCharSpeeds<Dim, UseBackgroundMagneticField>::get_clone() const {
  return std::make_unique<DemandOutgoingCharSpeeds>(*this);
}

template <size_t Dim, bool UseBackgroundMagneticField>
void DemandOutgoingCharSpeeds<Dim, UseBackgroundMagneticField>::pup(
    PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim, bool UseBackgroundMagneticField>
// NOLINTNEXTLINE
PUP::able::PUP_ID
    DemandOutgoingCharSpeeds<Dim, UseBackgroundMagneticField>::my_PUP_ID = 0;

template <size_t Dim, bool UseBackgroundMagneticField>
template <size_t ThermodynamicDim>
std::optional<std::string>
DemandOutgoingCharSpeeds<Dim, UseBackgroundMagneticField>::
    dg_demand_outgoing_char_speeds(
        const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
            face_mesh_velocity,
        const tnsr::i<DataVector, Dim, Frame::Inertial>&
            outward_directed_normal_covector,

        const tnsr::I<DataVector, Dim, Frame::Inertial>& magnetic_field,
        const Scalar<DataVector>& mass_density,
        const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
        const Scalar<DataVector>& specific_internal_energy,
        const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
            equation_of_state) {
  // Selected by an empty `dg_interior_temporary_tags`, so that B0 is never
  // projected onto element faces when the splitting is disabled.
  if constexpr (UseBackgroundMagneticField) {
    ERROR(
        "Called the boundary condition overload that takes no background "
        "magnetic field, but the background-field splitting is enabled.");
  } else {
    return dg_demand_outgoing_char_speeds<ThermodynamicDim>(
        face_mesh_velocity, outward_directed_normal_covector, magnetic_field,
        mass_density, velocity, specific_internal_energy, {},
        equation_of_state);
  }
}

template <size_t Dim, bool UseBackgroundMagneticField>
template <size_t ThermodynamicDim>
std::optional<std::string>
DemandOutgoingCharSpeeds<Dim, UseBackgroundMagneticField>::
    dg_demand_outgoing_char_speeds(
        const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
            face_mesh_velocity,
        const tnsr::i<DataVector, Dim, Frame::Inertial>&
            outward_directed_normal_covector,

        const tnsr::I<DataVector, Dim, Frame::Inertial>& magnetic_field,
        const Scalar<DataVector>& mass_density,
        const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
        const Scalar<DataVector>& specific_internal_energy,
        const BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
            background_magnetic_field,
        const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
            equation_of_state) {
  double min_char_speed = std::numeric_limits<double>::signaling_NaN();

  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>>>
      buffer{get(mass_density).size()};

  auto& sound_speed_sq = get<::Tags::TempScalar<0>>(buffer);
  sound_speed_squared(make_not_null(&sound_speed_sq), mass_density,
                      specific_internal_energy, equation_of_state);
  auto& fast_speed = get<::Tags::TempScalar<1>>(buffer);
  fast_magnetosonic_speed<Dim, UseBackgroundMagneticField>(
      make_not_null(&fast_speed), mass_density, sound_speed_sq, magnetic_field,
      background_magnetic_field);

  auto& normal_dot_velocity = get<::Tags::TempScalar<2>>(buffer);
  dot_product(make_not_null(&normal_dot_velocity),
              outward_directed_normal_covector, velocity);

  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<3>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());
    min_char_speed = min(get(normal_dot_velocity) - get(fast_speed) -
                         get(normal_dot_mesh_velocity));
  } else {
    min_char_speed = min(get(normal_dot_velocity) - get(fast_speed));
  }

  if (min_char_speed < 0.0) {
    return {MakeString{}
            << "DemandOutgoingCharSpeeds boundary condition violated with the "
               "characteristic speed : "
            << min_char_speed << "\n"};
  }

  return std::nullopt;
}

template <size_t Dim, bool UseBackgroundMagneticField>
void DemandOutgoingCharSpeeds<Dim, UseBackgroundMagneticField>::
    fd_demand_outgoing_char_speeds(
        const gsl::not_null<Scalar<DataVector>*> mass_density,
        const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
            velocity,
        const gsl::not_null<Scalar<DataVector>*> pressure,
        const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
            magnetic_field,
        const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
        const Direction<Dim>& direction,
        const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
            face_mesh_velocity,
        const tnsr::i<DataVector, Dim, Frame::Inertial>&
            outward_directed_normal_covector,
        const Mesh<Dim>& subcell_mesh,
        const BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
            interior_background_magnetic_field,
        const Scalar<DataVector>& interior_mass_density,
        const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
        const Scalar<DataVector>& interior_specific_internal_energy,
        const Scalar<DataVector>& interior_pressure,
        const tnsr::I<DataVector, Dim, Frame::Inertial>&
            interior_magnetic_field,
        const Scalar<DataVector>& interior_divergence_cleaning_field,
        const EquationsOfState::EquationOfState<false, 2>& equation_of_state,
        const fd::Reconstructor<Dim>& reconstructor) {
  const size_t dim_direction = direction.dimension();
  const auto subcell_extents = subcell_mesh.extents();
  const size_t num_face_pts =
      subcell_extents.slice_away(dim_direction).product();

  const auto get_boundary_val = [&direction,
                                 &subcell_extents](const auto& volume_tensor) {
    return evolution::dg::subcell::slice_tensor_for_subcell(
        volume_tensor, subcell_extents, 1, direction, {});
  };

  const auto boundary_mass_density = get_boundary_val(interior_mass_density);
  const auto boundary_velocity = get_boundary_val(interior_velocity);
  const auto boundary_magnetic_field =
      get_boundary_val(interior_magnetic_field);

  {
    Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                         ::Tags::TempScalar<2>, ::Tags::TempScalar<3>>>
        buffer{num_face_pts};
    auto& sound_speed_sq = get<::Tags::TempScalar<0>>(buffer);
    sound_speed_squared(make_not_null(&sound_speed_sq), boundary_mass_density,
                        get_boundary_val(interior_specific_internal_energy),
                        equation_of_state);
    auto& fast_speed = get<::Tags::TempScalar<1>>(buffer);
    if constexpr (UseBackgroundMagneticField) {
      fast_magnetosonic_speed<Dim, true>(
          make_not_null(&fast_speed), boundary_mass_density, sound_speed_sq,
          boundary_magnetic_field,
          get_boundary_val(interior_background_magnetic_field));
    } else {
      fast_magnetosonic_speed<Dim, false>(make_not_null(&fast_speed),
                                          boundary_mass_density, sound_speed_sq,
                                          boundary_magnetic_field);
    }

    auto& normal_dot_velocity = get<::Tags::TempScalar<2>>(buffer);
    dot_product(make_not_null(&normal_dot_velocity),
                outward_directed_normal_covector, boundary_velocity);

    double min_char_speed = std::numeric_limits<double>::signaling_NaN();
    if (face_mesh_velocity.has_value()) {
      auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<3>>(buffer);
      dot_product(make_not_null(&normal_dot_mesh_velocity),
                  outward_directed_normal_covector, face_mesh_velocity.value());
      min_char_speed = min(get(normal_dot_velocity) - get(fast_speed) -
                           get(normal_dot_mesh_velocity));
    } else {
      min_char_speed = min(get(normal_dot_velocity) - get(fast_speed));
    }
    if (min_char_speed < 0.0) {
      ERROR(
          "Subcell DemandOutgoingCharSpeeds boundary condition violated. "
          "Speed: "
          << min_char_speed << "\nn_i: " << outward_directed_normal_covector
          << "\n");
    }
  }

  using MassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, Dim>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, Dim>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;
  using prim_tags = tmpl::list<MassDensity, Velocity, Pressure, MagneticField,
                               DivergenceCleaningField>;

  Variables<prim_tags> outermost_prim_vars{num_face_pts};
  get<MassDensity>(outermost_prim_vars) = boundary_mass_density;
  get<Velocity>(outermost_prim_vars) = boundary_velocity;
  get<Pressure>(outermost_prim_vars) = get_boundary_val(interior_pressure);
  get<MagneticField>(outermost_prim_vars) = boundary_magnetic_field;
  get<DivergenceCleaningField>(outermost_prim_vars) =
      get_boundary_val(interior_divergence_cleaning_field);

  const size_t ghost_zone_size = reconstructor.ghost_zone_size();
  Index<Dim> ghost_data_extents = subcell_extents;
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

template <size_t Dim, bool UseBackgroundMagneticField>
void DemandOutgoingCharSpeeds<Dim, UseBackgroundMagneticField>::
    fd_demand_outgoing_char_speeds(
        const gsl::not_null<Scalar<DataVector>*> mass_density,
        const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
            velocity,
        const gsl::not_null<Scalar<DataVector>*> pressure,
        const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
            magnetic_field,
        const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
        const Direction<Dim>& direction,
        const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
            face_mesh_velocity,
        const tnsr::i<DataVector, Dim, Frame::Inertial>&
            outward_directed_normal_covector,
        const Mesh<Dim>& subcell_mesh,
        const Scalar<DataVector>& interior_mass_density,
        const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
        const Scalar<DataVector>& interior_specific_internal_energy,
        const Scalar<DataVector>& interior_pressure,
        const tnsr::I<DataVector, Dim, Frame::Inertial>&
            interior_magnetic_field,
        const Scalar<DataVector>& interior_divergence_cleaning_field,
        const EquationsOfState::EquationOfState<false, 2>& equation_of_state,
        const fd::Reconstructor<Dim>& reconstructor) {
  // Selected by an empty background entry in `fd_interior_temporary_tags`, so
  // that B0 is never requested when the splitting is disabled.
  if constexpr (UseBackgroundMagneticField) {
    ERROR(
        "Called the boundary condition overload that takes no background "
        "magnetic field, but the background-field splitting is enabled.");
  } else {
    fd_demand_outgoing_char_speeds(
        mass_density, velocity, pressure, magnetic_field,
        divergence_cleaning_field, direction, face_mesh_velocity,
        outward_directed_normal_covector, subcell_mesh, {},
        interior_mass_density, interior_velocity,
        interior_specific_internal_energy, interior_pressure,
        interior_magnetic_field, interior_divergence_cleaning_field,
        equation_of_state, reconstructor);
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define USE_BG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data) \
  template class DemandOutgoingCharSpeeds<DIM(data), USE_BG(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (true, false))

#undef INSTANTIATION

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(_, data)                                               \
  template std::optional<std::string>                                        \
  DemandOutgoingCharSpeeds<DIM(data), USE_BG(data)>::                        \
      dg_demand_outgoing_char_speeds<THERMODIM(data)>(                       \
          const std::optional<tnsr::I<DataVector, DIM(data),                 \
                                      Frame::Inertial>>& face_mesh_velocity, \
          const tnsr::i<DataVector, DIM(data), Frame::Inertial>&             \
              outward_directed_normal_covector,                              \
          const tnsr::I<DataVector, DIM(data), Frame::Inertial>&             \
              magnetic_field,                                                \
          const Scalar<DataVector>& mass_density,                            \
          const tnsr::I<DataVector, DIM(data), Frame::Inertial>& velocity,   \
          const Scalar<DataVector>& specific_internal_energy,                \
          NewtonianMhd::BackgroundMagneticFieldArgument<DIM(data),           \
                                                        USE_BG(data)>        \
              background_magnetic_field,                                     \
          const EquationsOfState::EquationOfState<false, THERMODIM(data)>&   \
              equation_of_state);                                            \
  template std::optional<std::string>                                        \
  DemandOutgoingCharSpeeds<DIM(data), USE_BG(data)>::                        \
      dg_demand_outgoing_char_speeds<THERMODIM(data)>(                       \
          const std::optional<tnsr::I<DataVector, DIM(data),                 \
                                      Frame::Inertial>>& face_mesh_velocity, \
          const tnsr::i<DataVector, DIM(data), Frame::Inertial>&             \
              outward_directed_normal_covector,                              \
          const tnsr::I<DataVector, DIM(data), Frame::Inertial>&             \
              magnetic_field,                                                \
          const Scalar<DataVector>& mass_density,                            \
          const tnsr::I<DataVector, DIM(data), Frame::Inertial>& velocity,   \
          const Scalar<DataVector>& specific_internal_energy,                \
          const EquationsOfState::EquationOfState<false, THERMODIM(data)>&   \
              equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (true, false), (1, 2))

#undef INSTANTIATION
#undef THERMODIM
#undef USE_BG
#undef DIM
}  // namespace NewtonianMhd::BoundaryConditions
