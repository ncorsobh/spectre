// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/ElementMap.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/Systems/NewtonianMhd/AllSolutions.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd::BoundaryConditions {
template <bool UseBackgroundMagneticField>
DirichletAnalytic<UseBackgroundMagneticField>::DirichletAnalytic(
    const DirichletAnalytic<UseBackgroundMagneticField>& rhs)
    : BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

template <bool UseBackgroundMagneticField>
DirichletAnalytic<UseBackgroundMagneticField>&
DirichletAnalytic<UseBackgroundMagneticField>::operator=(
    const DirichletAnalytic<UseBackgroundMagneticField>& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

template <bool UseBackgroundMagneticField>
DirichletAnalytic<UseBackgroundMagneticField>::DirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

template <bool UseBackgroundMagneticField>
DirichletAnalytic<UseBackgroundMagneticField>::DirichletAnalytic(
    CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

template <bool UseBackgroundMagneticField>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic<UseBackgroundMagneticField>::get_clone() const {
  return std::make_unique<DirichletAnalytic>(*this);
}

template <bool UseBackgroundMagneticField>
void DirichletAnalytic<UseBackgroundMagneticField>::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | analytic_prescription_;
}

template <bool UseBackgroundMagneticField>
std::optional<std::string>
DirichletAnalytic<UseBackgroundMagneticField>::dg_ghost(
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
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,

    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const double divergence_cleaning_speed) const {
  // Selected by an empty `dg_package_data_temporary_tags` on the boundary
  // correction, so that B0 is never projected onto element faces when the
  // splitting is disabled.
  if constexpr (UseBackgroundMagneticField) {
    ERROR(
        "Called the boundary condition overload that takes no background "
        "magnetic field, but the background-field splitting is enabled.");
  } else {
    return dg_ghost(
        mass_density_cons, momentum_density, energy_density,
        magnetic_field_cons, divergence_cleaning_field_cons, flux_mass_density,
        flux_momentum_density, flux_energy_density, flux_magnetic_field,
        flux_divergence_cleaning_field, {}, velocity, specific_internal_energy,
        face_mesh_velocity, normal_covector, coords, {}, time,
        divergence_cleaning_speed);
  }
}

template <bool UseBackgroundMagneticField>
std::optional<std::string>
DirichletAnalytic<UseBackgroundMagneticField>::dg_ghost(
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

    const std::optional<
        tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,

    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
    const BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
        interior_background_magnetic_field,
    const double time, const double divergence_cleaning_speed) const {
  using boundary_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

  auto boundary_values =
      call_with_dynamic_type<tuples::tagged_tuple_from_typelist<boundary_tags>,
                             NewtonianMhd::InitialData::initial_data_list>(
          analytic_prescription_.get(),
          [&coords, &time](const auto* const initial_data) {
            if constexpr (is_analytic_solution_v<
                              std::decay_t<decltype(*initial_data)>>) {
              return initial_data->variables(coords, time, boundary_tags{});
            } else {
              (void)time;
              return initial_data->variables(coords, boundary_tags{});
            }
          });

  *velocity = get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values);
  *specific_internal_energy =
      get<hydro::Tags::SpecificInternalEnergy<DataVector>>(boundary_values);
  auto& total_magnetic_field =
      get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_values);
  if constexpr (UseBackgroundMagneticField) {
    // B0 is smooth and continuous across the boundary, so the exterior value is
    // the interior one. The prescription gives the total field, so the evolved
    // perturbation is what is left after removing B0.
    *background_magnetic_field = interior_background_magnetic_field;
    for (size_t i = 0; i < 3; ++i) {
      total_magnetic_field.get(i) -= interior_background_magnetic_field.get(i);
    }
  }

  ConservativeFromPrimitive::apply(
      mass_density_cons, momentum_density, energy_density, magnetic_field_cons,
      divergence_cleaning_field_cons,
      get<hydro::Tags::RestMassDensity<DataVector>>(boundary_values), *velocity,
      *specific_internal_energy, total_magnetic_field,
      get<hydro::Tags::DivergenceCleaningField<DataVector>>(boundary_values));
  ComputeFluxes<UseBackgroundMagneticField>::apply(
      flux_mass_density, flux_momentum_density, flux_energy_density,
      flux_magnetic_field, flux_divergence_cleaning_field, *momentum_density,
      *energy_density, *magnetic_field_cons, *divergence_cleaning_field_cons,
      *velocity, get<hydro::Tags::Pressure<DataVector>>(boundary_values),
      divergence_cleaning_speed, interior_background_magnetic_field);

  return {};
}

template <bool UseBackgroundMagneticField>
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic<UseBackgroundMagneticField>::my_PUP_ID = 0;

template <bool UseBackgroundMagneticField>
void DirichletAnalytic<UseBackgroundMagneticField>::fd_ghost(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    const Direction<3>& direction, const Mesh<3>& subcell_mesh,
    const double time,
    const std::unordered_map<
        std::string,
        std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const ElementMap<3, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const fd::Reconstructor& reconstructor) const {
  const auto ghost_logical_coords =
      evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
          subcell_mesh, reconstructor.ghost_zone_size(), direction);
  const auto coords = grid_to_inertial_map(
      logical_to_grid_map(ghost_logical_coords), time, functions_of_time);

  using boundary_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

  auto boundary_values =
      call_with_dynamic_type<tuples::tagged_tuple_from_typelist<boundary_tags>,
                             NewtonianMhd::InitialData::initial_data_list>(
          analytic_prescription_.get(),
          [&coords, &time](const auto* const initial_data) {
            if constexpr (is_analytic_solution_v<
                              std::decay_t<decltype(*initial_data)>>) {
              return initial_data->variables(coords, time, boundary_tags{});
            } else {
              (void)time;
              return initial_data->variables(coords, boundary_tags{});
            }
          });

  *mass_density =
      get<hydro::Tags::RestMassDensity<DataVector>>(boundary_values);
  *velocity = get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values);
  *pressure = get<hydro::Tags::Pressure<DataVector>>(boundary_values);
  *magnetic_field =
      get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_values);
  *divergence_cleaning_field =
      get<hydro::Tags::DivergenceCleaningField<DataVector>>(boundary_values);

  if constexpr (UseBackgroundMagneticField) {
    // The prescription gives the total field, but the reconstructed variable
    // is the evolved perturbation, so B0 is evaluated on the ghost zone and
    // removed.
    using background_tag =
        tmpl::list<NewtonianMhd::Tags::BackgroundMagneticFieldVolume<>>;
    const auto background = call_with_dynamic_type<
        tuples::tagged_tuple_from_typelist<background_tag>,
        NewtonianMhd::InitialData::background_magnetic_field_initial_data_list>(
        analytic_prescription_.get(),
        [&coords, &time](const auto* const initial_data) {
          if constexpr (is_analytic_solution_v<
                            std::decay_t<decltype(*initial_data)>>) {
            return initial_data->variables(coords, time, background_tag{});
          } else {
            (void)time;
            return initial_data->variables(coords, background_tag{});
          }
        });
    for (size_t i = 0; i < 3; ++i) {
      magnetic_field->get(i) -=
          get<NewtonianMhd::Tags::BackgroundMagneticFieldVolume<>>(background)
              .get(i);
    }
  }
}

#define USE_BG(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class DirichletAnalytic<USE_BG(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (true, false))

#undef INSTANTIATION
#undef USE_BG
}  // namespace NewtonianMhd::BoundaryConditions
