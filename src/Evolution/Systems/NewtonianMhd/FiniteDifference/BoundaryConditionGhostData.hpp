// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::fd {
/*!
 * \brief Computes finite difference ghost data for external boundary
 * conditions.
 *
 * If the element is at the external boundary, computes FD ghost data with a
 * given boundary condition and stores it into neighbor data with {direction,
 * ElementId::external_boundary_id()} as the mortar_id key.
 *
 * \note Subcell needs to be enabled for boundary elements. Otherwise this
 * function would never be called.
 */
struct BoundaryConditionGhostData {
  template <typename DbTagsList>
  static void apply(gsl::not_null<db::DataBox<DbTagsList>*> box,
                    const Element<3>& element,
                    const Reconstructor& reconstructor);

 private:
  template <typename FdBoundaryConditionHelper, typename DbTagsList,
            typename... FdBoundaryConditionArgsTags>
  // A helper function for calling fd_ghost() of BoundaryCondition subclasses
  static void apply_subcell_boundary_condition_impl(
      FdBoundaryConditionHelper& fd_boundary_condition_helper,
      const gsl::not_null<db::DataBox<DbTagsList>*>& box,
      tmpl::list<FdBoundaryConditionArgsTags...> /*meta*/) {
    return fd_boundary_condition_helper(
        db::get<FdBoundaryConditionArgsTags>(*box)...);
  }
};

template <typename DbTagsList>
void BoundaryConditionGhostData::apply(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const Element<3>& element, const Reconstructor& reconstructor) {
  const auto& external_boundary_condition =
      db::get<domain::Tags::ExternalBoundaryConditions<3>>(*box).at(
          element.id().block_id());

  ASSERT(not element.external_boundaries().empty(),
         "The element (ID : " << element.id()
                              << ") is not on external boundaries");

  const Mesh<3> subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box);

  const size_t ghost_zone_size{reconstructor.ghost_zone_size()};

  using MassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;

  // Must match `PrimitiveGhostVariables::prims_to_reconstruct_tags`, since the
  // reconstructor reads neighbour and boundary ghost data through the same
  // layout.
  using prims_for_reconstruction =
      tmpl::list<MassDensity, Velocity, Pressure, MagneticField,
                 DivergenceCleaningField>;

  for (const auto& direction : element.external_boundaries()) {
    const auto& boundary_condition_at_direction =
        *external_boundary_condition.at(direction);

    const size_t num_face_pts{
        subcell_mesh.extents().slice_away(direction.dimension()).product()};

    auto& all_ghost_data = db::get_mutable_reference<
        evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);
    const DirectionalId<3> mortar_id{direction,
                                     ElementId<3>::external_boundary_id()};

    all_ghost_data[mortar_id] = evolution::dg::subcell::GhostData{1};
    DataVector& boundary_ghost_data =
        all_ghost_data.at(mortar_id).neighbor_ghost_data_for_reconstruction();
    const size_t prims_size =
        Variables<prims_for_reconstruction>::number_of_independent_components *
        ghost_zone_size * num_face_pts;
    boundary_ghost_data.destructive_resize(prims_size);
    Variables<prims_for_reconstruction> ghost_data_vars{
        boundary_ghost_data.data(), prims_size};

    // Periodic and none conditions never reach here, so they are excluded to
    // keep them out of the dynamic dispatch.
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::factory_creation::factory_classes;
    using derived_boundary_conditions_for_subcell = tmpl::remove_if<
        tmpl::at<factory_classes,
                 NewtonianMhd::BoundaryConditions::BoundaryCondition>,
        tmpl::or_<
            std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic,
                            tmpl::_1>,
            std::is_base_of<domain::BoundaryConditions::MarkAsNone, tmpl::_1>>>;

    call_with_dynamic_type<void, derived_boundary_conditions_for_subcell>(
        &boundary_condition_at_direction,
        [&box, &direction, &ghost_data_vars](const auto* boundary_condition) {
          using BoundaryCondition = std::decay_t<decltype(*boundary_condition)>;
          using bcondition_interior_evolved_vars_tags =
              typename BoundaryCondition::fd_interior_evolved_variables_tags;
          using bcondition_interior_temporary_tags =
              typename BoundaryCondition::fd_interior_temporary_tags;
          using bcondition_interior_primitive_vars_tags =
              typename BoundaryCondition::fd_interior_primitive_variables_tags;
          using bcondition_gridless_tags =
              typename BoundaryCondition::fd_gridless_tags;

          using bcondition_interior_tags =
              tmpl::append<bcondition_interior_evolved_vars_tags,
                           bcondition_interior_temporary_tags,
                           bcondition_interior_primitive_vars_tags,
                           bcondition_gridless_tags>;

          if constexpr (BoundaryCondition::bc_type ==
                        evolution::BoundaryConditions::Type::Ghost) {
            const auto apply_fd_ghost =
                [&boundary_condition, &direction,
                 &ghost_data_vars](const auto&... boundary_ghost_data_args) {
                  (*boundary_condition)
                      .fd_ghost(
                          make_not_null(&get<MassDensity>(ghost_data_vars)),
                          make_not_null(&get<Velocity>(ghost_data_vars)),
                          make_not_null(&get<Pressure>(ghost_data_vars)),
                          make_not_null(&get<MagneticField>(ghost_data_vars)),
                          make_not_null(
                              &get<DivergenceCleaningField>(ghost_data_vars)),
                          direction, boundary_ghost_data_args...);
                };
            apply_subcell_boundary_condition_impl(apply_fd_ghost, box,
                                                  bcondition_interior_tags{});
          } else if constexpr (BoundaryCondition::bc_type ==
                               evolution::BoundaryConditions::Type::
                                   DemandOutgoingCharSpeeds) {
            const auto& volume_mesh_velocity =
                db::get<domain::Tags::MeshVelocity<3, Frame::Inertial>>(*box);
            if (volume_mesh_velocity.has_value()) {
              ERROR("Subcell currently does not support moving mesh");
            }

            const std::optional<tnsr::I<DataVector, 3>> face_mesh_velocity{};

            const auto& normal_covector_and_magnitude =
                db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<3>>(
                    *box);
            const auto& outward_directed_normal_covector =
                get<evolution::dg::Tags::NormalCovector<3>>(
                    normal_covector_and_magnitude.at(direction).value());

            const auto apply_fd_demand_outgoing_char_speeds =
                [&boundary_condition, &direction, &face_mesh_velocity,
                 &ghost_data_vars, &outward_directed_normal_covector](
                    const auto&... boundary_ghost_data_args) {
                  return (*boundary_condition)
                      .fd_demand_outgoing_char_speeds(
                          make_not_null(&get<MassDensity>(ghost_data_vars)),
                          make_not_null(&get<Velocity>(ghost_data_vars)),
                          make_not_null(&get<Pressure>(ghost_data_vars)),
                          make_not_null(&get<MagneticField>(ghost_data_vars)),
                          make_not_null(
                              &get<DivergenceCleaningField>(ghost_data_vars)),
                          direction, face_mesh_velocity,
                          outward_directed_normal_covector,
                          boundary_ghost_data_args...);
                };
            apply_subcell_boundary_condition_impl(
                apply_fd_demand_outgoing_char_speeds, box,
                bcondition_interior_tags{});
          } else {
            ERROR("Unsupported boundary condition "
                  << pretty_type::short_name<BoundaryCondition>()
                  << " when using finite-difference");
          }
        });
  }
}
}  // namespace NewtonianMhd::fd
