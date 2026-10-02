// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <unordered_set>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::NewtonianMhd::fd {
using GhostData = evolution::dg::subcell::GhostData;

template <typename F>
DirectionalIdMap<3, GhostData> compute_ghost_data(
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& volume_logical_coords,
    const DirectionMap<3, Neighbors<3>>& neighbors,
    const size_t ghost_zone_size, const F& compute_variables_of_neighbor_data) {
  DirectionalIdMap<3, GhostData> ghost_data{};
  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    REQUIRE(neighbors_in_direction.size() == 1);
    const ElementId<3>& neighbor_id = *neighbors_in_direction.begin();
    auto neighbor_logical_coords = volume_logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    const auto neighbor_vars_for_reconstruction =
        compute_variables_of_neighbor_data(neighbor_logical_coords);

    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars_for_reconstruction.data(),
                       neighbor_vars_for_reconstruction.size()),
        subcell_mesh.extents(), ghost_zone_size,
        std::unordered_set{direction.opposite()}, 0, {});
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    ghost_data[DirectionalId<3>{direction, neighbor_id}] = GhostData{1};
    ghost_data.at(DirectionalId<3>{direction, neighbor_id})
        .neighbor_ghost_data_for_reconstruction() =
        sliced_data.at(direction.opposite());
  }
  return ghost_data;
}

namespace detail {
template <typename Reconstructor>
void test_prim_reconstructor_impl(
    const size_t points_per_dimension,
    const Reconstructor& derived_reconstructor,
    const EquationsOfState::EquationOfState<false, 2>& eos) {
  // Reconstruct a field that is linear in the logical coordinates, which every
  // reconstruction scheme here must reproduce exactly, and check both the
  // reconstructed primitives and the conservative variables computed from them.
  namespace nm = ::NewtonianMhd;
  const nm::fd::Reconstructor& reconstructor = derived_reconstructor;
  static_assert(
      tmpl::list_contains_v<typename nm::fd::Reconstructor::creatable_classes,
                            Reconstructor>);

  using MassDensityCons = nm::Tags::MassDensityCons;
  using MomentumDensity = nm::Tags::MomentumDensity<>;
  using EnergyDensity = nm::Tags::EnergyDensity;
  using MagneticFieldCons = nm::Tags::MagneticFieldCons<>;
  using DivergenceCleaningFieldCons = nm::Tags::DivergenceCleaningFieldCons;

  using MassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
  using SpecificInternalEnergy =
      hydro::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;

  using prims_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure,
                 MagneticField, DivergenceCleaningField>;
  using cons_tags = tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity,
                               MagneticFieldCons, DivergenceCleaningFieldCons>;
  using flux_tags = db::wrap_tags_in<::Tags::Flux, cons_tags, tmpl::size_t<3>,
                                     Frame::Inertial>;
  using prim_tags_for_reconstruction =
      tmpl::list<MassDensity, Velocity, Pressure, MagneticField,
                 DivergenceCleaningField>;

  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(subcell_mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < 3; ++i) {
    logical_coords.get(i) += 4.0 * static_cast<double>(i);
  }

  DirectionMap<3, Neighbors<3>> neighbors{};
  for (size_t i = 0; i < 2_st * 3; ++i) {
    neighbors[gsl::at(Direction<3>::all_directions(), i)] = Neighbors<3>{
        {ElementId<3>{i + 1, {}}}, OrientationMap<3>::create_aligned()};
  }
  const Element<3> element{ElementId<3>{0, {}}, neighbors};
  const auto compute_solution = [](const auto& coords) {
    Variables<prim_tags_for_reconstruction> vars{get<0>(coords).size(), 0.0};
    for (size_t i = 0; i < 3; ++i) {
      get(get<MassDensity>(vars)) += coords.get(i);
      get(get<Pressure>(vars)) += coords.get(i);
      get(get<DivergenceCleaningField>(vars)) += 0.1 * coords.get(i);
      for (size_t j = 0; j < 3; ++j) {
        get<Velocity>(vars).get(j) += coords.get(i);
        get<MagneticField>(vars).get(j) += 0.5 * coords.get(i);
      }
    }
    get(get<MassDensity>(vars)) += 2.0;
    get(get<Pressure>(vars)) += 30.0;
    get(get<DivergenceCleaningField>(vars)) += 0.3;
    for (size_t j = 0; j < 3; ++j) {
      get<Velocity>(vars).get(j) +=
          1.0e-2 * (static_cast<double>(j) + 2.0) + 10.0;
      get<MagneticField>(vars).get(j) +=
          1.0e-2 * (static_cast<double>(j) + 3.0) + 1.0;
    }
    return vars;
  };

  const DirectionalIdMap<3, GhostData> ghost_data =
      compute_ghost_data(subcell_mesh, logical_coords, element.neighbors(),
                         reconstructor.ghost_zone_size(), compute_solution);

  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();

  using dg_package_data_argument_tags =
      tmpl::append<cons_tags, prims_tags, flux_tags>;
  auto vars_on_lower_face = make_array<3>(
      Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
  auto vars_on_upper_face = make_array<3>(
      Variables<dg_package_data_argument_tags>(reconstructed_num_pts));

  Variables<prims_tags> volume_prims{subcell_mesh.number_of_grid_points()};
  volume_prims.assign_subset(compute_solution(logical_coords));

  dynamic_cast<const Reconstructor&>(reconstructor)
      .reconstruct(make_not_null(&vars_on_lower_face),
                   make_not_null(&vars_on_upper_face), volume_prims, eos,
                   element, ghost_data, subcell_mesh);

  for (size_t dim = 0; dim < 3; ++dim) {
    CAPTURE(dim);
    const auto basis = make_array<3>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<3>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<3>(points_per_dimension);
    gsl::at(extents, dim) = points_per_dimension + 1;
    gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
    const Mesh<3> face_centered_mesh{extents, basis, quadrature};
    auto logical_coords_face_centered = logical_coordinates(face_centered_mesh);
    for (size_t i = 1; i < 3; ++i) {
      logical_coords_face_centered.get(i) =
          logical_coords_face_centered.get(i) + 4.0 * static_cast<double>(i);
    }
    Variables<dg_package_data_argument_tags> expected_face_values{
        face_centered_mesh.number_of_grid_points()};
    expected_face_values.assign_subset(
        compute_solution(logical_coords_face_centered));
    get<SpecificInternalEnergy>(expected_face_values) =
        eos.specific_internal_energy_from_density_and_pressure(
            get<MassDensity>(expected_face_values),
            get<Pressure>(expected_face_values));
    nm::ConservativeFromPrimitive::apply(
        make_not_null(&get<MassDensityCons>(expected_face_values)),
        make_not_null(&get<MomentumDensity>(expected_face_values)),
        make_not_null(&get<EnergyDensity>(expected_face_values)),
        make_not_null(&get<MagneticFieldCons>(expected_face_values)),
        make_not_null(&get<DivergenceCleaningFieldCons>(expected_face_values)),
        get<MassDensity>(expected_face_values),
        get<Velocity>(expected_face_values),
        get<SpecificInternalEnergy>(expected_face_values),
        get<MagneticField>(expected_face_values),
        get<DivergenceCleaningField>(expected_face_values));

    tmpl::for_each<tmpl::append<cons_tags, prims_tags>>(
        [dim, &expected_face_values, &vars_on_lower_face,
         &vars_on_upper_face](auto tag_to_check_v) {
          using tag_to_check = tmpl::type_from<decltype(tag_to_check_v)>;
          CAPTURE(db::tag_name<tag_to_check>());
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(gsl::at(vars_on_lower_face, dim)),
              get<tag_to_check>(expected_face_values));
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(gsl::at(vars_on_upper_face, dim)),
              get<tag_to_check>(expected_face_values));
        });
  }
}
}  // namespace detail

template <typename Reconstructor>
void test_prim_reconstructor(const size_t points_per_dimension,
                             const Reconstructor& derived_reconstructor) {
  detail::test_prim_reconstructor_impl<>(
      points_per_dimension, derived_reconstructor,
      EquationsOfState::IdealFluid<false>{1.4});
}
}  // namespace TestHelpers::NewtonianMhd::fd
