// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

constexpr double divergence_cleaning_speed = 1.5;

auto make_coord_map() {
  const Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine3D{affine_map, affine_map, affine_map});
}

auto make_element() {
  const Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  std::vector<Block<3>> blocks;
  blocks.emplace_back(Block<3>(
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Affine3D{affine_map, affine_map, affine_map}),
      0, {}));
  return domain::create_initial_element(
      ElementId<3>{0, {SegmentId{3, 4}, SegmentId{3, 4}, SegmentId{3, 4}}},
      blocks,
      std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{3, 3, 3}}});
}

template <bool UseBackgroundMagneticField>
struct MetaVars {
  using system = NewtonianMhd::System<UseBackgroundMagneticField>;
};

// A uniform field is curl-free and divergence-free, so splitting it off leaves
// the total state, and hence the characteristic speeds, unchanged.
tnsr::I<DataVector, 3, Frame::Inertial> uniform_background_field(
    const size_t num_points) {
  tnsr::I<DataVector, 3, Frame::Inertial> field{num_points};
  get<0>(field) = 0.4;
  get<1>(field) = -0.7;
  get<2>(field) = 0.2;
  return field;
}

// Returns the packaged data on each mortar, and the evolved variables sliced to
// the corresponding face.
template <bool UseBackgroundMagneticField>
std::pair<DirectionalIdMap<3, DataVector>,
          DirectionalIdMap<
              3, Variables<typename NewtonianMhd::System<
                     UseBackgroundMagneticField>::variables_tag::tags_list>>>
compute_packaged_data(const size_t num_dg_pts) {
  using solution = NewtonianMhd::Solutions::AlfvenWave;
  using system = NewtonianMhd::System<UseBackgroundMagneticField>;
  using variables_tag = typename system::variables_tag;
  using prim_tags = typename system::primitive_variables_tag::tags_list;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;

  const auto coordinate_map = make_coord_map();
  const auto moving_mesh_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});
  const auto element = make_element();

  const solution soln{{{1.0, 1.0, 1.0}}, 1.0, 1.0, 1.0, 0.1, 5.0 / 3.0};

  const double time = 0.0;
  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto dg_coords = coordinate_map(logical_coordinates(dg_mesh));

  using prims_to_reconstruct_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::Pressure<DataVector>, MagneticField,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

  typename evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      neighbor_data{};
  for (const Direction<3>& direction : Direction<3>::all_directions()) {
    auto neighbor_logical_coords = logical_coordinates(subcell_mesh);
    neighbor_logical_coords.get(direction.dimension()) +=
        2.0 * direction.sign();
    const auto neighbor_coords = coordinate_map(neighbor_logical_coords);
    const auto neighbor_prims =
        soln.variables(neighbor_coords, time, prim_tags{});
    Variables<prims_to_reconstruct_tags> prims_to_reconstruct{
        subcell_mesh.number_of_grid_points()};
    tmpl::for_each<prims_to_reconstruct_tags>(
        [&prims_to_reconstruct, &neighbor_prims](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(prims_to_reconstruct) = get<tag>(neighbor_prims);
        });
    if constexpr (UseBackgroundMagneticField) {
      const auto background =
          uniform_background_field(subcell_mesh.number_of_grid_points());
      for (size_t i = 0; i < 3; ++i) {
        get<MagneticField>(prims_to_reconstruct).get(i) -= background.get(i);
      }
    }
    const DataVector neighbor_data_in_direction =
        evolution::dg::subcell::slice_data(
            prims_to_reconstruct, subcell_mesh.extents(),
            NewtonianMhd::fd::MonotonisedCentralPrim{}.ghost_zone_size(),
            std::unordered_set{direction.opposite()}, 0, {})
            .at(direction.opposite());
    const auto key =
        DirectionalId<3>{direction, *element.neighbors().at(direction).begin()};
    neighbor_data[key] = evolution::dg::subcell::GhostData{1};
    neighbor_data[key].neighbor_ghost_data_for_reconstruction() =
        neighbor_data_in_direction;
  }

  Variables<prim_tags> dg_prim_vars{dg_mesh.number_of_grid_points()};
  dg_prim_vars.assign_subset(soln.variables(dg_coords, time, prim_tags{}));
  if constexpr (UseBackgroundMagneticField) {
    const auto background =
        uniform_background_field(dg_mesh.number_of_grid_points());
    for (size_t i = 0; i < 3; ++i) {
      get<MagneticField>(dg_prim_vars).get(i) -= background.get(i);
    }
  }

  DirectionMap<3, std::optional<Variables<
                      tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<3>>>>>
      normal_vectors{};
  for (const auto& direction : Direction<3>::all_directions()) {
    const Mesh<3 - 1> face_mesh = dg_mesh.slice_away(direction.dimension());
    const auto face_logical_coords =
        interface_logical_coordinates(face_mesh, direction);
    std::unordered_map<Direction<3>, tnsr::i<DataVector, 3, Frame::Inertial>>
        unnormalized_normal_covectors{};
    tnsr::i<DataVector, 3, Frame::Inertial> unnormalized_covector{};
    for (size_t i = 0; i < 3; ++i) {
      unnormalized_covector.get(i) =
          coordinate_map.inv_jacobian(face_logical_coords)
              .get(direction.dimension(), i);
    }
    unnormalized_normal_covectors[direction] = unnormalized_covector;
    Variables<tmpl::list<
        evolution::dg::Actions::detail::NormalVector<3>,
        evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>>
        fields_on_face{face_mesh.number_of_grid_points()};
    normal_vectors[direction] = std::nullopt;
    evolution::dg::Actions::detail::
        unit_normal_vector_and_covector_and_magnitude<system>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face),
            direction, unnormalized_normal_covectors, moving_mesh_map);
  }

  const auto make_box = [&](auto... background_at_faces) {
    return db::create<
        db::AddSimpleTags<
            Parallel::Tags::MetavariablesImpl<
                MetaVars<UseBackgroundMagneticField>>,
            domain::Tags::Element<3>, domain::Tags::Mesh<3>,
            evolution::dg::subcell::Tags::Mesh<3>,
            NewtonianMhd::fd::Tags::Reconstructor,
            evolution::Tags::BoundaryCorrection,
            hydro::Tags::EquationOfState<false, 2>,
            typename system::primitive_variables_tag, variables_tag,
            evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
            evolution::dg::Tags::MortarData<3>, domain::Tags::MeshVelocity<3>,
            evolution::dg::Tags::NormalCovectorAndMagnitude<3>,
            evolution::dg::subcell::Tags::SubcellOptions<3>,
            NewtonianMhd::Tags::DivergenceCleaningSpeed,
            decltype(background_at_faces)...>,
        db::AddComputeTags<
            evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>>>(
        MetaVars<UseBackgroundMagneticField>{}, element, dg_mesh, subcell_mesh,
        std::unique_ptr<NewtonianMhd::fd::Reconstructor>{
            std::make_unique<NewtonianMhd::fd::MonotonisedCentralPrim>()},
        std::unique_ptr<evolution::BoundaryCorrection>{
            std::make_unique<NewtonianMhd::BoundaryCorrections::Hll<
                UseBackgroundMagneticField>>()},
        soln.equation_of_state().get_clone(), dg_prim_vars,
        typename variables_tag::type{dg_mesh.number_of_grid_points()},
        neighbor_data, typename evolution::dg::Tags::MortarData<3>::type{},
        std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>{},
        normal_vectors,
        evolution::dg::subcell::SubcellOptions{
            4.0, 1_st, 1.0e-3, 1.0e-4, false, false,
            evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
            ::fd::DerivativeOrder::Two, 1, 1, 1},
        divergence_cleaning_speed,
        typename decltype(background_at_faces)::type{
            make_array<3>(uniform_background_field(
                (subcell_mesh.extents(0) + 1) *
                subcell_mesh.extents().slice_away(0).product()))}...);
  };

  auto box = [&make_box]() {
    if constexpr (UseBackgroundMagneticField) {
      return make_box(evolution::dg::subcell::Tags::OnSubcellFaces<
                      NewtonianMhd::Tags::BackgroundMagneticField<>, 3>{});
    } else {
      return make_box();
    }
  }();

  db::mutate_apply<NewtonianMhd::ConservativeFromPrimitive>(
      make_not_null(&box));

  std::vector<DirectionalId<3>> mortars_to_reconstruct_to{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    mortars_to_reconstruct_to.emplace_back(direction, *neighbors.begin());
  }

  auto all_packaged_data = NewtonianMhd::subcell::NeighborPackagedData<
      UseBackgroundMagneticField>::apply(box, mortars_to_reconstruct_to);

  DirectionalIdMap<3, Variables<typename variables_tag::tags_list>>
      sliced_evolved_vars{};
  for (const auto& directional_id : mortars_to_reconstruct_to) {
    const auto& direction = directional_id.direction();
    sliced_evolved_vars[directional_id] = data_on_slice(
        db::get<variables_tag>(box), dg_mesh.extents(), direction.dimension(),
        direction.side() == Side::Upper
            ? dg_mesh.extents(direction.dimension()) - 1
            : 0);
  }
  return {std::move(all_packaged_data), std::move(sliced_evolved_vars)};
}

// The reconstructed interface data must approach the exact (sliced) values as
// the grid is refined.
double reconstruction_error(const size_t num_dg_pts) {
  const auto [all_packaged_data, sliced_evolved_vars] =
      compute_packaged_data<false>(num_dg_pts);
  using system = NewtonianMhd::System<false>;
  using variables_tag = typename system::variables_tag;
  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  double max_abs_error = 0.0;
  for (const auto& [directional_id, data] : all_packaged_data) {
    const auto& direction = directional_id.direction();
    using Hll = NewtonianMhd::BoundaryCorrections::Hll<false>;
    using dg_package_field_tags = typename Hll::dg_package_field_tags;
    const Mesh<3 - 1> face_mesh = dg_mesh.slice_away(direction.dimension());
    Variables<dg_package_field_tags> packaged_data{
        face_mesh.number_of_grid_points()};
    std::ranges::copy(data, packaged_data.data());

    tmpl::for_each<typename variables_tag::type::tags_list>(
        [&sliced_vars = sliced_evolved_vars.at(directional_id), &max_abs_error,
         &packaged_data](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          const auto& sliced_tensor = get<tag>(sliced_vars);
          const auto& packaged_data_tensor = get<tag>(packaged_data);
          for (size_t tensor_index = 0; tensor_index < sliced_tensor.size();
               ++tensor_index) {
            max_abs_error = std::max(
                max_abs_error, max(abs(sliced_tensor[tensor_index] -
                                       packaged_data_tensor[tensor_index])));
          }
        });
  }
  return max_abs_error;
}

// Splitting off a uniform background field leaves the total magnetic field, and
// so the characteristic speeds, unchanged: the reconstruction is exact for a
// constant, so the background read back from the face-centred grids must cancel
// the shift applied to the evolved field exactly. This is what checks that the
// background reaches the flux computation on the correct face.
void test_background_field_splitting(const size_t num_dg_pts) {
  const auto [unsplit_data, unsplit_sliced] =
      compute_packaged_data<false>(num_dg_pts);
  const auto [split_data, split_sliced] =
      compute_packaged_data<true>(num_dg_pts);
  REQUIRE(unsplit_data.size() == split_data.size());

  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  using UnsplitHll = NewtonianMhd::BoundaryCorrections::Hll<false>;
  using SplitHll = NewtonianMhd::BoundaryCorrections::Hll<true>;
  using unsplit_fields = typename UnsplitHll::dg_package_field_tags;
  using split_fields = typename SplitHll::dg_package_field_tags;

  for (const auto& [directional_id, data] : unsplit_data) {
    CAPTURE(directional_id);
    const Mesh<3 - 1> face_mesh =
        dg_mesh.slice_away(directional_id.direction().dimension());
    Variables<unsplit_fields> unsplit{face_mesh.number_of_grid_points()};
    Variables<split_fields> split{face_mesh.number_of_grid_points()};
    std::ranges::copy(data, unsplit.data());
    const auto& split_raw = split_data.at(directional_id);
    std::ranges::copy(split_raw, split.data());

    CHECK_ITERABLE_APPROX(
        get<typename UnsplitHll::LargestOutgoingCharSpeed>(unsplit),
        get<typename SplitHll::LargestOutgoingCharSpeed>(split));
    CHECK_ITERABLE_APPROX(
        get<typename UnsplitHll::LargestIngoingCharSpeed>(unsplit),
        get<typename SplitHll::LargestIngoingCharSpeed>(split));

    // The evolved magnetic field differs by exactly the background.
    const auto background =
        uniform_background_field(face_mesh.number_of_grid_points());
    const auto& unsplit_b =
        get<NewtonianMhd::Tags::MagneticFieldCons<>>(unsplit);
    const auto& split_b = get<NewtonianMhd::Tags::MagneticFieldCons<>>(split);
    for (size_t i = 0; i < 3; ++i) {
      const DataVector expected = unsplit_b.get(i) - background.get(i);
      CHECK_ITERABLE_APPROX(split_b.get(i), expected);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianMhd.Subcell.NeighborPackagedData",
    "[Unit][Evolution]") {
  // Sets up a cube [2,3]^3 holding an Alfven wave and checks that the
  // difference between the reconstructed evolved variables and the sliced
  // (exact on the LGL grid) evolved variables on the interfaces decreases with
  // resolution.
  CHECK(reconstruction_error(3) > reconstruction_error(6));
  CHECK(reconstruction_error(6) < 5.0e-2);

  test_background_field_splitting(4);
}
