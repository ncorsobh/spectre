// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain {
namespace {

void test_1d_domains() {
  using Translation = domain::CoordinateMaps::TimeDependent::Translation<1>;
  using TranslationGridDistorted =
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, Translation>;
  using TranslationDistortedInertial =
      domain::CoordinateMap<Frame::Distorted, Frame::Inertial, Translation>;
  {
    using LogicalToGridCoordinateMap =
        CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                      CoordinateMaps::Identity<1>>;

    using GridToInertialCoordinateMap =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, Translation>;
    using GridToDistortedCoordinateMap = TranslationGridDistorted;
    using DistortedToInertialCoordinateMap = TranslationDistortedInertial;

    PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::BlockLogical, Frame::Grid,
                                         CoordinateMaps::Identity<1>>));
    PUPable_reg(GridToInertialCoordinateMap);

    PUPable_reg(LogicalToGridCoordinateMap);
    PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::BlockLogical, Frame::Grid,
                                         CoordinateMaps::Identity<1>>));
    PUPable_reg(GridToDistortedCoordinateMap);
    PUPable_reg(DistortedToInertialCoordinateMap);

    PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                         CoordinateMaps::Affine>));
    PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::BlockLogical, Frame::Grid,
                                         CoordinateMaps::Affine>));
    PUPable_reg(
        SINGLE_ARG(CoordinateMap<Frame::Grid, Frame::Inertial, Translation>));

    // Test construction of two intervals which have anti-aligned logical axes.
    Domain<1> domain_from_corners(
        make_vector<std::unique_ptr<
            CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 1>>>(
            std::make_unique<CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                           CoordinateMaps::Affine>>(
                make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
                    CoordinateMaps::Affine{-1., 1., -2., 0.})),
            std::make_unique<CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                           CoordinateMaps::Affine>>(
                make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
                    CoordinateMaps::Affine{-1., 1., 0., 2.}))),
        std::vector<std::array<size_t, 2>>{{{1, 2}}, {{3, 2}}}, {}, {},
        {"Left", "Right"}, {{"All", {"Left", "Right"}}});
    CHECK(domain_from_corners.blocks()[0].name() == "Left");
    CHECK(domain_from_corners.blocks()[1].name() == "Right");
    CHECK(domain_from_corners.block_groups().at("All") ==
          std::unordered_set<std::string>{"Left", "Right"});

    Domain<1> domain_no_corners(
        make_vector<std::unique_ptr<
            CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 1>>>(
            std::make_unique<CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                           CoordinateMaps::Affine>>(
                make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
                    CoordinateMaps::Affine{-1., 1., -2., 0.})),
            std::make_unique<CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                           CoordinateMaps::Affine>>(
                make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
                    CoordinateMaps::Affine{-1., 1., 2., 0.}))),
        {{"ExcisionSphere",
          ExcisionSphere<1>{1.0, tnsr::I<double, 1, Frame::Grid>{0.0}, {}}}},
        {"Left", "Right"}, {{"All", {"Left", "Right"}}});
    CHECK_FALSE(domain_no_corners.is_time_dependent());
    CHECK(domain_no_corners.blocks()[0].name() == "Left");
    CHECK(domain_no_corners.blocks()[1].name() == "Right");
    CHECK(domain_no_corners.block_groups().at("All") ==
          std::unordered_set<std::string>{"Left", "Right"});

    test_serialization(domain_no_corners);

    const OrientationMap<1> unaligned_orientation{{{Direction<1>::lower_xi()}},
                                                  {{Direction<1>::upper_xi()}}};

    const std::vector<DirectionMap<1, BlockNeighbor<1>>> expected_neighbors{
        {{Direction<1>::upper_xi(),
          BlockNeighbor<1>{1, unaligned_orientation}}},
        {{Direction<1>::upper_xi(),
          BlockNeighbor<1>{0, unaligned_orientation}}}};

    const std::vector<std::unordered_set<Direction<1>>> expected_boundaries{
        {Direction<1>::lower_xi()}, {Direction<1>::lower_xi()}};

    const auto expected_stationary_maps = make_vector(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::Affine{-1., 1., -2., 0.}),
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::Affine{-1., 1., 0., 2.}));

    const auto expected_stationary_maps_no_corners = make_vector(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::Affine{-1., 1., -2., 0.}),
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::Affine{-1., 1., 2., 0.}));

    const GridToDistortedCoordinateMap translation_grid_to_distorted_map =
        domain::make_coordinate_map<Frame::Grid, Frame::Distorted>(
            Translation{"TranslationGridToDistorted"});
    const DistortedToInertialCoordinateMap
        translation_distorted_to_inertial_map =
            domain::make_coordinate_map<Frame::Distorted, Frame::Inertial>(
                Translation{"TranslationDistortedToInertial"});

    test_domain_construction(domain_from_corners, expected_neighbors,
                             expected_boundaries, expected_stationary_maps);

    test_domain_construction(serialize_and_deserialize(domain_from_corners),
                             expected_neighbors, expected_boundaries,
                             expected_stationary_maps);

    test_domain_construction(domain_no_corners, expected_neighbors,
                             expected_boundaries,
                             expected_stationary_maps_no_corners);

    test_domain_construction(serialize_and_deserialize(domain_no_corners),
                             expected_neighbors, expected_boundaries,
                             expected_stationary_maps_no_corners);

    // Test injection of a translation map.
    REQUIRE(domain_from_corners.blocks().size() == 2);
    REQUIRE(domain_no_corners.blocks().size() == 2);
    domain_from_corners.inject_time_dependent_map_for_block(
        0,
        make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation{"Translation0"}),
        translation_grid_to_distorted_map.get_clone(),
        translation_distorted_to_inertial_map.get_clone());
    domain_from_corners.inject_time_dependent_map_for_block(
        1,
        make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation{"Translation1"}),
        translation_grid_to_distorted_map.get_clone(),
        translation_distorted_to_inertial_map.get_clone());

    domain_no_corners.inject_time_dependent_map_for_block(
        0,
        make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation{"Translation0"}),
        translation_grid_to_distorted_map.get_clone(),
        translation_distorted_to_inertial_map.get_clone());
    domain_no_corners.inject_time_dependent_map_for_block(
        1,
        make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation{"Translation1"}),
        translation_grid_to_distorted_map.get_clone(),
        translation_distorted_to_inertial_map.get_clone());
    domain_no_corners.inject_time_dependent_map_for_excision_sphere(
        "ExcisionSphere",
        make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation{"Translation0"}));
#ifdef SPECTRE_DEBUG
    CHECK_THROWS_WITH(
        domain_no_corners.inject_time_dependent_map_for_excision_sphere(
            "NonExistentExcisionSphere",
            make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
                Translation{"Translation0"})),
        Catch::Matchers::ContainsSubstring(
            "Cannot inject time dependent maps into excision "
            "sphere 'NonExistentExcisionSphere'"));
#endif
    CHECK(domain_no_corners.is_time_dependent());

    // Excision spheres
    const auto& excision_spheres_corners =
        domain_from_corners.excision_spheres();
    CHECK(excision_spheres_corners.empty());
    const auto& excision_spheres_no_corners =
        domain_no_corners.excision_spheres();
    CHECK(excision_spheres_no_corners.size() == 1);
    CHECK(excision_spheres_no_corners.count("ExcisionSphere") == 1);
    CHECK(excision_spheres_no_corners.at("ExcisionSphere").is_time_dependent());

    const auto expected_logical_to_grid_maps =
        make_vector(make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                        CoordinateMaps::Affine{-1., 1., -2., 0.}),
                    make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                        CoordinateMaps::Affine{-1., 1., 0., 2.}));
    const auto expected_logical_to_grid_maps_no_corners =
        make_vector(make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                        CoordinateMaps::Affine{-1., 1., -2., 0.}),
                    make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                        CoordinateMaps::Affine{-1., 1., 2., 0.}));
    const auto expected_grid_to_inertial_maps =
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation{"Translation0"}, Translation{"Translation1"});

    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};
    functions_of_time["Translation0"] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            1.0, std::array<DataVector, 3>{{{0.0}, {2.3}, {0.0}}}, 10.0);
    functions_of_time["Translation1"] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            1.0, std::array<DataVector, 3>{{{0.0}, {5.3}, {0.0}}}, 10.0);

    test_domain_construction(domain_from_corners, expected_neighbors,
                             expected_boundaries, expected_logical_to_grid_maps,
                             10.0, functions_of_time,
                             expected_grid_to_inertial_maps);
    test_domain_construction(serialize_and_deserialize(domain_from_corners),
                             expected_neighbors, expected_boundaries,
                             expected_logical_to_grid_maps, 10.0,
                             functions_of_time, expected_grid_to_inertial_maps);

    test_domain_construction(domain_no_corners, expected_neighbors,
                             expected_boundaries,
                             expected_logical_to_grid_maps_no_corners, 10.0,
                             functions_of_time, expected_grid_to_inertial_maps);
    test_domain_construction(serialize_and_deserialize(domain_no_corners),
                             expected_neighbors, expected_boundaries,
                             expected_logical_to_grid_maps_no_corners, 10.0,
                             functions_of_time, expected_grid_to_inertial_maps);

    // Test construction from a vector of blocks
    auto vector_of_blocks = [&expected_neighbors]() {
      std::vector<Block<1>> vec;
      vec.emplace_back(Block<1>{
          std::make_unique<CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                         CoordinateMaps::Affine>>(
              make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
                  CoordinateMaps::Affine{-1., 1., -2., 0.})),
          0, expected_neighbors[0]});
      vec.emplace_back(Block<1>{
          std::make_unique<CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                         CoordinateMaps::Affine>>(
              make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
                  CoordinateMaps::Affine{-1., 1., 0., 2.})),
          1, expected_neighbors[1]});
      return vec;
    }();

    test_domain_construction(Domain<1>{std::move(vector_of_blocks)},
                             expected_neighbors, expected_boundaries,
                             expected_stationary_maps);

    CHECK(get_output(domain_from_corners) ==
          "Domain with 2 blocks:\n" +
              get_output(domain_from_corners.blocks()[0]) + "\n" +
              get_output(domain_from_corners.blocks()[1]) + "\n" +
              "Excision spheres:\n" +
              get_output(domain_from_corners.excision_spheres()) + "\n");

    CHECK(get_output(domain_no_corners) ==
          "Domain with 2 blocks:\n" +
              get_output(domain_no_corners.blocks()[0]) + "\n" +
              get_output(domain_no_corners.blocks()[1]) + "\n" +
              "Excision spheres:\n" +
              get_output(domain_no_corners.excision_spheres()) + "\n");
  }

  {
    // Test construction of a periodic domain
    const auto expected_maps = make_vector(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::Affine{-1., 1., -2., 2.}));

    const Domain<1> domain{
        make_vector<std::unique_ptr<
            CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 1>>>(
            std::make_unique<CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                           CoordinateMaps::Affine>>(
                make_coordinate_map<Frame::BlockLogical, Frame::Inertial>(
                    CoordinateMaps::Affine{-1., 1., -2., 2.}))),
        std::vector<std::array<size_t, 2>>{{{1, 2}}},
        std::vector<PairOfFaces>{{{1}, {2}}}};

    test_serialization(domain);

    const auto expected_neighbors = []() {
      OrientationMap<1> orientation{{{Direction<1>::lower_xi()}},
                                    {{Direction<1>::lower_xi()}}};
      return std::vector<DirectionMap<1, BlockNeighbor<1>>>{
          {{Direction<1>::lower_xi(), BlockNeighbor<1>{0, orientation}},
           {Direction<1>::upper_xi(), BlockNeighbor<1>{0, orientation}}}};
    }();

    test_domain_construction(domain, expected_neighbors,
                             std::vector<std::unordered_set<Direction<1>>>{1},
                             expected_maps);
  }
}

void test_1d_rectilinear_domains() {
  {
    INFO("Aligned domain.");
    const std::vector<std::unordered_set<Direction<1>>>
        expected_external_boundaries{
            {{Direction<1>::lower_xi()}}, {}, {{Direction<1>::upper_xi()}}};
    const auto domain = rectilinear_domain<1>(
        Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}},
        {}, {}, {{false}}, {}, true);
    const OrientationMap<1> aligned{};
    std::vector<DirectionMap<1, BlockNeighbor<1>>> expected_block_neighbors{
        {{Direction<1>::upper_xi(), {1, aligned}}},
        {{Direction<1>::lower_xi(), {0, aligned}},
         {Direction<1>::upper_xi(), {2, aligned}}},
        {{Direction<1>::lower_xi(), {1, aligned}}}};
    for (size_t i = 0; i < domain.blocks().size(); i++) {
      CAPTURE(i);
      CHECK(domain.blocks()[i].external_boundaries() ==
            expected_external_boundaries[i]);
      CHECK(domain.blocks()[i].neighbors() == expected_block_neighbors[i]);
    }
  }
  {
    INFO("Antialigned domain.");
    const OrientationMap<1> aligned{};
    const OrientationMap<1> antialigned{
        std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}}};
    const std::vector<std::unordered_set<Direction<1>>>
        expected_external_boundaries{
            {{Direction<1>::lower_xi()}}, {}, {{Direction<1>::upper_xi()}}};

    const auto domain = rectilinear_domain<1>(
        Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}},
        {}, std::vector<OrientationMap<1>>{aligned, antialigned, aligned},
        {{false}}, {}, true);
    std::vector<DirectionMap<1, BlockNeighbor<1>>> expected_block_neighbors{
        {{Direction<1>::upper_xi(), {1, antialigned}}},
        {{Direction<1>::lower_xi(), {2, antialigned}},
         {Direction<1>::upper_xi(), {0, antialigned}}},
        {{Direction<1>::lower_xi(), {1, antialigned}}}};
    for (size_t i = 0; i < domain.blocks().size(); i++) {
      INFO(i);
      CHECK(domain.blocks()[i].external_boundaries() ==
            expected_external_boundaries[i]);
      CHECK(domain.blocks()[i].neighbors() == expected_block_neighbors[i]);
    }
  }
}

void test_2d_rectilinear_domains() {
  const OrientationMap<2> half_turn{std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}};
  const OrientationMap<2> quarter_turn_cw{std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}};
  const OrientationMap<2> quarter_turn_ccw{std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}};
  auto orientations_of_all_blocks =
      std::vector<OrientationMap<2>>{4, OrientationMap<2>{}};
  orientations_of_all_blocks[1] = half_turn;
  orientations_of_all_blocks[2] = quarter_turn_cw;
  orientations_of_all_blocks[3] = quarter_turn_ccw;

  const std::vector<std::unordered_set<Direction<2>>>
      expected_external_boundaries{
          {{Direction<2>::lower_xi(), Direction<2>::lower_eta()}},
          {{Direction<2>::upper_eta(), Direction<2>::lower_xi()}},
          {{Direction<2>::lower_xi(), Direction<2>::lower_eta()}},
          {{Direction<2>::upper_xi(), Direction<2>::lower_eta()}}};

  const auto domain = rectilinear_domain<2>(
      Index<2>{2, 2},
      std::array<std::vector<double>, 2>{{{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}},
      {}, orientations_of_all_blocks);
  std::vector<DirectionMap<2, BlockNeighbor<2>>> expected_block_neighbors{
      {{Direction<2>::upper_xi(), {1, half_turn}},
       {Direction<2>::upper_eta(), {2, quarter_turn_cw}}},
      {{Direction<2>::upper_xi(), {0, half_turn}},
       {Direction<2>::lower_eta(), {3, quarter_turn_cw}}},
      {{Direction<2>::upper_xi(), {0, quarter_turn_ccw}},
       {Direction<2>::upper_eta(), {3, half_turn}}},
      {{Direction<2>::lower_xi(), {1, quarter_turn_ccw}},
       {Direction<2>::upper_eta(), {2, half_turn}}}};
  for (size_t i = 0; i < domain.blocks().size(); i++) {
    INFO(i);
    CHECK(domain.blocks()[i].external_boundaries() ==
          expected_external_boundaries[i]);
    CHECK(domain.blocks()[i].neighbors() == expected_block_neighbors[i]);
  }
}

void test_3d_rectilinear_domains() {
  const OrientationMap<3> aligned{};
  const OrientationMap<3> quarter_turn_cw_xi{std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
       Direction<3>::lower_eta()}}};
  auto orientations_of_all_blocks =
      std::vector<OrientationMap<3>>{aligned, quarter_turn_cw_xi};

  const std::vector<std::unordered_set<Direction<3>>>
      expected_external_boundaries{
          {{Direction<3>::lower_xi(), Direction<3>::lower_eta(),
            Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
            Direction<3>::upper_zeta()}},
          {{Direction<3>::upper_xi(), Direction<3>::lower_eta(),
            Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
            Direction<3>::upper_zeta()}}};

  const auto domain =
      rectilinear_domain<3>(Index<3>{2, 1, 1},
                            std::array<std::vector<double>, 3>{
                                {{0.0, 1.0, 2.0}, {0.0, 1.0}, {0.0, 1.0}}},
                            {}, orientations_of_all_blocks);
  std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{
      {{Direction<3>::upper_xi(), {1, quarter_turn_cw_xi}}},
      {{Direction<3>::lower_xi(), {0, quarter_turn_cw_xi.inverse_map()}}}};
  for (size_t i = 0; i < domain.blocks().size(); i++) {
    INFO(i);
    CHECK(domain.blocks()[i].external_boundaries() ==
          expected_external_boundaries[i]);
    CHECK(domain.blocks()[i].neighbors() == expected_block_neighbors[i]);
  }
}

// We can't call the DomainCreator because they aren't serialized and can change
// at any time. What we actually want to test is that we can deserialize a
// domain and have it be exactly what we expect. Not whatever the latest version
// of the DomainCreator makes. For this, we copy a lot of code from
// BinaryCompactObject because the binary domain is complicated (which is a good
// test of versioning because of so many moving parts)
Domain<3> create_serialized_domain() {
  std::vector<std::string> block_names{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups{};

  static std::array<std::string, 6> wedge_directions{
      "UpperZ", "LowerZ", "UpperY", "LowerY", "UpperX", "LowerX"};
  const auto add_object_region = [&block_names, &block_groups](
                                     const std::string& object_name,
                                     const std::string& region_name) {
    for (const std::string& wedge_direction : wedge_directions) {
      const std::string block_name =
          // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
          object_name + region_name + wedge_direction;
      block_names.push_back(block_name);
      block_groups[object_name + region_name].insert(block_name);
    }
  };
  const auto add_outer_region =
      [&block_names, &block_groups](const std::string& region_name) {
        for (const std::string& wedge_direction : wedge_directions) {
          for (const std::string& leftright : {"Left"s, "Right"s}) {
            if ((wedge_direction == "UpperX" and leftright == "Left") or
                (wedge_direction == "LowerX" and leftright == "Right")) {
              // The outer regions are divided in half perpendicular to the
              // x-axis at x=0. Therefore, the left side only has a block in
              // negative x-direction, and the right side only has one in
              // positive x-direction.
              continue;
            }
            // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
            const std::string block_name =
                region_name + wedge_direction +
                (wedge_direction == "UpperX" or wedge_direction == "LowerX"
                     ? ""
                     : leftright);
            block_names.push_back(block_name);
            block_groups[region_name].insert(block_name);
          }
        }
      };
  add_object_region("ObjectA", "Shell");  // 6 blocks
  add_object_region("ObjectA", "Cube");   // 6 blocks
  add_object_region("ObjectB", "Shell");  // 6 blocks
  add_object_region("ObjectB", "Cube");   // 6 blocks
  add_outer_region("Envelope");           // 10 blocks
  add_outer_region("OuterShell");         // 10 blocks

  // Expand initial refinement and number of grid points over all blocks
  const ExpandOverBlocks<std::array<size_t, 3>> expand_over_blocks{
      block_names, block_groups};

  using BCO = domain::creators::BinaryCompactObject;

  const BCO::InitialRefinement::type initial_refinement_variant{1_st};
  const BCO::InitialGridPoints::type initial_grid_points_variant{3_st};
  std::vector<std::array<size_t, 3>> initial_refinement =
      std::visit(expand_over_blocks, initial_refinement_variant);
  std::vector<std::array<size_t, 3>> initial_grid_points =
      std::visit(expand_over_blocks, initial_grid_points_variant);

  const std::vector<domain::CoordinateMaps::Distribution>
      object_A_radial_distribution{
          domain::CoordinateMaps::Distribution::Logarithmic};
  const std::vector<domain::CoordinateMaps::Distribution>
      object_B_radial_distribution = object_A_radial_distribution;
  const domain::CoordinateMaps::Distribution radial_distribution_envelope =
      domain::CoordinateMaps::Distribution::Linear;
  const domain::CoordinateMaps::Distribution radial_distribution_outer_shell =
      domain::CoordinateMaps::Distribution::Linear;

  using Maps = std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>;
  using Affine = CoordinateMaps::Affine;
  using Identity2D = CoordinateMaps::Identity<2>;
  using Translation = CoordinateMaps::ProductOf2Maps<Affine, Identity2D>;

  domain::creators::BinaryCompactObject::Object object_A{0.45825, 6., 7.683,
                                                         true, true};
  domain::creators::BinaryCompactObject::Object object_B{0.45825, 6., -7.683,
                                                         true, true};

  const double x_coord_a = object_A.x_coord;
  const double x_coord_b = object_B.x_coord;
  const double envelope_radius = 100.0;
  const double outer_radius = 300.0;
  const double inner_sphericity_A = 1.0;
  const double inner_sphericity_B = 1.0;
  const bool use_equiangular_map = true;
  const double translation = 0.5 * (x_coord_a + x_coord_b);
  const double length_inner_cube = abs(x_coord_a - x_coord_b);
  const double opening_angle = M_PI / 2.0;
  const double tan_half_opening_angle = tan(0.5 * opening_angle);
  const double length_outer_cube =
      2.0 * envelope_radius / sqrt(2.0 + square(tan_half_opening_angle));

  Maps maps{};

  const Translation translation_A{
      Affine{-1.0, 1.0, -1.0 + x_coord_a, 1.0 + x_coord_a}, Identity2D{}};
  const Translation translation_B{
      Affine{-1.0, 1.0, -1.0 + x_coord_b, 1.0 + x_coord_b}, Identity2D{}};

  Maps maps_center_A =
      domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                              Frame::Inertial, 3>(
          sph_wedge_coordinate_maps(object_A.inner_radius,
                                    object_A.outer_radius, inner_sphericity_A,
                                    1.0, use_equiangular_map, false, {},
                                    object_A_radial_distribution),
          translation_A);
  Maps maps_cube_A =
      domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                              Frame::Inertial, 3>(
          sph_wedge_coordinate_maps(object_A.outer_radius,
                                    sqrt(3.0) * 0.5 * length_inner_cube, 1.0,
                                    0.0, use_equiangular_map),
          translation_A);
  std::move(maps_center_A.begin(), maps_center_A.end(),
            std::back_inserter(maps));
  std::move(maps_cube_A.begin(), maps_cube_A.end(), std::back_inserter(maps));

  Maps maps_center_B =
      domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                              Frame::Inertial, 3>(
          sph_wedge_coordinate_maps(object_B.inner_radius,
                                    object_B.outer_radius, inner_sphericity_B,
                                    1.0, use_equiangular_map, false, {},
                                    object_B_radial_distribution),
          translation_B);
  Maps maps_cube_B =
      domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                              Frame::Inertial, 3>(
          sph_wedge_coordinate_maps(object_B.outer_radius,
                                    sqrt(3.0) * 0.5 * length_inner_cube, 1.0,
                                    0.0, use_equiangular_map),
          translation_B);
  std::move(maps_center_B.begin(), maps_center_B.end(),
            std::back_inserter(maps));
  std::move(maps_cube_B.begin(), maps_cube_B.end(), std::back_inserter(maps));

  Maps maps_frustums = domain::make_vector_coordinate_map_base<
      Frame::BlockLogical, Frame::Inertial, 3>(frustum_coordinate_maps(
      length_inner_cube, length_outer_cube, use_equiangular_map,
      {{-translation, 0.0, 0.0}}, radial_distribution_envelope,
      radial_distribution_envelope ==
              domain::CoordinateMaps::Distribution::Projective
          ? std::optional<double>(length_inner_cube / length_outer_cube)
          : std::nullopt,
      1.0, opening_angle));
  std::move(maps_frustums.begin(), maps_frustums.end(),
            std::back_inserter(maps));

  Maps maps_outer_shell = domain::make_vector_coordinate_map_base<
      Frame::BlockLogical, Frame::Inertial, 3>(sph_wedge_coordinate_maps(
      envelope_radius, outer_radius, 1.0, 1.0, use_equiangular_map, true, {},
      {radial_distribution_outer_shell}, ShellWedges::All, opening_angle));
  std::move(maps_outer_shell.begin(), maps_outer_shell.end(),
            std::back_inserter(maps));

  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};
  excision_spheres.emplace(
      "ExcisionSphereA",
      ExcisionSphere<3>{object_A.inner_radius,
                        tnsr::I<double, 3, Frame::Grid>{{x_coord_a, 0.0, 0.0}},
                        {{0, Direction<3>::lower_zeta()},
                         {1, Direction<3>::lower_zeta()},
                         {2, Direction<3>::lower_zeta()},
                         {3, Direction<3>::lower_zeta()},
                         {4, Direction<3>::lower_zeta()},
                         {5, Direction<3>::lower_zeta()}}});
  excision_spheres.emplace(
      "ExcisionSphereB",
      ExcisionSphere<3>{object_B.inner_radius,
                        tnsr::I<double, 3, Frame::Grid>{{x_coord_b, 0.0, 0.0}},
                        {{12, Direction<3>::lower_zeta()},
                         {13, Direction<3>::lower_zeta()},
                         {14, Direction<3>::lower_zeta()},
                         {15, Direction<3>::lower_zeta()},
                         {16, Direction<3>::lower_zeta()},
                         {17, Direction<3>::lower_zeta()}}});

  Domain<3> domain{std::move(maps), std::move(excision_spheres), block_names,
                   block_groups};

  // We have IsCylindrical = true even though this is the rectangular domain
  // because the cylindrical domain uses the SphereTransition transition
  // function for the shape map which is what is serialized. The rectangular
  // domain uses the Wedge transition function.
  using TimeDepOps = domain::creators::bco::TimeDependentMapOptions<true>;
  TimeDepOps time_dependent_options{
      0.,
      TimeDepOps::ExpansionMapOptions{
          {{1.0, -4.6148457646200002e-05}}, -1.0e-6, 50.},
      TimeDepOps::RotationMapOptions{{0.0, 0.0, 1.5264577062000000e-02}},
      TimeDepOps::ShapeMapOptions<domain::ObjectLabel::A>{8, {0., 0., 0.}},
      TimeDepOps::ShapeMapOptions<domain::ObjectLabel::B>{8, {0., 0., 0.}}};

  const std::optional<std::array<double, 2>> inner_outer_radii_A =
      std::array{object_A.inner_radius, object_A.outer_radius};
  const std::optional<std::array<double, 2>> inner_outer_radii_B =
      std::array{object_B.inner_radius, object_B.outer_radius};
  const std::array<std::array<double, 3>, 2> centers{
      std::array{x_coord_a, 0.0, 0.0}, std::array{x_coord_b, 0.0, 0.0}};

  time_dependent_options.build_maps(centers, inner_outer_radii_A,
                                    inner_outer_radii_B, outer_radius);

  const size_t number_of_blocks = 44;

  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
      grid_to_inertial_block_maps{number_of_blocks};
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
      grid_to_distorted_block_maps{number_of_blocks};
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
      distorted_to_inertial_block_maps{number_of_blocks};

  grid_to_inertial_block_maps[number_of_blocks - 1] =
      time_dependent_options.grid_to_inertial_map<domain::ObjectLabel::None>(
          false);

  grid_to_inertial_block_maps[0] =
      time_dependent_options.grid_to_inertial_map<domain::ObjectLabel::A>(true);
  grid_to_distorted_block_maps[0] =
      time_dependent_options.grid_to_distorted_map<domain::ObjectLabel::A>(
          true);
  distorted_to_inertial_block_maps[0] =
      time_dependent_options.distorted_to_inertial_map<domain::ObjectLabel::A>(
          true);

  const size_t first_block_object_B = 12;
  grid_to_inertial_block_maps[first_block_object_B] =
      time_dependent_options.grid_to_inertial_map<domain::ObjectLabel::B>(true);
  grid_to_distorted_block_maps[first_block_object_B] =
      time_dependent_options.grid_to_distorted_map<domain::ObjectLabel::B>(
          true);
  distorted_to_inertial_block_maps[first_block_object_B] =
      time_dependent_options.distorted_to_inertial_map<domain::ObjectLabel::B>(
          true);

  for (size_t block = 1; block < number_of_blocks - 1; ++block) {
    if (block < 6) {
      // We always have a grid to inertial map. We may or may not have maps to
      // the distorted frame.
      grid_to_inertial_block_maps[block] =
          grid_to_inertial_block_maps[0]->get_clone();
      if (grid_to_distorted_block_maps[0] != nullptr) {
        grid_to_distorted_block_maps[block] =
            grid_to_distorted_block_maps[0]->get_clone();
        distorted_to_inertial_block_maps[block] =
            distorted_to_inertial_block_maps[0]->get_clone();
      }
    } else if (block == first_block_object_B) {
      continue;  // already initialized
    } else if (block > first_block_object_B and
               block < first_block_object_B + 6) {
      // We always have a grid to inertial map. We may or may not have maps to
      // the distorted frame.
      grid_to_inertial_block_maps[block] =
          grid_to_inertial_block_maps[first_block_object_B]->get_clone();
      if (grid_to_distorted_block_maps[first_block_object_B] != nullptr) {
        grid_to_distorted_block_maps[block] =
            grid_to_distorted_block_maps[first_block_object_B]->get_clone();
        distorted_to_inertial_block_maps[block] =
            distorted_to_inertial_block_maps[first_block_object_B]->get_clone();
      }
    } else {
      grid_to_inertial_block_maps[block] =
          grid_to_inertial_block_maps[number_of_blocks - 1]->get_clone();
    }
  }

  for (size_t block = 0; block < number_of_blocks; ++block) {
    domain.inject_time_dependent_map_for_block(
        block, std::move(grid_to_inertial_block_maps[block]),
        std::move(grid_to_distorted_block_maps[block]),
        std::move(distorted_to_inertial_block_maps[block]));
  }

  return domain;
}

void test_versioning() {
  // Check that we can deserialize the domain stored in this old file
  domain::creators::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  h5::H5File<h5::AccessType::ReadOnly> h5file{unit_test_src_path() +
                                              "/Domain/SerializedDomain.h5"};
  const auto& volfile = h5file.get<h5::VolumeData>("/element_data");
  const size_t obs_id = volfile.list_observation_ids().front();
  const auto serialized_domain = *volfile.get_domain(obs_id);
  const auto domain = deserialize<Domain<3>>(serialized_domain.data());
  const Domain<3> expected_domain = create_serialized_domain();
  CHECK(domain == expected_domain);
  // Also check that we can deserialize the functions of time.
  const auto serialized_fot = *volfile.get_functions_of_time(obs_id);
  const auto functions_of_time = deserialize<std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>>(
      serialized_fot.data());
  CHECK(functions_of_time.count("Rotation") == 1);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Domain", "[Domain][Unit]") {
  {
    INFO("Equality operator");
    Domain<1> lhs{
        make_vector(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::Affine{-1., 1., -2., 0.})),
        {},
        {"Block0"},
        {{"All", {"Block0"}}}};
    {
      Domain<1> rhs{
          make_vector(
              make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                  CoordinateMaps::Affine{-1., 1., -2., 0.})),
          {},
          {"Block1"},
          {{"All", {"Block0"}}}};
      CHECK_FALSE(lhs == rhs);
    }
    {
      Domain<1> rhs{
          make_vector(
              make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                  CoordinateMaps::Affine{-1., 1., -2., 0.})),
          {},
          {"Block0"},
          {}};
      CHECK_FALSE(lhs == rhs);
    }
  }

  test_1d_domains();
  test_1d_rectilinear_domains();
  test_2d_rectilinear_domains();
  test_3d_rectilinear_domains();
  test_versioning();

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      Domain<1>(
          make_vector(
              make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                  CoordinateMaps::Affine{-1., 1., -1., 1.}),
              make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                  CoordinateMaps::Affine{-1., 1., -1., 1.})),
          std::vector<std::array<size_t, 2>>{{{1, 2}}}),
      Catch::Matchers::ContainsSubstring(
          "Must pass same number of maps as block corner sets"));
#endif
}
}  // namespace domain
