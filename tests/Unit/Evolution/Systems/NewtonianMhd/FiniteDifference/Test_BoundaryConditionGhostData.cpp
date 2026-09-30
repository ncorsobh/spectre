// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/ConductorReflection.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/Reflection.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr double adiabatic_index = 5.0 / 3.0;

// The interior state is uniform, so the expected ghost values are the same
// constants in every ghost cell and each boundary condition can be checked
// directly rather than against a second implementation.
constexpr double interior_density = 1.3;
constexpr double interior_pressure = 0.8;
constexpr std::array<double, 3> interior_velocity{{0.2, -0.3, 0.4}};
// Fast speed of the state above is ~1.24, so this is comfortably supersonic.
constexpr double supersonic_speed = 5.0;
constexpr std::array<double, 3> interior_magnetic_field{{0.5, 0.25, -0.6}};
constexpr double interior_divergence_cleaning_field = 0.35;

struct EvolutionMetaVars {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<NewtonianMhd::BoundaryConditions::BoundaryCondition,
                   NewtonianMhd::BoundaryConditions::
                       standard_boundary_conditions<false>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::list<NewtonianMhd::Solutions::AlfvenWave>>>;
  };
};

using prim_tags = tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                             hydro::Tags::SpatialVelocity<DataVector, 3>,
                             hydro::Tags::SpecificInternalEnergy<DataVector>,
                             hydro::Tags::Pressure<DataVector>,
                             hydro::Tags::MagneticField<DataVector, 3>,
                             hydro::Tags::DivergenceCleaningField<DataVector>>;

using recons_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::SpatialVelocity<DataVector, 3>,
               hydro::Tags::Pressure<DataVector>,
               hydro::Tags::MagneticField<DataVector, 3>,
               hydro::Tags::DivergenceCleaningField<DataVector>>;

using ReconstructorForTest = NewtonianMhd::fd::MonotonisedCentralPrim;

using MassDensityTag = hydro::Tags::RestMassDensity<DataVector>;
using VelocityTag = hydro::Tags::SpatialVelocity<DataVector, 3>;
using PressureTag = hydro::Tags::Pressure<DataVector>;
using MagneticFieldTag = hydro::Tags::MagneticField<DataVector, 3>;
using DivergenceCleaningFieldTag =
    hydro::Tags::DivergenceCleaningField<DataVector>;

template <typename BoundaryConditionType>
void test(const BoundaryConditionType& boundary_condition,
          const Direction<3>& direction,
          const std::array<double, 3>& velocity = interior_velocity) {
  CAPTURE(direction);
  const size_t num_dg_pts = 3;

  // Only the face under test gets `boundary_condition`; the rest reflect.
  // `BoundaryConditionGhostData` fills every external face, and a uniform
  // state cannot be outgoing through all six of them at once.
  std::array<
      std::array<std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
                 2>,
      3>
      face_conditions{};
  for (size_t d = 0; d < 3; ++d) {
    for (size_t side = 0; side < 2; ++side) {
      gsl::at(gsl::at(face_conditions, d), side) =
          (d == direction.dimension() and
           side == (direction.side() == Side::Upper ? 1 : 0))
              ? boundary_condition.get_clone()
              : NewtonianMhd::BoundaryConditions::Reflection<false>{}
                    .get_clone();
    }
  }
  const auto brick = domain::creators::Brick(
      std::array<double, 3>{{-1.0, -1.0, -1.0}},
      std::array<double, 3>{{1.0, 1.0, 1.0}}, std::array<size_t, 3>{{0, 0, 0}},
      std::array<size_t, 3>{{num_dg_pts, num_dg_pts, num_dg_pts}},
      std::move(face_conditions));
  auto domain = brick.create_domain();
  auto boundary_conditions = brick.external_boundary_conditions();
  const auto element = domain::create_initial_element(
      ElementId<3>{0, {SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}},
      domain.blocks(), std::vector<std::array<size_t, 3>>{{{0, 0, 0}}});

  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  std::unique_ptr<EquationsOfState::EquationOfState<false, 2>> eos =
      std::make_unique<EquationsOfState::IdealFluid<false>>(adiabatic_index);

  Variables<prim_tags> volume_prims{subcell_mesh.number_of_grid_points()};
  get(get<hydro::Tags::RestMassDensity<DataVector>>(volume_prims)) =
      interior_density;
  get(get<hydro::Tags::Pressure<DataVector>>(volume_prims)) = interior_pressure;
  get(get<hydro::Tags::DivergenceCleaningField<DataVector>>(volume_prims)) =
      interior_divergence_cleaning_field;
  for (size_t i = 0; i < 3; ++i) {
    get<hydro::Tags::SpatialVelocity<DataVector, 3>>(volume_prims).get(i) =
        gsl::at(velocity, i);
    get<hydro::Tags::MagneticField<DataVector, 3>>(volume_prims).get(i) =
        gsl::at(interior_magnetic_field, i);
  }
  get<hydro::Tags::SpecificInternalEnergy<DataVector>>(volume_prims) =
      eos->specific_internal_energy_from_density_and_pressure(
          get<hydro::Tags::RestMassDensity<DataVector>>(volume_prims),
          get<hydro::Tags::Pressure<DataVector>>(volume_prims));

  const double time = 0.0;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  const ElementMap<3, Frame::Grid> logical_to_grid_map(
      ElementId<3>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<3>{}));
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});

  const std::optional<tnsr::I<DataVector, 3>> volume_mesh_velocity{};
  typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type
      normal_vectors{};
  for (const auto& dir : Direction<3>::all_directions()) {
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<3>{});
    const auto moving_mesh_map =
        domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Identity<3>{});
    const Mesh<3 - 1> face_mesh = subcell_mesh.slice_away(dir.dimension());
    const auto face_logical_coords =
        interface_logical_coordinates(face_mesh, dir);
    std::unordered_map<Direction<3>, tnsr::i<DataVector, 3, Frame::Inertial>>
        unnormalized_normal_covectors{};
    tnsr::i<DataVector, 3, Frame::Inertial> unnormalized_covector{
        face_mesh.number_of_grid_points()};
    for (size_t i = 0; i < 3; ++i) {
      unnormalized_covector.get(i) =
          dir.sign() * coordinate_map.inv_jacobian(face_logical_coords)
                           .get(dir.dimension(), i);
    }
    unnormalized_normal_covectors[dir] = unnormalized_covector;
    Variables<tmpl::list<
        evolution::dg::Actions::detail::NormalVector<3>,
        evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>>
        fields_on_face{face_mesh.number_of_grid_points()};
    normal_vectors[dir] = std::nullopt;
    evolution::dg::Actions::detail::
        unit_normal_vector_and_covector_and_magnitude<
            NewtonianMhd::System<false>>(
            make_not_null(&normal_vectors), make_not_null(&fields_on_face), dir,
            unnormalized_normal_covectors, moving_mesh_map);
  }

  typename evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      ghost_data{};

  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<EvolutionMetaVars>,
      domain::Tags::Domain<3>, domain::Tags::ExternalBoundaryConditions<3>,
      evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::ElementLogical>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
      NewtonianMhd::fd::Tags::Reconstructor, domain::Tags::MeshVelocity<3>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>, ::Tags::Time,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::ElementMap<3, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                  Frame::Inertial>,
      hydro::Tags::EquationOfState<false, 2>, ::Tags::Variables<prim_tags>>>(
      EvolutionMetaVars{}, std::move(domain), std::move(boundary_conditions),
      subcell_mesh, logical_coordinates(subcell_mesh), ghost_data,
      std::unique_ptr<NewtonianMhd::fd::Reconstructor>{
          std::make_unique<ReconstructorForTest>()},
      volume_mesh_velocity, normal_vectors, time,
      clone_unique_ptrs(functions_of_time),
      ElementMap<3, Frame::Grid>{
          ElementId<3>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<3>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}),
      std::move(eos), volume_prims);

  NewtonianMhd::fd::BoundaryConditionGhostData::apply(
      make_not_null(&box), element, ReconstructorForTest{});

  const DirectionalId<3> mortar_id{direction,
                                   ElementId<3>::external_boundary_id()};
  const DataVector& fd_ghost_data =
      get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box)
          .at(mortar_id)
          .neighbor_ghost_data_for_reconstruction();

  Variables<recons_tags> ghost_vars{
      const_cast<double*>(fd_ghost_data.data()),  // NOLINT
      fd_ghost_data.size()};
  const size_t num_ghost_pts = get(get<MassDensityTag>(ghost_vars)).size();
  const DataVector ones{num_ghost_pts, 1.0};

  const bool is_reflection =
      typeid(BoundaryConditionType) ==
          typeid(NewtonianMhd::BoundaryConditions::Reflection<false>) or
      typeid(BoundaryConditionType) ==
          typeid(NewtonianMhd::BoundaryConditions::ConductorReflection<false>);
  const bool no_slip =
      typeid(BoundaryConditionType) ==
      typeid(NewtonianMhd::BoundaryConditions::ConductorReflection<false>);

  if (is_reflection or
      typeid(BoundaryConditionType) ==
          typeid(NewtonianMhd::BoundaryConditions::DemandOutgoingCharSpeeds<
                 false>)) {
    // Density and pressure are copied by every one of these conditions.
    CHECK_ITERABLE_APPROX(get(get<MassDensityTag>(ghost_vars)),
                          interior_density * ones);
    CHECK_ITERABLE_APPROX(get(get<PressureTag>(ghost_vars)),
                          interior_pressure * ones);
    CHECK_ITERABLE_APPROX(get(get<DivergenceCleaningFieldTag>(ghost_vars)),
                          (is_reflection ? -1.0 : 1.0) *
                              interior_divergence_cleaning_field * ones);
    for (size_t i = 0; i < 3; ++i) {
      CAPTURE(i);
      const double velocity_sign =
          is_reflection and (no_slip or i == direction.dimension()) ? -1.0
                                                                    : 1.0;
      const double field_sign =
          is_reflection and i == direction.dimension() ? -1.0 : 1.0;
      CHECK_ITERABLE_APPROX(get<VelocityTag>(ghost_vars).get(i),
                            velocity_sign * gsl::at(velocity, i) * ones);
      CHECK_ITERABLE_APPROX(
          get<MagneticFieldTag>(ghost_vars).get(i),
          field_sign * gsl::at(interior_magnetic_field, i) * ones);
    }
  } else {
    // DirichletAnalytic: the ghost zone holds the solution evaluated on the
    // ghost-cell coordinates.
    const auto ghost_logical_coords =
        evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
            subcell_mesh, ReconstructorForTest{}.ghost_zone_size(), direction);
    const auto ghost_inertial_coords = (*grid_to_inertial_map)(
        logical_to_grid_map(ghost_logical_coords), time, functions_of_time);
    const NewtonianMhd::Solutions::AlfvenWave solution{
        {{1.0, 1.0, 1.0}}, 1.0, 1.0, 1.0, 0.1, adiabatic_index};
    const auto expected =
        solution.variables(ghost_inertial_coords, time, recons_tags{});
    tmpl::for_each<recons_tags>([&expected, &ghost_vars](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      const auto& expected_tensor = get<tag>(expected);
      const auto& actual_tensor = get<tag>(ghost_vars);
      for (size_t component = 0; component < expected_tensor.size();
           ++component) {
        CAPTURE(component);
        CHECK_ITERABLE_APPROX(expected_tensor[component],
                              actual_tensor[component]);
      }
    });
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianMhd.Fd.BoundaryConditionGhostData",
    "[Unit][Evolution]") {
  for (const auto& direction : Direction<3>::all_directions()) {
    test(NewtonianMhd::BoundaryConditions::Reflection<false>{}, direction);
    test(NewtonianMhd::BoundaryConditions::ConductorReflection<false>{},
         direction);
    // The condition errors unless every characteristic leaves the domain, so
    // this one gets a supersonic outflow along the boundary normal.
    std::array<double, 3> outflow_velocity{{0.0, 0.0, 0.0}};
    gsl::at(outflow_velocity, direction.dimension()) =
        direction.sign() * supersonic_speed;
    test(NewtonianMhd::BoundaryConditions::DemandOutgoingCharSpeeds<false>{},
         direction, outflow_velocity);
    test(
        NewtonianMhd::BoundaryConditions::DirichletAnalytic<false>{
            std::make_unique<NewtonianMhd::Solutions::AlfvenWave>(
                std::array<double, 3>{{1.0, 1.0, 1.0}}, 1.0, 1.0, 1.0, 0.1,
                adiabatic_index)},
        direction);
  }
}
