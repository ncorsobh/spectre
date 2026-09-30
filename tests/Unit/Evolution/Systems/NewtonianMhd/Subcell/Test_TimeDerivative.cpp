// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
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
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/AoWeno.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/Source.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

constexpr size_t Dim = 3;
constexpr double divergence_cleaning_speed = 1.5;
constexpr double constraint_damping_parameter = 0.3;
constexpr double adiabatic_index = 5.0 / 3.0;

auto make_grid_map() {
  const Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  return domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
      Affine3D{affine_map, affine_map, affine_map});
}

auto make_element() {
  const Affine affine_map{-1.0, 1.0, 2.0, 3.0};
  std::vector<Block<Dim>> blocks;
  blocks.emplace_back(Block<Dim>(
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Affine3D{affine_map, affine_map, affine_map}),
      0, {}));
  return domain::create_initial_element(
      ElementId<Dim>{0, {SegmentId{2, 2}, SegmentId{2, 2}, SegmentId{2, 2}}},
      blocks,
      std::vector<std::array<size_t, Dim>>{std::array<size_t, Dim>{{3, 3, 3}}});
}

template <bool UseBackgroundMagneticField>
struct MetaVars {
  static constexpr size_t volume_dim = Dim;
  using system = NewtonianMhd::System<Dim, UseBackgroundMagneticField>;
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = false;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        evolution::BoundaryCorrection,
        NewtonianMhd::BoundaryCorrections::standard_boundary_corrections<
            Dim, UseBackgroundMagneticField>>>;
  };
};

tnsr::I<DataVector, Dim, Frame::Inertial> uniform_background_field(
    const size_t num_points) {
  tnsr::I<DataVector, Dim, Frame::Inertial> field{num_points};
  get<0>(field) = 0.4;
  get<1>(field) = -0.7;
  get<2>(field) = 0.2;
  return field;
}

using prim_tags_list =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::SpatialVelocity<DataVector, Dim>,
               hydro::Tags::SpecificInternalEnergy<DataVector>,
               hydro::Tags::Pressure<DataVector>,
               hydro::Tags::MagneticField<DataVector, Dim>,
               hydro::Tags::DivergenceCleaningField<DataVector>>;

using prims_to_reconstruct_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::SpatialVelocity<DataVector, Dim>,
               hydro::Tags::Pressure<DataVector>,
               hydro::Tags::MagneticField<DataVector, Dim>,
               hydro::Tags::DivergenceCleaningField<DataVector>>;

// Evaluates the time derivative on the finite-difference grid for the state
// produced by `set_prims(coords, vars)`. With the splitting enabled the
// caller's magnetic field is the total one, and the uniform background is
// subtracted here so that both settings describe the same physical state.
template <bool UseBackgroundMagneticField, typename SetPrims>
Variables<typename db::add_tag_prefix<
    ::Tags::dt, typename NewtonianMhd::System<
                    Dim, UseBackgroundMagneticField>::variables_tag>::tags_list>
compute_time_derivative(const size_t num_dg_pts, const SetPrims& set_prims) {
  using system = NewtonianMhd::System<Dim, UseBackgroundMagneticField>;
  using variables_tag = typename system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, Dim>;

  const auto element = make_element();
  const ElementMap<Dim, Frame::Grid> element_map{element.id(), make_grid_map()};
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{});

  const Mesh<Dim> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t num_subcell_pts = subcell_mesh.number_of_grid_points();

  const auto prims_at = [&set_prims, &element_map,
                         &grid_to_inertial_map](const auto& logical_coords) {
    const auto coords =
        (*grid_to_inertial_map)(element_map(logical_coords), 0.0, {});
    Variables<prim_tags_list> vars{get<0>(coords).size()};
    set_prims(coords, make_not_null(&vars));
    if constexpr (UseBackgroundMagneticField) {
      const auto background = uniform_background_field(get<0>(coords).size());
      for (size_t i = 0; i < Dim; ++i) {
        get<MagneticField>(vars).get(i) -= background.get(i);
      }
    }
    return vars;
  };

  Variables<prim_tags_list> cell_centered_prim_vars =
      prims_at(logical_coordinates(subcell_mesh));

  typename evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>::type
      neighbor_data{};
  for (const Direction<Dim>& direction : Direction<Dim>::all_directions()) {
    auto neighbor_logical_coords = logical_coordinates(subcell_mesh);
    neighbor_logical_coords.get(direction.dimension()) +=
        2.0 * direction.sign();
    const auto neighbor_prims = prims_at(neighbor_logical_coords);
    Variables<prims_to_reconstruct_tags> prims_to_reconstruct{num_subcell_pts};
    tmpl::for_each<prims_to_reconstruct_tags>(
        [&prims_to_reconstruct, &neighbor_prims](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(prims_to_reconstruct) = get<tag>(neighbor_prims);
        });
    const DataVector neighbor_data_in_direction =
        evolution::dg::subcell::slice_data(
            prims_to_reconstruct, subcell_mesh.extents(),
            NewtonianMhd::fd::MonotonisedCentralPrim<Dim>{}.ghost_zone_size(),
            std::unordered_set{direction.opposite()}, 0, {})
            .at(direction.opposite());
    const auto key = DirectionalId<Dim>{
        direction, *element.neighbors().at(direction).begin()};
    neighbor_data[key] = evolution::dg::subcell::GhostData{1};
    neighbor_data[key].neighbor_ghost_data_for_reconstruction() =
        neighbor_data_in_direction;
  }

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      dummy_functions_of_time{};

  const auto make_box = [&](auto... background_tags) {
    return db::create<
        db::AddSimpleTags<
            Parallel::Tags::MetavariablesImpl<
                MetaVars<UseBackgroundMagneticField>>,
            NewtonianMhd::Tags::SourceTerm<Dim, UseBackgroundMagneticField>,
            domain::Tags::Element<Dim>,
            domain::Tags::ElementMap<Dim, Frame::Grid>,
            evolution::dg::subcell::Tags::Mesh<Dim>,
            NewtonianMhd::fd::Tags::Reconstructor<Dim>,
            evolution::Tags::BoundaryCorrection,
            hydro::Tags::EquationOfState<false, 2>,
            typename system::primitive_variables_tag, dt_variables_tag,
            variables_tag,
            evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
            evolution::dg::Tags::MortarData<Dim>,
            domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                        Frame::Inertial>,
            ::Tags::Time, domain::Tags::FunctionsOfTimeInitialize,
            NewtonianMhd::Tags::DivergenceCleaningSpeed,
            NewtonianMhd::Tags::ConstraintDampingParameter,
            decltype(background_tags)...>,
        db::AddComputeTags<
            evolution::dg::subcell::Tags::LogicalCoordinatesCompute<Dim>,
            ::domain::Tags::MappedCoordinates<
                ::domain::Tags::ElementMap<Dim, Frame::Grid>,
                evolution::dg::subcell::Tags::Coordinates<
                    Dim, Frame::ElementLogical>,
                evolution::dg::subcell::Tags::Coordinates>,
            evolution::dg::subcell::Tags::InertialCoordinatesCompute<
                ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                              Frame::Inertial>>,
            evolution::dg::subcell::fd::Tags::
                InverseJacobianLogicalToGridCompute<
                    ::domain::Tags::ElementMap<Dim, Frame::Grid>, Dim>,
            evolution::dg::subcell::fd::Tags::
                DetInverseJacobianLogicalToGridCompute<Dim>,
            evolution::dg::subcell::fd::Tags::
                InverseJacobianLogicalToInertialCompute<
                    ::domain::CoordinateMaps::Tags::CoordinateMap<
                        Dim, Frame::Grid, Frame::Inertial>,
                    Dim>,
            evolution::dg::subcell::fd::Tags::
                DetInverseJacobianLogicalToInertialCompute<
                    ::domain::CoordinateMaps::Tags::CoordinateMap<
                        Dim, Frame::Grid, Frame::Inertial>,
                    Dim>>>(
        MetaVars<UseBackgroundMagneticField>{},
        std::unique_ptr<
            NewtonianMhd::Sources::Source<Dim, UseBackgroundMagneticField>>{
            std::make_unique<NewtonianMhd::Sources::NoSource<
                Dim, UseBackgroundMagneticField>>()},
        element, ElementMap<Dim, Frame::Grid>{element.id(), make_grid_map()},
        subcell_mesh,
        std::unique_ptr<NewtonianMhd::fd::Reconstructor<Dim>>{
            std::make_unique<NewtonianMhd::fd::MonotonisedCentralPrim<Dim>>()},
        std::unique_ptr<evolution::BoundaryCorrection>{
            std::make_unique<NewtonianMhd::BoundaryCorrections::Hll<
                Dim, UseBackgroundMagneticField>>()},
        EquationsOfState::IdealFluid<false>{adiabatic_index}
            .promote_to_2d_eos(),
        cell_centered_prim_vars,
        Variables<typename dt_variables_tag::tags_list>{num_subcell_pts},
        typename variables_tag::type{}, neighbor_data,
        typename evolution::dg::Tags::MortarData<Dim>::type{},
        grid_to_inertial_map->get_clone(), 0.0,
        clone_unique_ptrs(dummy_functions_of_time), divergence_cleaning_speed,
        constraint_damping_parameter,
        typename decltype(background_tags)::type{}...);
  };

  auto box = [&make_box]() {
    if constexpr (UseBackgroundMagneticField) {
      return make_box(
          NewtonianMhd::Tags::BackgroundMagneticFieldVolume<Dim>{},
          evolution::dg::subcell::Tags::OnSubcellFaces<
              NewtonianMhd::Tags::BackgroundMagneticField<Dim>, Dim>{});
    } else {
      return make_box();
    }
  }();

  if constexpr (UseBackgroundMagneticField) {
    db::mutate<NewtonianMhd::Tags::BackgroundMagneticFieldVolume<Dim>,
               evolution::dg::subcell::Tags::OnSubcellFaces<
                   NewtonianMhd::Tags::BackgroundMagneticField<Dim>, Dim>>(
        [&num_subcell_pts, &subcell_mesh](const auto volume_ptr,
                                          const auto faces_ptr) {
          *volume_ptr = uniform_background_field(num_subcell_pts);
          const size_t num_face_pts =
              (subcell_mesh.extents(0) + 1) *
              subcell_mesh.extents().slice_away(0).product();
          *faces_ptr = make_array<Dim>(uniform_background_field(num_face_pts));
        },
        make_not_null(&box));
  }

  db::mutate_apply<NewtonianMhd::ConservativeFromPrimitive<Dim>>(
      make_not_null(&box));
  NewtonianMhd::subcell::TimeDerivative<Dim>::apply(make_not_null(&box));
  return db::get<dt_variables_tag>(box);
}

// A state with no gradients and no flow is an exact equilibrium, so every time
// derivative but the constraint damping must vanish.
template <bool UseBackgroundMagneticField>
void test_uniform_state(const double divergence_cleaning_field) {
  CAPTURE(UseBackgroundMagneticField);
  CAPTURE(divergence_cleaning_field);
  const auto dt_vars = compute_time_derivative<UseBackgroundMagneticField>(
      4, [&divergence_cleaning_field](const auto& coords, const auto vars) {
        const size_t num_points = get<0>(coords).size();
        get(get<hydro::Tags::RestMassDensity<DataVector>>(*vars)) = 1.2;
        get(get<hydro::Tags::Pressure<DataVector>>(*vars)) = 0.8;
        get(get<hydro::Tags::SpecificInternalEnergy<DataVector>>(*vars)) =
            0.8 / (1.2 * (adiabatic_index - 1.0));
        for (size_t i = 0; i < Dim; ++i) {
          get<hydro::Tags::SpatialVelocity<DataVector, Dim>>(*vars).get(i) =
              DataVector{num_points, 0.0};
        }
        get<0>(get<hydro::Tags::MagneticField<DataVector, Dim>>(*vars)) = 0.9;
        get<1>(get<hydro::Tags::MagneticField<DataVector, Dim>>(*vars)) = -0.3;
        get<2>(get<hydro::Tags::MagneticField<DataVector, Dim>>(*vars)) = 0.5;
        get(get<hydro::Tags::DivergenceCleaningField<DataVector>>(*vars)) =
            divergence_cleaning_field;
      });

  const DataVector zero{
      get(get<::Tags::dt<NewtonianMhd::Tags::MassDensityCons>>(dt_vars)).size(),
      0.0};
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::dt<NewtonianMhd::Tags::MassDensityCons>>(dt_vars)), zero);
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::dt<NewtonianMhd::Tags::EnergyDensity>>(dt_vars)), zero);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(
        get<::Tags::dt<NewtonianMhd::Tags::MomentumDensity<Dim>>>(dt_vars).get(
            i),
        zero);
    CHECK_ITERABLE_APPROX(
        get<::Tags::dt<NewtonianMhd::Tags::MagneticFieldCons<Dim>>>(dt_vars)
            .get(i),
        zero);
  }
  // The only non-zero term is the GLM constraint damping.
  const DataVector expected_dt_psi = zero - constraint_damping_parameter *
                                                divergence_cleaning_speed *
                                                divergence_cleaning_field;
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::dt<NewtonianMhd::Tags::DivergenceCleaningFieldCons>>(
          dt_vars)),
      expected_dt_psi);
}

// Splitting off a uniform background field must not change the physics. The
// evolved magnetic field is shifted by a constant, so its time derivative, and
// those of the mass and momentum densities, are unchanged. The evolved energy
// density differs from the total one by B0.B1, so its time derivative differs
// by B0.dt(B1).
template <typename SetPrims>
void test_background_field_splitting(const SetPrims& set_prims) {
  const auto unsplit = compute_time_derivative<false>(4, set_prims);
  const auto split = compute_time_derivative<true>(4, set_prims);

  // The two formulations group the same sums differently, so they agree only to
  // the cancellation error between the background and the perturbation.
  Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<::Tags::dt<NewtonianMhd::Tags::MassDensityCons>>(unsplit)),
      get(get<::Tags::dt<NewtonianMhd::Tags::MassDensityCons>>(split)),
      custom_approx);
  for (size_t i = 0; i < Dim; ++i) {
    CAPTURE(i);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<::Tags::dt<NewtonianMhd::Tags::MomentumDensity<Dim>>>(unsplit).get(
            i),
        get<::Tags::dt<NewtonianMhd::Tags::MomentumDensity<Dim>>>(split).get(i),
        custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<::Tags::dt<NewtonianMhd::Tags::MagneticFieldCons<Dim>>>(unsplit)
            .get(i),
        get<::Tags::dt<NewtonianMhd::Tags::MagneticFieldCons<Dim>>>(split).get(
            i),
        custom_approx);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<::Tags::dt<NewtonianMhd::Tags::DivergenceCleaningFieldCons>>(
          unsplit)),
      get(get<::Tags::dt<NewtonianMhd::Tags::DivergenceCleaningFieldCons>>(
          split)),
      custom_approx);

  const auto& dt_split_b =
      get<::Tags::dt<NewtonianMhd::Tags::MagneticFieldCons<Dim>>>(split);
  const auto background = uniform_background_field(get<0>(dt_split_b).size());
  DataVector expected_dt_energy =
      get(get<::Tags::dt<NewtonianMhd::Tags::EnergyDensity>>(unsplit));
  for (size_t i = 0; i < Dim; ++i) {
    expected_dt_energy -= background.get(i) * dt_split_b.get(i);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<::Tags::dt<NewtonianMhd::Tags::EnergyDensity>>(split)),
      expected_dt_energy, custom_approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianMhd.Subcell.TimeDerivative",
                  "[Unit][Evolution]") {
  test_uniform_state<false>(0.0);
  test_uniform_state<false>(0.7);
  test_uniform_state<true>(0.0);
  test_uniform_state<true>(0.7);

  test_background_field_splitting([](const auto& coords, const auto vars) {
    const NewtonianMhd::Solutions::AlfvenWave soln{
        {{1.0, 1.0, 1.0}}, 1.0, 1.0, 1.0, 0.1, adiabatic_index};
    vars->assign_subset(soln.variables(coords, 0.0, prim_tags_list{}));
  });

  // The GLM cleaning field enters the split energy flux through -B0.psi, a
  // term that only shows up where psi has a gradient: both the Alfven wave
  // above and the uniform state carry a psi whose gradient vanishes.
  test_background_field_splitting([](const auto& coords, const auto vars) {
    const NewtonianMhd::Solutions::AlfvenWave soln{
        {{1.0, 1.0, 1.0}}, 1.0, 1.0, 1.0, 0.1, adiabatic_index};
    vars->assign_subset(soln.variables(coords, 0.0, prim_tags_list{}));
    get(get<hydro::Tags::DivergenceCleaningField<DataVector>>(*vars)) =
        0.3 * get<0>(coords) - 0.2 * get<1>(coords) + 0.45 * get<2>(coords);
  });
}
