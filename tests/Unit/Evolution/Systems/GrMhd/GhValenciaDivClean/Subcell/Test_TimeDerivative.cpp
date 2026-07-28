// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/GhostZoneInverseJacobian.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/GhostZoneInverseJacobian.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/AnalyticChristoffel.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/FilterOptions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhRelativisticEuler/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace grmhd::GhValenciaDivClean {
namespace {

// Selects which analytic solution the test<> template instantiates. We
// deliberately test each subset of evolved variables against the solution best
// suited for it:
//
//  - BondiMichel: an accreting perfect-fluid + monopole magnetic field solution
//    that satisfies the fluid equations on a fixed Schwarzschild Kerr-Schild
//    background. Because it does *not* satisfy Einstein's equations (there is
//    no back-reaction of the matter on the metric), the GH matter source term
//    -16pi alpha S_ab in dt(Pi) does not cancel and dt(Pi) does not converge to
//    zero. However, the *fluid* variables (TildeD, TildeYe, TildeTau, TildeS,
//    TildeB) all satisfy dt=0 analytically on the fixed background, and
//    BondiMichel has non-trivial fluid velocity and non-zero magnetic field so
//    the flux divergence is a real FD signal for every GRMHD variable. This
//    makes it the right solution for exercising higher-order GRMHD FD.
//
//  - TovStar: a static, spherically-symmetric fluid star that is a
//    self-consistent solution of the full Einstein+matter equations. All dt
//    values are analytically zero, including dt(Pi). Because the fluid is
//    static the GRMHD fluxes involving velocity vanish identically, so TOV is
//    useless for exercising GRMHD FD, but it is exactly what we need to verify
//    that the GH time derivative converges to zero.
enum class SolutionKind { BondiMichel, TovStar };

// Error-norm tag lists. test<> is templated on one of these so that the
// per-tag residual computation is only done for the variables that are
// meaningful for the chosen solution.
using GrmhdErrorTags = tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                                  grmhd::ValenciaDivClean::Tags::TildeYe,
                                  grmhd::ValenciaDivClean::Tags::TildeTau,
                                  grmhd::ValenciaDivClean::Tags::TildeS<>,
                                  grmhd::ValenciaDivClean::Tags::TildeB<>>;
// Only gh::Tags::Pi is exercised: on a stationary TovStar background both
// dt(SpacetimeMetric) = -alpha*Pi + shift^i*Phi_i and dt(Phi_iab) =
// d_i(dt(g_ab)) evaluate to zero arithmetically (respectively to roundoff at
// ~1e-13), so their residuals cannot decrease with resolution and would only
// pollute the check.
using GhErrorTags = tmpl::list<gh::Tags::Pi<DataVector, 3, Frame::Inertial>>;

template <typename System>
struct DummyEvolutionMetaVars {
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = true;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<BoundaryConditions::BoundaryCondition,
                   tmpl::push_back<
                       BoundaryConditions::standard_boundary_conditions<System>,
                       BoundaryConditions::DirichletAnalytic<System>>>,
        tmpl::pair<evolution::BoundaryCorrection,
                   BoundaryCorrections::standard_boundary_corrections>,
        tmpl::pair<evolution::initial_data::InitialData,
                   ghmhd::GhValenciaDivClean::InitialData::
                       analytic_solutions_and_data_list>>;
  };
};

template <bool aligned>
using CoordinateMap = tmpl::conditional_t<
    aligned,
    domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
                                           domain::CoordinateMaps::Affine,
                                           domain::CoordinateMaps::Affine>,
    domain::CoordinateMaps::Frustum>;

// Metadata for each SolutionKind: the WrappedGr type wrapping the underlying
// GR solution.
template <SolutionKind Kind>
struct SolutionTraits;
template <>
struct SolutionTraits<SolutionKind::BondiMichel> {
  using type = gh::Solutions::WrappedGr<grmhd::Solutions::BondiMichel>;
};
template <>
struct SolutionTraits<SolutionKind::TovStar> {
  using type =
      gh::Solutions::WrappedGr<::RelativisticEuler::Solutions::TovStar>;
};

template <SolutionKind Kind>
typename SolutionTraits<Kind>::type make_solution() {
  if constexpr (Kind == SolutionKind::BondiMichel) {
    // (mass, sonic_radius, sonic_density, polytropic_exponent, B0). Matches
    // the ValenciaDivClean sibling test so the FD signal on the fluid
    // variables is the same order of magnitude.
    return typename SolutionTraits<Kind>::type{1.0, 5.0, 0.05, 1.4, 2.0};
  } else {
    // Standard develop-test params. TOV radius R ~ pi*sqrt(K/(2pi)) ~= 12.5.
    return typename SolutionTraits<Kind>::type{
        1.28e-3,
        std::make_unique<EquationsOfState::PolytropicFluid<true>>(100.0, 2.0)
            ->get_clone(),
        RelativisticEuler::Solutions::TovCoordinates::Schwarzschild};
  }
}

template <typename System, bool aligned_coordinates, SolutionKind Kind,
          typename ErrorTagsList>
std::array<double, tmpl::size<ErrorTagsList>::value> test(
    const size_t num_dg_pts, const ::fd::DerivativeOrder fd_derivative_order,
    std::optional<double> expansion_velocity) {
  static constexpr bool computing_grmhd_errors =
      std::is_same_v<ErrorTagsList, GrmhdErrorTags>;
  static_assert(
      computing_grmhd_errors or std::is_same_v<ErrorTagsList, GhErrorTags>,
      "ErrorTagsList must be either GrmhdErrorTags or GhErrorTags");
  CoordinateMap<aligned_coordinates> coordinate_map;
  ElementId<3> element_id{};
  // Each solution gets its own element location so the DG boundary correction
  // and the FD ghost data can evaluate the analytic solution at every relevant
  // grid point without hitting a singularity or the exterior of a compact
  // support. Segment indices are chosen so the element is fully interior to
  // the (single) block, so no external boundary conditions are exercised.
  if constexpr (aligned_coordinates) {
    using Affine = domain::CoordinateMaps::Affine;
    using Affine3D =
        domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
    if constexpr (Kind == SolutionKind::BondiMichel) {
      // Cube [1, 15]^3 with an interior element [4.5, 6.25]^3 (r in
      // [~7.79, ~10.83]) placed far from the Schwarzschild singularity at
      // r=0 and well outside the horizon r_H = 2M = 2. Matches the sibling
      // ValenciaDivClean test.
      const Affine affine_map{-1.0, 1.0, 1.0, 15.0};
      coordinate_map = Affine3D{affine_map, affine_map, affine_map};
      element_id =
          ElementId<3>{0, {SegmentId{3, 2}, SegmentId{3, 2}, SegmentId{3, 2}}};
    } else {
      // TovStar with rho_c=1.28e-3, K=100, gamma=2 has an areal radius
      // R ~ pi*sqrt(K/(2pi)) ~= 12.5. Placing the element inside the cube
      // [-4, 4]^3 at segment {3,4}^3 (physical coords [0, 1]^3) keeps every
      // subcell and every DG-face point well inside the star (max corner
      // radius sqrt(3) ~= 1.73 << R).
      const Affine affine_map{-1.0, 1.0, -4.0, 4.0};
      coordinate_map = Affine3D{affine_map, affine_map, affine_map};
      element_id =
          ElementId<3>{0, {SegmentId{3, 4}, SegmentId{3, 4}, SegmentId{3, 4}}};
    }
  } else {
    // The Frustum sweep is only used with BondiMichel (see the
    // SPECTRE_TEST_CASE body): TovStar convergence is only exercised on the
    // aligned map.
    static_assert(Kind == SolutionKind::BondiMichel,
                  "Non-aligned coordinates only supported with BondiMichel.");
    const std::array<std::array<double, 2>, 4> face_vertices{
        {{{-5., -5.}}, {{5., 5.}}, {{-3., -3.}}, {{3., 3.}}}};
    coordinate_map = domain::CoordinateMaps::Frustum(
        face_vertices, -2., 4., OrientationMap<3>::create_aligned());
    element_id =
        ElementId<3>{0, {SegmentId{3, 2}, SegmentId{3, 2}, SegmentId{3, 2}}};
  }

  const auto soln = make_solution<Kind>();
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_boundary_conditions(1);
  for (const auto& direction : Direction<3>::all_directions()) {
    external_boundary_conditions.at(0)[direction] =
        grmhd::GhValenciaDivClean::BoundaryConditions::DirichletAnalytic<
            System>(std::make_unique<typename SolutionTraits<Kind>::type>(soln))
            .get_clone();
  }
  std::vector<Block<3>> blocks;
  blocks.emplace_back(Block<3>{
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          coordinate_map),
      0,
      {}});
  const auto& block = blocks[0];

  // alpha_9=4.0 gives ghost_zone_size=5, supporting fd_do up to Ten.
  // num_dg_pts must be >= 6 so the KO filter (order 10, stencil 11)
  // fits within the subcell grid (2*6-1=11 >= 10 pts required).
  const grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
      System>
      recons{
          3.8, 4.0, 4.0,
          ::fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
          ::VariableFixing::FixReconstructedStateToAtmosphere::Never,
          // reconstruct_rho_times_temperature=false: the DirichletAnalytic
          // boundary condition otherwise applies the atmosphere fixer to
          // reconstructed ghost data (needed by the TovStar path where the
          // solution returns rho=0 in the exterior), and the test intentionally
          // uses a default-constructed FixToAtmosphere whose thresholds are
          // signaling NaN.
          false};
  REQUIRE((static_cast<int>(fd_derivative_order) < 0 or
           (static_cast<size_t>(fd_derivative_order) / 2 <=
            recons.ghost_zone_size())));

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  ElementMap<3, Frame::Grid> element_map{
      element_id, block.is_time_dependent()
                      ? block.moving_mesh_logical_to_grid_map().get_clone()
                      : block.stationary_map().get_to_grid_frame()};
  const auto grid_to_inertial_map =
      ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          ::domain::CoordinateMaps::Identity<3>{});
  const auto element = domain::create_initial_element(
      element_id, blocks,
      std::vector<std::array<size_t, 3>>{std::array<size_t, 3>{{3, 3, 3}}});

  const double time = 0.5;
  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t num_dg_pts_3d = num_dg_pts * num_dg_pts * num_dg_pts;

  const auto logical_coords = logical_coordinates(subcell_mesh);
  const auto cell_centered_coords = (*grid_to_inertial_map)(
      element_map(logical_coords), time, functions_of_time);
  const auto dg_coords = (*grid_to_inertial_map)(
      element_map(logical_coordinates(dg_mesh)), time, functions_of_time);

  typename evolution::dg::subcell::Tags::GhostZoneInverseJacobian<3>::type
      ghost_zone_inv_jac{};

  evolution::dg::subcell::GhostZoneInverseJacobian<
      3, grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
             System>>::apply(make_not_null(&ghost_zone_inv_jac), subcell_mesh,
                             element_map, recons);

  const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>
      cell_centered_logical_to_grid_inv_jacobian =
          element_map.inv_jacobian(logical_coords);
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian =
          InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>(
              subcell_mesh.number_of_grid_points());
  const auto& cell_centered_grid_to_inertial_inv_jacobian =
      grid_to_inertial_map->inv_jacobian(element_map(logical_coords));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      auto& inv_jacobian_component =
          cell_centered_logical_to_inertial_inv_jacobian.get(i, j);
      inv_jacobian_component = 0.;
      for (size_t k = 0; k < 3; k++) {
        inv_jacobian_component +=
            cell_centered_logical_to_grid_inv_jacobian.get(i, k) *
            cell_centered_grid_to_inertial_inv_jacobian.get(k, j);
      }
    }
  }

  const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>
      dg_logical_to_grid_inv_jacobian =
          element_map.inv_jacobian(logical_coordinates(dg_mesh));
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      dg_logical_to_inertial_inv_jacobian =
          InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>(dg_mesh.number_of_grid_points());
  const auto& dg_grid_to_inertial_inv_jacobian =
      grid_to_inertial_map->inv_jacobian(
          element_map(logical_coordinates(dg_mesh)));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      auto& inv_jacobian_component =
          dg_logical_to_inertial_inv_jacobian.get(i, j);
      inv_jacobian_component = 0.;
      for (size_t k = 0; k < 3; k++) {
        inv_jacobian_component += dg_logical_to_grid_inv_jacobian.get(i, k) *
                                  dg_grid_to_inertial_inv_jacobian.get(k, j);
      }
    }
  }

  using variables_tag = typename System::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using evolved_tags = typename System::variables_tag::tags_list;
  using conserved_tags =
      typename grmhd::ValenciaDivClean::ConservativeFromPrimitive::return_tags;
  Variables<typename System::spacetime_variables_tag::tags_list>
      cell_centered_spacetime_vars{subcell_mesh.number_of_grid_points()};
  cell_centered_spacetime_vars.assign_subset(
      soln.variables(cell_centered_coords, time,
                     typename System::spacetime_variables_tag::tags_list{}));
  Variables<typename System::primitive_variables_tag::tags_list>
      cell_centered_prim_vars{subcell_mesh.number_of_grid_points()};
  cell_centered_prim_vars.assign_subset(
      soln.variables(cell_centered_coords, time,
                     typename System::primitive_variables_tag::tags_list{}));

  typename variables_tag::type initial_variables{
      subcell_mesh.number_of_grid_points()};
  initial_variables.assign_subset(
      soln.variables(cell_centered_coords, time,
                     typename System::gh_system::variables_tag::tags_list{}));
  std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> dg_mesh_velocity{};
  std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>
      subcell_mesh_velocity{};
  if (expansion_velocity.has_value()) {
    dg_mesh_velocity = std::optional<tnsr::I<DataVector, 3>>(
        tnsr::I<DataVector, 3>(num_dg_pts_3d));
    for (int i = 0; i < 3; i++) {
      dg_mesh_velocity.value().get(i) =
          dg_coords.get(i) * expansion_velocity.value();
    }

    subcell_mesh_velocity = std::optional<tnsr::I<DataVector, 3>>(
        tnsr::I<DataVector, 3>(subcell_mesh.number_of_grid_points()));
    for (int i = 0; i < 3; i++) {
      subcell_mesh_velocity.value().get(i) =
          cell_centered_coords.get(i) * expansion_velocity.value();
    }
  }
  std::optional<Scalar<DataVector>> div_dg_mesh_velocity{};
  if (expansion_velocity.has_value()) {
    div_dg_mesh_velocity =
        std::optional<Scalar<DataVector>>(Scalar<DataVector>(num_dg_pts_3d));
    div_dg_mesh_velocity.value().get() = 3.0 * expansion_velocity.value();
  }
  // To be recomputed on each face.
  std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};

  // Neighbor data for reconstruction.
  //
  // 0. neighbors coords (our logical coords +2)
  // 1. compute prims from solution
  // 2. compute prims needed for reconstruction
  // 3. set neighbor data
  evolution::dg::subcell::Tags::GhostDataForReconstruction<3>::type
      neighbor_data{};
  using prims_to_reconstruct_tags = grmhd::GhValenciaDivClean::Tags::
      primitive_grmhd_and_spacetime_reconstruction_tags;
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    auto neighbor_logical_coords = logical_coordinates(subcell_mesh);
    neighbor_logical_coords.get(direction.dimension()) +=
        2.0 * direction.sign();
    auto neighbor_coords = (*grid_to_inertial_map)(
        element_map(neighbor_logical_coords), time, functions_of_time);
    const auto neighbor_prims = soln.variables(
        neighbor_coords, time,
        tmpl::append<typename System::primitive_variables_tag::tags_list,
                     typename System::gh_system::variables_tag::tags_list>{});
    static constexpr size_t prim_components =
        Variables<prims_to_reconstruct_tags>::number_of_independent_components;
    using flux_tags =
        typename grmhd::ValenciaDivClean::ComputeFluxes::return_tags;
    DataVector volume_neighbor_data{
        (prim_components +
         ((fd_derivative_order != ::fd::DerivativeOrder::Two)
              ? Variables<flux_tags>::number_of_independent_components
              : 0)) *
            subcell_mesh.number_of_grid_points(),
        0.0};
    if (fd_derivative_order != ::fd::DerivativeOrder::Two) {
      Variables<typename tmpl::list<
          gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
          gr::Tags::SqrtDetSpatialMetric<DataVector>,
          gr::Tags::SpatialMetric<DataVector, 3>,
          gr::Tags::InverseSpatialMetric<DataVector, 3>>>
          neighbor_cell_centered_spacetime_vars{
              subcell_mesh.number_of_grid_points()};
      neighbor_cell_centered_spacetime_vars.assign_subset(soln.variables(
          neighbor_coords, time,
          typename tmpl::list<
              gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
              gr::Tags::SqrtDetSpatialMetric<DataVector>,
              gr::Tags::SpatialMetric<DataVector, 3>,
              gr::Tags::InverseSpatialMetric<DataVector, 3>>{}));
      Variables<
          typename grmhd::ValenciaDivClean::System::variables_tag::tags_list>
          neighbor_cons{subcell_mesh.number_of_grid_points()};
      apply(make_not_null(&neighbor_cons),
            grmhd::ValenciaDivClean::ConservativeFromPrimitive{},
            neighbor_cell_centered_spacetime_vars, neighbor_prims);

      Variables<flux_tags> neighbor_fluxes{
          std::next(
              volume_neighbor_data.data(),
              static_cast<std::ptrdiff_t>(
                  prim_components * subcell_mesh.number_of_grid_points())),
          Variables<flux_tags>::number_of_independent_components *
              subcell_mesh.number_of_grid_points()};
      apply(make_not_null(&neighbor_fluxes),
            grmhd::ValenciaDivClean::ComputeFluxes{},
            neighbor_cell_centered_spacetime_vars, neighbor_prims,
            neighbor_cons);
      if (expansion_velocity.has_value()) {
        using evolved_vars_tags =
            typename grmhd::ValenciaDivClean::System::variables_tag::tags_list;
        tnsr::I<DataVector, 3> neighbor_mesh_velocity{
            subcell_mesh.number_of_grid_points()};
        for (size_t i = 0; i < 3; i++) {
          neighbor_mesh_velocity.get(i) =
              neighbor_coords.get(i) * expansion_velocity.value();
        }
        tmpl::for_each<evolved_vars_tags>([&neighbor_cons, &neighbor_fluxes,
                                           &neighbor_mesh_velocity](
                                              auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          using flux_tag = ::Tags::Flux<tag, tmpl::size_t<3>, Frame::Inertial>;
          using FluxTensor = typename flux_tag::type;
          const auto& var = get<tag>(neighbor_cons);
          auto& flux = get<flux_tag>(neighbor_fluxes);
          for (size_t storage_index = 0; storage_index < var.size();
               ++storage_index) {
            const auto tensor_index = var.get_tensor_index(storage_index);
            for (size_t i = 0; i < 3; i++) {
              const auto flux_storage_index =
                  FluxTensor::get_storage_index(prepend(tensor_index, i));
              flux[flux_storage_index] -=
                  var[storage_index] * neighbor_mesh_velocity.get(i);
            }
          }
        });
      }
    }
    Variables<prims_to_reconstruct_tags> prims_to_reconstruct{
        volume_neighbor_data.data(),
        prim_components * subcell_mesh.number_of_grid_points()};
    prims_to_reconstruct.assign_subset(neighbor_prims);
    get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
        prims_to_reconstruct) =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(neighbor_prims);
    for (auto& component :
         get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
             prims_to_reconstruct)) {
      component *=
          get(get<hydro::Tags::LorentzFactor<DataVector>>(neighbor_prims));
    }

    // Slice data so we can add it to the element's neighbor data
    DataVector neighbor_data_in_direction =
        evolution::dg::subcell::slice_data(
            volume_neighbor_data, subcell_mesh.extents(),
            recons.ghost_zone_size(), std::unordered_set{direction.opposite()},
            0, {})
            .at(direction.opposite());
    const auto key =
        DirectionalId<3>{direction, *element.neighbors().at(direction).begin()};
    neighbor_data[key] = evolution::dg::subcell::GhostData{1};
    neighbor_data[key].neighbor_ghost_data_for_reconstruction() =
        neighbor_data_in_direction;
  }

  Domain<3> domain{std::move(blocks)};

  const auto gamma1 =  // Gamma1, taken from SpEC BNS
      std::make_unique<ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
          -0.999, 0.999 * 1.0,
          10.0 * 10.0,  // second 10 is "separation" of NSes
          std::array{0.0, 0.0, 0.0});
  const auto gamma2 =  // Gamma2, taken from SpEC BNS
      std::make_unique<ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
          0.01, 1.35 * 1.0 / 1.4, 5.5 * 1.4, std::array{0.0, 0.0, 0.0});

  // Set mortar data, both for ourselves on some interfaces and for our
  // neighbors to emulate a rollback and DG-FD interface.
  evolution::dg::Tags::MortarData<3>::type mortar_data{};
  const Slab slab{0.0, 1.0};
  using BoundaryCorrection = BoundaryCorrections::ProductOfCorrections<
      gh::BoundaryCorrections::UpwindPenalty<3>,
      ValenciaDivClean::BoundaryCorrections::Hll>;
  const BoundaryCorrection boundary_correction{
      gh::BoundaryCorrections::UpwindPenalty<3>{},
      ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8}};
  const auto insert_dg_data = [&](const Direction<3>& direction,
                                  const bool local_data) {
    const Mesh<2> interface_mesh = dg_mesh.slice_away(2);
    const auto face_grid_coords =
        element_map(interface_logical_coordinates(interface_mesh, direction));
    const auto face_coords =
        (*grid_to_inertial_map)(face_grid_coords, time, functions_of_time);
    const auto face_prims = soln.variables(
        face_coords, time,
        tmpl::append<
            typename System::primitive_variables_tag::tags_list,
            typename System::gh_system::variables_tag::tags_list,
            tmpl::list<gr::Tags::Lapse<DataVector>,
                       gr::Tags::Shift<DataVector, 3>,
                       gr::Tags::SqrtDetSpatialMetric<DataVector>,
                       gr::Tags::SpatialMetric<DataVector, 3>,
                       gr::Tags::InverseSpatialMetric<DataVector, 3>>>{});
    using flux_tags =
        typename grmhd::ValenciaDivClean::ComputeFluxes::return_tags;
    using flux_argument_tags =
        typename grmhd::ValenciaDivClean::ComputeFluxes::argument_tags;
    using dg_package_data_temporary_tags =
        typename BoundaryCorrection::dg_package_data_temporary_tags;
    Variables<tmpl::remove_duplicates<tmpl::append<
        typename System::primitive_variables_tag::tags_list,
        typename System::gh_system::variables_tag::tags_list, flux_tags,
        flux_argument_tags,
        tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                   gr::Tags::SqrtDetSpatialMetric<DataVector>,
                   gr::Tags::SpatialMetric<DataVector, 3>,
                   gr::Tags::InverseSpatialMetric<DataVector, 3>,
                   ::gh::Tags::ConstraintGamma1, ::gh::Tags::ConstraintGamma2>,
        dg_package_data_temporary_tags, prims_to_reconstruct_tags>>>
        prims_to_reconstruct{interface_mesh.number_of_grid_points()};
    prims_to_reconstruct.assign_subset(face_prims);
    get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
        prims_to_reconstruct) =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(face_prims);
    for (auto& component :
         get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
             prims_to_reconstruct)) {
      component *= get(get<hydro::Tags::LorentzFactor<DataVector>>(face_prims));
    }

    using p2c_argument_tags = typename grmhd::ValenciaDivClean::
        ConservativeFromPrimitive::argument_tags;
    grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
        make_not_null(
            &get<tmpl::at_c<conserved_tags, 0>>(prims_to_reconstruct)),
        make_not_null(
            &get<tmpl::at_c<conserved_tags, 1>>(prims_to_reconstruct)),
        make_not_null(
            &get<tmpl::at_c<conserved_tags, 2>>(prims_to_reconstruct)),
        make_not_null(
            &get<tmpl::at_c<conserved_tags, 3>>(prims_to_reconstruct)),
        make_not_null(
            &get<tmpl::at_c<conserved_tags, 4>>(prims_to_reconstruct)),
        make_not_null(
            &get<tmpl::at_c<conserved_tags, 5>>(prims_to_reconstruct)),

        get<tmpl::at_c<p2c_argument_tags, 0>>(prims_to_reconstruct),
        get<tmpl::at_c<p2c_argument_tags, 1>>(prims_to_reconstruct),
        get<tmpl::at_c<p2c_argument_tags, 2>>(prims_to_reconstruct),
        get<tmpl::at_c<p2c_argument_tags, 3>>(prims_to_reconstruct),
        get<tmpl::at_c<p2c_argument_tags, 4>>(prims_to_reconstruct),
        get<tmpl::at_c<p2c_argument_tags, 5>>(prims_to_reconstruct),
        get<tmpl::at_c<p2c_argument_tags, 6>>(prims_to_reconstruct),
        get<tmpl::at_c<p2c_argument_tags, 7>>(prims_to_reconstruct),
        get<tmpl::at_c<p2c_argument_tags, 8>>(prims_to_reconstruct),
        get<tmpl::at_c<p2c_argument_tags, 9>>(prims_to_reconstruct));

    grmhd::ValenciaDivClean::ComputeFluxes::apply(
        make_not_null(&get<tmpl::at_c<flux_tags, 0>>(prims_to_reconstruct)),
        make_not_null(&get<tmpl::at_c<flux_tags, 1>>(prims_to_reconstruct)),
        make_not_null(&get<tmpl::at_c<flux_tags, 2>>(prims_to_reconstruct)),
        make_not_null(&get<tmpl::at_c<flux_tags, 3>>(prims_to_reconstruct)),
        make_not_null(&get<tmpl::at_c<flux_tags, 4>>(prims_to_reconstruct)),
        make_not_null(&get<tmpl::at_c<flux_tags, 5>>(prims_to_reconstruct)),

        get<tmpl::at_c<flux_argument_tags, 0>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 1>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 2>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 3>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 4>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 5>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 6>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 7>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 8>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 9>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 10>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 11>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 12>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 13>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_argument_tags, 14>>(prims_to_reconstruct));

    // Add mesh velocity contribution to neighbor fluxes
    std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> face_mesh_velocity =
        {};
    if (expansion_velocity.has_value()) {
      face_mesh_velocity = tnsr::I<DataVector, 3, Frame::Inertial>{
          interface_mesh.number_of_grid_points()};
      for (size_t i = 0; i < 3; i++) {
        face_mesh_velocity.value().get(i) =
            face_coords.get(i) * expansion_velocity.value();
      }

      tmpl::for_each<conserved_tags>([&prims_to_reconstruct,
                                      &face_mesh_velocity](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        using flux_tag = ::Tags::Flux<tag, tmpl::size_t<3>, Frame::Inertial>;
        using FluxTensor = typename flux_tag::type;
        const auto& var = get<tag>(prims_to_reconstruct);
        auto& flux = get<flux_tag>(prims_to_reconstruct);
        for (size_t storage_index = 0; storage_index < var.size();
             ++storage_index) {
          const auto tensor_index = var.get_tensor_index(storage_index);
          for (size_t j = 0; j < 3; j++) {
            const auto flux_storage_index =
                FluxTensor::get_storage_index(prepend(tensor_index, j));
            flux[flux_storage_index] -=
                face_mesh_velocity.value().get(j) * var[storage_index];
          }
        }
      });
    }

    (*gamma1)(
        make_not_null(&get<::gh::Tags::ConstraintGamma1>(prims_to_reconstruct)),
        face_grid_coords, time, functions_of_time);
    (*gamma2)(
        make_not_null(&get<::gh::Tags::ConstraintGamma2>(prims_to_reconstruct)),
        face_grid_coords, time, functions_of_time);

    tnsr::i<DataVector, 3, Frame::Inertial> normal_covector =
        unnormalized_face_normal(interface_mesh, element_map,
                                 *grid_to_inertial_map, time, functions_of_time,
                                 direction);
    const auto normal_magnitude = magnitude(
        normal_covector, get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                             prims_to_reconstruct));
    for (auto& component : normal_covector) {
      component /= get(normal_magnitude);
    }
    tnsr::I<DataVector, 3, Frame::Inertial> normal_vector{
        interface_mesh.number_of_grid_points(), 0.0};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        normal_vector.get(i) +=
            get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                prims_to_reconstruct)
                .get(i, j) *
            normal_covector.get(j);
      }
    }
    if (not local_data) {
      for (size_t i = 0; i < 3; ++i) {
        normal_covector.get(i) *= -1.0;
        normal_vector.get(i) *= -1.0;
      }
    }

    if (expansion_velocity.has_value()) {
      normal_dot_mesh_velocity =
          Scalar<DataVector>{interface_mesh.number_of_grid_points()};
      normal_dot_mesh_velocity.value() =
          dot_product(face_mesh_velocity.value(), normal_covector);
    }

    {
      auto& spatial_velocity_one_form =
          get<tmpl::at_c<dg_package_data_temporary_tags, 4>>(
              prims_to_reconstruct);
      const auto& spatial_velocity =
          get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
              prims_to_reconstruct);
      const auto& spatial_metric =
          get<gr::Tags::SpatialMetric<DataVector, 3>>(prims_to_reconstruct);

      for (size_t i = 0; i < 3; ++i) {
        spatial_velocity_one_form.get(i) = 0.0;
        for (size_t j = 0; j < 3; ++j) {
          spatial_velocity_one_form.get(i) +=
              spatial_metric.get(i, j) * spatial_velocity.get(j);
        }
      }
    }

    using dg_package_fields =
        typename BoundaryCorrection::dg_package_field_tags;
    Variables<dg_package_fields> dg_packaged_data{
        interface_mesh.number_of_grid_points()};
    using dg_package_data_primitive_tags =
        typename BoundaryCorrection::dg_package_data_primitive_tags;
    boundary_correction.dg_package_data(
        make_not_null(&get<tmpl::at_c<dg_package_fields, 0>>(dg_packaged_data)),
        make_not_null(&get<tmpl::at_c<dg_package_fields, 1>>(dg_packaged_data)),
        make_not_null(&get<tmpl::at_c<dg_package_fields, 2>>(dg_packaged_data)),
        make_not_null(&get<tmpl::at_c<dg_package_fields, 3>>(dg_packaged_data)),
        make_not_null(&get<tmpl::at_c<dg_package_fields, 4>>(dg_packaged_data)),
        make_not_null(&get<tmpl::at_c<dg_package_fields, 5>>(dg_packaged_data)),
        make_not_null(&get<tmpl::at_c<dg_package_fields, 6>>(dg_packaged_data)),
        make_not_null(&get<tmpl::at_c<dg_package_fields, 7>>(dg_packaged_data)),
        make_not_null(&get<tmpl::at_c<dg_package_fields, 8>>(dg_packaged_data)),
        make_not_null(&get<tmpl::at_c<dg_package_fields, 9>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 10>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 11>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 12>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 13>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 14>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 15>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 16>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 17>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 18>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 19>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 20>>(dg_packaged_data)),
        make_not_null(
            &get<tmpl::at_c<dg_package_fields, 21>>(dg_packaged_data)),

        // vars,
        get<tmpl::at_c<evolved_tags, 0>>(prims_to_reconstruct),
        get<tmpl::at_c<evolved_tags, 1>>(prims_to_reconstruct),
        get<tmpl::at_c<evolved_tags, 2>>(prims_to_reconstruct),
        get<tmpl::at_c<evolved_tags, 3>>(prims_to_reconstruct),
        get<tmpl::at_c<evolved_tags, 4>>(prims_to_reconstruct),
        get<tmpl::at_c<evolved_tags, 5>>(prims_to_reconstruct),
        get<tmpl::at_c<evolved_tags, 6>>(prims_to_reconstruct),
        get<tmpl::at_c<evolved_tags, 7>>(prims_to_reconstruct),
        get<tmpl::at_c<evolved_tags, 8>>(prims_to_reconstruct),

        // fluxes,
        get<tmpl::at_c<flux_tags, 0>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_tags, 1>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_tags, 2>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_tags, 3>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_tags, 4>>(prims_to_reconstruct),
        get<tmpl::at_c<flux_tags, 5>>(prims_to_reconstruct),

        // temporaries,
        get<tmpl::at_c<dg_package_data_temporary_tags, 0>>(
            prims_to_reconstruct),
        get<tmpl::at_c<dg_package_data_temporary_tags, 1>>(
            prims_to_reconstruct),
        get<tmpl::at_c<dg_package_data_temporary_tags, 2>>(
            prims_to_reconstruct),
        get<tmpl::at_c<dg_package_data_temporary_tags, 3>>(
            prims_to_reconstruct),
        get<tmpl::at_c<dg_package_data_temporary_tags, 4>>(
            prims_to_reconstruct),

        // prims
        get<tmpl::at_c<dg_package_data_primitive_tags, 0>>(
            prims_to_reconstruct),
        get<tmpl::at_c<dg_package_data_primitive_tags, 1>>(
            prims_to_reconstruct),
        get<tmpl::at_c<dg_package_data_primitive_tags, 2>>(
            prims_to_reconstruct),
        get<tmpl::at_c<dg_package_data_primitive_tags, 3>>(
            prims_to_reconstruct),

        normal_covector, normal_vector, face_mesh_velocity,
        normal_dot_mesh_velocity,
        *(soln.equation_of_state().promote_to_3d_eos()));

    DataVector interface_data{dg_packaged_data.size(),
                              std::numeric_limits<double>::signaling_NaN()};
    std::copy(std::data(dg_packaged_data),
              std::next(std::data(dg_packaged_data),
                        static_cast<std::ptrdiff_t>(dg_packaged_data.size())),
              interface_data.begin());

    auto& the_mortar_data = mortar_data[DirectionalId<3>{
        direction, *element.neighbors().at(direction).begin()}];
    if (local_data) {
      the_mortar_data.local().face_mesh = interface_mesh;
      the_mortar_data.local().mortar_data = std::move(interface_data);
    } else {
      the_mortar_data.neighbor().face_mesh = interface_mesh;
      the_mortar_data.neighbor().mortar_data = std::move(interface_data);
    }
  };
  insert_dg_data(Direction<3>::lower_zeta(), true);
  insert_dg_data(Direction<3>::lower_xi(), false);

  // Below are also dummy variables required for compilation due to boundary
  // condition FD ghost data. Since the element used here for testing has
  // neighbors in all directions, BoundaryConditionGhostData::apply() is not
  // actually called so it is okay to leave these variables somewhat poorly
  // initialized.
  typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type
      dummy_normal_covector_and_magnitude{};
  using DampingFunction = ConstraintDamping::DampingFunction<3, Frame::Grid>;
  typename evolution::dg::subcell::Tags::ReconstructionOrder<3>::type
      dummy_reconstruction_order{};

  using CellCenteredFluxesTag = evolution::dg::subcell::Tags::CellCenteredFlux<
      typename System::flux_variables, 3>;
  typename CellCenteredFluxesTag::type cell_centered_fluxes{};

  Variables<typename grmhd::ValenciaDivClean::System::variables_tag::tags_list>
      cell_centered_cons_vars{// grmhd variables
                              subcell_mesh.number_of_grid_points()};
  apply(make_not_null(&cell_centered_cons_vars),
        grmhd::ValenciaDivClean::ConservativeFromPrimitive{},
        cell_centered_spacetime_vars, cell_centered_prim_vars);
  if (fd_derivative_order != ::fd::DerivativeOrder::Two) {
    using flux_tags =
        typename grmhd::ValenciaDivClean::ComputeFluxes::return_tags;
    cell_centered_fluxes =
        Variables<flux_tags>{subcell_mesh.number_of_grid_points()};
    apply(make_not_null(&(cell_centered_fluxes.value())),
          grmhd::ValenciaDivClean::ComputeFluxes{},
          cell_centered_spacetime_vars, cell_centered_prim_vars,
          cell_centered_cons_vars);
    if (expansion_velocity.has_value()) {
      using evolved_vars_tags = typename grmhd::ValenciaDivClean::System::
          variables_tag::tags_list;  // grmhd variables
      tmpl::for_each<evolved_vars_tags>([&cell_centered_cons_vars,
                                         &cell_centered_fluxes,
                                         &subcell_mesh_velocity](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        using flux_tag = ::Tags::Flux<tag, tmpl::size_t<3>, Frame::Inertial>;
        using FluxTensor = typename flux_tag::type;
        const auto& var = get<tag>(cell_centered_cons_vars);
        auto& flux = get<flux_tag>(cell_centered_fluxes.value());
        for (size_t storage_index = 0; storage_index < var.size();
             ++storage_index) {
          const auto tensor_index = var.get_tensor_index(storage_index);
          for (size_t i = 0; i < 3; i++) {
            const auto flux_storage_index =
                FluxTensor::get_storage_index(prepend(tensor_index, i));
            flux[flux_storage_index] -=
                var[storage_index] * subcell_mesh_velocity.value().get(i);
          }
        }
      });
    }
  }
  auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Element<3>, evolution::dg::subcell::Tags::Mesh<3>,
          domain::Tags::Mesh<3>, fd::Tags::Reconstructor<System>,
          evolution::Tags::BoundaryCorrection,
          hydro::Tags::GrmhdEquationOfState,
          typename System::spacetime_variables_tag,
          typename System::primitive_variables_tag, dt_variables_tag,
          variables_tag,
          evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
          evolution::dg::subcell::Tags::ReconstructionOrder<3>,
          evolution::dg::subcell::Tags::GhostZoneInverseJacobian<3>,
          ValenciaDivClean::Tags::ConstraintDampingParameter,
          evolution::dg::Tags::MortarData<3>,
          domain::Tags::ElementMap<3, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::Domain<3>, domain::Tags::ExternalBoundaryConditions<3>,
          domain::Tags::MeshVelocity<3, Frame::Inertial>,
          domain::Tags::DivMeshVelocity,
          domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                        Frame::Inertial>,
          evolution::dg::Tags::NormalCovectorAndMagnitude<3>, ::Tags::Time,
          domain::Tags::FunctionsOfTimeInitialize,
          Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<System>>,
          CellCenteredFluxesTag,
          evolution::dg::subcell::Tags::SubcellOptions<3>,
          gh::Tags::DampingFunctionGamma0<3, Frame::Grid>,
          gh::Tags::DampingFunctionGamma1<3, Frame::Grid>,
          gh::Tags::DampingFunctionGamma2<3, Frame::Grid>,
          ::gh::gauges::Tags::GaugeCondition,
          grmhd::GhValenciaDivClean::fd::Tags::FilterOptions,
          ::Tags::VariableFixer<::VariableFixing::FixToAtmosphere<3>>>,
      db::AddComputeTags<
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<3>,
          ::domain::Tags::MappedCoordinates<
              ::domain::Tags::ElementMap<3, Frame::Grid>,
              evolution::dg::subcell::Tags::Coordinates<3,
                                                        Frame::ElementLogical>,
              evolution::dg::subcell::Tags::Coordinates>,
          evolution::dg::subcell::Tags::InertialCoordinatesCompute<
              ::domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                            Frame::Inertial>>,
          evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGridCompute<
              ::domain::Tags::ElementMap<3, Frame::Grid>, 3>,
          evolution::dg::subcell::fd::Tags::
              DetInverseJacobianLogicalToGridCompute<3>,
          evolution::dg::subcell::fd::Tags::
              InverseJacobianLogicalToInertialCompute<
                  ::domain::CoordinateMaps::Tags::CoordinateMap<
                      3, Frame::Grid, Frame::Inertial>,
                  3>,
          evolution::dg::subcell::fd::Tags::
              DetInverseJacobianLogicalToInertialCompute<
                  ::domain::CoordinateMaps::Tags::CoordinateMap<
                      3, Frame::Grid, Frame::Inertial>,
                  3>,
          domain::Tags::DetInvJacobianCompute<3, Frame::ElementLogical,
                                              Frame::Inertial>>>(
      element, subcell_mesh, dg_mesh,
      std::unique_ptr<grmhd::GhValenciaDivClean::fd::Reconstructor<System>>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      std::unique_ptr<evolution::BoundaryCorrection>{
          std::make_unique<BoundaryCorrections::ProductOfCorrections<
              gh::BoundaryCorrections::UpwindPenalty<3>,
              ValenciaDivClean::BoundaryCorrections::Hll>>(
              gh::BoundaryCorrections::UpwindPenalty<3>{},
              ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8})},
      soln.equation_of_state().promote_to_3d_eos(),
      cell_centered_spacetime_vars, cell_centered_prim_vars,
      // Set incorrect size for dt variables because
      // they should get resized.
      Variables<typename dt_variables_tag::tags_list>{}, initial_variables,
      neighbor_data, dummy_reconstruction_order, ghost_zone_inv_jac, 1.0,
      mortar_data, std::move(element_map),
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}),
      std::move(domain), std::move(external_boundary_conditions),
      dg_mesh_velocity, div_dg_mesh_velocity,
      dg_logical_to_inertial_inv_jacobian, dummy_normal_covector_and_magnitude,
      time, clone_unique_ptrs(functions_of_time),
      DummyEvolutionMetaVars<System>{},
      // Note: These damping functions all assume
      // Grid==Inertial. We need to rescale the widths
      // in the Grid frame for binaries.
      cell_centered_fluxes,
      evolution::dg::subcell::SubcellOptions{
          4.0, 1_st, 1.0e-3, 1.0e-4, false, false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
          std::nullopt, fd_derivative_order, 1, 1, 1},
      std::unique_ptr<DampingFunction>(  // Gamma0,
                                         // taken from
                                         // SpEC BNS
          std::make_unique<
              ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
              0.01, 0.09 * 1.0 / 1.4, 5.5 * 1.4, std::array{0.0, 0.0, 0.0})),
      gamma1->get_clone(), gamma2->get_clone(),
      std::unique_ptr<gh::gauges::GaugeCondition>(
          std::make_unique<gh::gauges::AnalyticChristoffel>(soln.get_clone())),
      grmhd::GhValenciaDivClean::fd::FilterOptions{std::nullopt},
      // Just use a default-constructed fixer and have
      // the reconstructor not call it.
      ::VariableFixing::FixToAtmosphere<3>{});

  db::mutate_apply<ValenciaDivClean::ConservativeFromPrimitive>(
      make_not_null(&box));

  subcell::TimeDerivative<System>::apply(make_not_null(&box));

  // We test that the time derivative converges to zero, so we remove the
  // expected (analytic) value of the time derivative. For a stationary
  // solution on a static mesh this is zero; for a non-zero mesh expansion
  // velocity it is -expansion*x^j*d_j(var), the mesh-advection contribution.
  Variables<evolved_tags> output_minus_expected_dt_vars{
      subcell_mesh.number_of_grid_points()};
  const auto& dt_vars = db::get<dt_variables_tag>(box);

  if constexpr (computing_grmhd_errors) {
    tmpl::for_each<ErrorTagsList>(
        [&box, &subcell_mesh, &expansion_velocity, &cell_centered_coords,
         &cell_centered_logical_to_inertial_inv_jacobian,
         &output_minus_expected_dt_vars, &dt_vars](auto var_tag_v) {
          using var_tag = tmpl::type_from<decltype(var_tag_v)>;
          const auto& var = get<var_tag>(box);
          const auto deriv_var = partial_derivative(
              var, subcell_mesh,
              cell_centered_logical_to_inertial_inv_jacobian);
          auto& output_minus_expected_dt_var =
              get<var_tag>(output_minus_expected_dt_vars);
          const auto& output_dt_var = get<::Tags::dt<var_tag>>(dt_vars);
          for (size_t i = 0; i < output_minus_expected_dt_var.size(); ++i) {
            output_minus_expected_dt_var[i] = output_dt_var[i];
            if (expansion_velocity.has_value()) {
              for (size_t j = 0; j < 3; ++j) {
                const auto deriv_index =
                    j * output_minus_expected_dt_var.size() + i;
                output_minus_expected_dt_var[i] -= cell_centered_coords.get(j) *
                                                   deriv_var[deriv_index] *
                                                   expansion_velocity.value();
              }
            }
          }
        });
  } else {
    // GH residuals need the spatial derivative of the GH evolved variables on
    // the subcell grid, computed with the same FD stencil that the GH
    // TimeDerivative uses internally. partial_derivative does not work here.
    using gh_gradient_tags = typename TimeDerivativeTerms::gh_gradient_tags;
    const auto& gh_evolved_vars = db::get<variables_tag>(box);
    Variables<db::wrap_tags_in<::Tags::deriv, gh_gradient_tags, tmpl::size_t<3>,
                               Frame::Inertial>>
        cell_centered_gh_derivs{subcell_mesh.number_of_grid_points()};
    grmhd::GhValenciaDivClean::fd::spacetime_derivatives<System>(
        make_not_null(&cell_centered_gh_derivs), gh_evolved_vars,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
            box),
        db::get<evolution::dg::subcell::Tags::CellCenteredFlux<
            typename System::flux_variables, 3>>(box)
            .has_value(),
        static_cast<size_t>(fd_derivative_order), subcell_mesh,
        cell_centered_logical_to_inertial_inv_jacobian);

    tmpl::for_each<ErrorTagsList>([&expansion_velocity, &cell_centered_coords,
                                   &output_minus_expected_dt_vars, &dt_vars,
                                   &cell_centered_gh_derivs](auto var_tag_v) {
      using var_tag = tmpl::type_from<decltype(var_tag_v)>;
      using grad_tag = ::Tags::deriv<var_tag, tmpl::size_t<3>, Frame::Inertial>;
      using FluxTensor = typename grad_tag::type;
      const auto& deriv_var = get<grad_tag>(cell_centered_gh_derivs);

      auto& output_minus_expected_dt_var =
          get<var_tag>(output_minus_expected_dt_vars);
      const auto& output_dt_var = get<::Tags::dt<var_tag>>(dt_vars);
      for (size_t i = 0; i < output_minus_expected_dt_var.size(); ++i) {
        output_minus_expected_dt_var[i] = output_dt_var[i];
        if (expansion_velocity.has_value()) {
          const auto tensor_index = output_dt_var.get_tensor_index(i);
          for (size_t j = 0; j < 3; ++j) {
            const auto deriv_index =
                FluxTensor::get_storage_index(prepend(tensor_index, j));
            output_minus_expected_dt_var[i] -= cell_centered_coords.get(j) *
                                               deriv_var[deriv_index] *
                                               expansion_velocity.value();
          }
        }
      }
    });
  }

  std::array<double, tmpl::size<ErrorTagsList>::value> results{};
  size_t result_index = 0;
  tmpl::for_each<ErrorTagsList>([&output_minus_expected_dt_vars, &results,
                                 &result_index](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    gsl::at(results, result_index) =
        max(get(pointwise_l2_norm(get<tag>(output_minus_expected_dt_vars))));
    ++result_index;
  });
  return results;
}

// [[Timeout, 60]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GhValenciaDivClean.Subcell.TimeDerivative",
    "[Unit][Evolution]") {
  // GhValenciaDivClean carries two disjoint physics sectors (GRMHD conserved
  // variables and GH evolved variables) and no single analytic solution in
  // SpECTRE is well suited to exposing FD truncation error for both:
  //
  //   * BondiMichel gives non-trivial fluid velocity and non-zero magnetic
  //     field on a fixed Schwarzschild Kerr-Schild background, so every GRMHD
  //     flux is a real FD signal. It is not self-consistent with Einstein's
  //     equations (no fluid back-reaction on the metric), so the GH matter
  //     source in dt(Pi) does not cancel and the GH residual does not converge
  //     to zero.
  //
  //   * TovStar is a static, self-consistent Einstein+matter solution: every
  //     dt is analytically zero, including dt(Pi). Because the fluid is
  //     static, all GRMHD fluxes involving velocity vanish identically, so
  //     TovStar is useless for exercising GRMHD FD but is exactly what we need
  //     for GH.
  //
  // We therefore run two independent convergence sweeps: BondiMichel for the
  // GRMHD tags and TovStar for the GH tags.
  using NeutrinoTransportSystem = RadiationTransport::NoNeutrinos::System;
  using System = grmhd::GhValenciaDivClean::System<NeutrinoTransportSystem>;

  using DO = ::fd::DerivativeOrder;
  const std::optional<double> dummy_expansion_velocity{};

  // ==== BondiMichel: convergence of GRMHD conserved variables. ====
  //
  // The mesh-velocity variants also stress the mesh-motion code paths in
  // subcell::TimeDerivative that are exercised in the ValenciaDivClean sibling
  // test but were not previously covered by a GhValenciaDivClean convergence
  // test. If the mesh-velocity + higher-order FD interaction is broken this
  // sweep will catch it.
  const auto run_grmhd_convergence_sweep = [](const bool aligned,
                                              const std::optional<double>&
                                                  mesh_velocity) {
    std::optional<std::array<double, tmpl::size<GrmhdErrorTags>::value>>
        previous_error_6{};
    std::optional<std::array<double, tmpl::size<GrmhdErrorTags>::value>>
        previous_error_7{};
    // Higher-order flux corrections (DO::Six and above) do not further reduce
    // the residual in this integration test because the GH evolved variables
    // (SpacetimeMetric, Pi, Phi) are reconstructed to element faces by PPAO
    // along with the primitives, and that face-metric reconstruction has its
    // own limited order of accuracy. Once the flux-correction accuracy
    // matches the metric-reconstruction accuracy (around DO::Four on this
    // problem), additional FD flux orders cannot reduce the reconstruction
    // error that is baked into the Riemann-solver inputs. The higher-order
    // stencils themselves are tested through DO::Ten directly in
    // tests/Unit/NumericalAlgorithms/FiniteDifference/
    // Test_HighOrderFluxCorrection.cpp. The sibling ValenciaDivClean subcell
    // test converges through DO::Ten because it uses analytic (background)
    // metric on faces and never reconstructs it.
    for (const DO fd_do : {DO::Two, DO::Four}) {
      CAPTURE(fd_do);
      CAPTURE(aligned);
      CAPTURE(mesh_velocity.has_value());
      const auto low_res_data =
          aligned
              ? test<System, true, SolutionKind::BondiMichel, GrmhdErrorTags>(
                    6, fd_do, mesh_velocity)
              : test<System, false, SolutionKind::BondiMichel, GrmhdErrorTags>(
                    6, fd_do, mesh_velocity);
      const auto high_res_data =
          aligned
              ? test<System, true, SolutionKind::BondiMichel, GrmhdErrorTags>(
                    7, fd_do, mesh_velocity)
              : test<System, false, SolutionKind::BondiMichel, GrmhdErrorTags>(
                    7, fd_do, mesh_velocity);
      for (size_t i = 0; i < low_res_data.size(); ++i) {
        CAPTURE(i);
        CHECK(gsl::at(high_res_data, i) < gsl::at(low_res_data, i));
        if (previous_error_6.has_value()) {
          CHECK(gsl::at(low_res_data, i) <
                gsl::at(previous_error_6.value(), i));
        }
        if (previous_error_7.has_value()) {
          CHECK(gsl::at(high_res_data, i) <
                gsl::at(previous_error_7.value(), i));
        }
      }
      previous_error_6 = low_res_data;
      previous_error_7 = high_res_data;
    }
  };
  // Aligned coordinates (block-diagonal Jacobian).
  run_grmhd_convergence_sweep(true, dummy_expansion_velocity);
  // Non-aligned coordinates (Frustum): exercises the ghost-cell inverse
  // Jacobian path added for higher-order FD.
  run_grmhd_convergence_sweep(false, dummy_expansion_velocity);
  // Aligned + non-zero expansion mesh velocity: exercises the mesh-motion
  // path in the higher-order flux corrections.
  run_grmhd_convergence_sweep(true, std::optional<double>{0.1});
  // Non-aligned + non-zero expansion mesh velocity.
  run_grmhd_convergence_sweep(false, std::optional<double>{0.1});

  // A zero mesh velocity should give bit-identical output to the no-mesh
  // path (uses BondiMichel because that is the solution used elsewhere in the
  // GRMHD convergence sweep).
  {
    const std::optional<double> zero_expansion_velocity(0.0);
    const auto data_no_mesh_velocity =
        test<System, true, SolutionKind::BondiMichel, GrmhdErrorTags>(
            6, DO::Four, dummy_expansion_velocity);
    const auto data_mesh_velocity =
        test<System, true, SolutionKind::BondiMichel, GrmhdErrorTags>(
            6, DO::Four, zero_expansion_velocity);
    for (size_t i = 0; i < data_no_mesh_velocity.size(); ++i) {
      CAPTURE(i);
      CHECK(gsl::at(data_no_mesh_velocity, i) ==
            gsl::at(data_mesh_velocity, i));
    }
  }

  // ==== TovStar: convergence of GH evolved variables. ====
  //
  // No mesh velocity is applied here: dt of the GH variables is analytically
  // zero for a stationary self-consistent solution, and any non-zero mesh
  // motion would introduce a large analytic dt whose FD-order-10 residual
  // dominates the FD error and hides convergence. Note also that the GH
  // spatial derivatives inside subcell::TimeDerivative are always computed at
  // FD order 2 * ghost_zone_size (=10), independent of fd_derivative_order,
  // so this sweep only checks resolution convergence and does not vary the
  // derivative order.
  {
    const auto low_res_data =
        test<System, true, SolutionKind::TovStar, GhErrorTags>(
            6, DO::Four, dummy_expansion_velocity);
    const auto high_res_data =
        test<System, true, SolutionKind::TovStar, GhErrorTags>(
            7, DO::Four, dummy_expansion_velocity);
    for (size_t i = 0; i < low_res_data.size(); ++i) {
      CAPTURE(i);
      CHECK(gsl::at(high_res_data, i) < gsl::at(low_res_data, i));
    }
  }
}
}  // namespace
}  // namespace grmhd::GhValenciaDivClean
