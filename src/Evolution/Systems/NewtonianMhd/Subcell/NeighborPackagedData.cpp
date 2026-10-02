// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Subcell/NeighborPackagedData.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.tpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::subcell {
template <bool UseBackgroundMagneticField>
DirectionalIdMap<3, DataVector>
NeighborPackagedData<UseBackgroundMagneticField>::apply(
    const db::Access& box,
    const std::vector<DirectionalId<3>>& mortars_to_reconstruct_to) {
  using system = NewtonianMhd::System<UseBackgroundMagneticField>;
  using evolved_vars_tag = typename system::variables_tag;
  using evolved_vars_tags = typename evolved_vars_tag::tags_list;
  using prim_tags = typename system::primitive_variables_tag::tags_list;
  using fluxes_tags = db::wrap_tags_in<::Tags::Flux, evolved_vars_tags,
                                       tmpl::size_t<3>, Frame::Inertial>;

  ASSERT(not db::get<domain::Tags::MeshVelocity<3>>(box).has_value(),
         "Haven't yet added support for moving mesh to DG-subcell. This "
         "should be easy to generalize, but we will want to consider "
         "storing the mesh velocity on the faces instead of "
         "re-slicing/projecting.");

  DirectionalIdMap<3, DataVector> neighbor_package_data{};
  if (mortars_to_reconstruct_to.empty()) {
    return neighbor_package_data;
  }

  const auto& ghost_subcell_data =
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);
  const Mesh<3>& subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<3>>(box);
  const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(box);
  const auto& subcell_options =
      db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(box);

  // Note: we need to compare if projecting the entire mesh or only ghost
  // zones needed is faster. This probably depends on the number of neighbors
  // we have doing FD.
  const auto volume_prims = evolution::dg::subcell::fd::project(
      db::get<typename system::primitive_variables_tag>(box), dg_mesh,
      subcell_mesh.extents());

  const auto& recons = db::get<NewtonianMhd::fd::Tags::Reconstructor>(box);
  const auto& boundary_correction =
      db::get<evolution::Tags::BoundaryCorrection>(box);
  using derived_boundary_corrections =
      NewtonianMhd::BoundaryCorrections::standard_boundary_corrections<
          UseBackgroundMagneticField>;
  tmpl::for_each<derived_boundary_corrections>([&box, &boundary_correction,
                                                &dg_mesh,
                                                &mortars_to_reconstruct_to,
                                                &neighbor_package_data,
                                                &ghost_subcell_data, &recons,
                                                &subcell_mesh, &subcell_options,
                                                &volume_prims](
                                                   auto derived_correction_v) {
    using DerivedCorrection = tmpl::type_from<decltype(derived_correction_v)>;
    if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
      using dg_package_data_temporary_tags =
          typename DerivedCorrection::dg_package_data_temporary_tags;
      using dg_package_data_argument_tags =
          tmpl::append<evolved_vars_tags, prim_tags, fluxes_tags,
                       dg_package_data_temporary_tags>;

      const auto& element = db::get<domain::Tags::Element<3>>(box);
      const auto& eos = get<hydro::Tags::EquationOfState<false, 2>>(box);

      using dg_package_field_tags =
          typename DerivedCorrection::dg_package_field_tags;
      Variables<dg_package_data_argument_tags> vars_on_face;
      Variables<dg_package_field_tags> packaged_data;
      for (const auto& mortar_id : mortars_to_reconstruct_to) {
        const Direction<3>& direction = mortar_id.direction();

        Index<3> extents = subcell_mesh.extents();
        // Switch to face-centered instead of cell-centered points on the FD.
        // There are num_cell_centered+1 face-centered points.
        ++extents[direction.dimension()];

        // Computed prims and cons on face via reconstruction
        const size_t num_face_pts =
            subcell_mesh.extents().slice_away(direction.dimension()).product();
        vars_on_face.initialize(num_face_pts);

        call_with_dynamic_type<
            void, typename NewtonianMhd::fd::Reconstructor::creatable_classes>(
            &recons,
            [&element, &eos, &mortar_id, &ghost_subcell_data, &subcell_mesh,
             &vars_on_face, &volume_prims](const auto& reconstructor) {
              reconstructor->reconstruct_fd_neighbor(
                  make_not_null(&vars_on_face), volume_prims, eos, element,
                  ghost_subcell_data, subcell_mesh, mortar_id.direction());
            });

        // The background field is smooth and is not reconstructed, so its
        // face-centred values are sliced from the stored ones.
        if constexpr (UseBackgroundMagneticField) {
          Index<3> face_extents = subcell_mesh.extents();
          ++face_extents[direction.dimension()];
          data_on_slice(
              make_not_null(&get<NewtonianMhd::Tags::BackgroundMagneticField<>>(
                  vars_on_face)),
              gsl::at(
                  db::get<evolution::dg::subcell::Tags::OnSubcellFaces<
                      NewtonianMhd::Tags::BackgroundMagneticField<>, 3>>(box),
                  direction.dimension()),
              face_extents, direction.dimension(),
              direction.side() == Side::Lower
                  ? 0
                  : face_extents[direction.dimension()] - 1);
        }

        NewtonianMhd::subcell::compute_fluxes<UseBackgroundMagneticField>(
            make_not_null(&vars_on_face),
            db::get<NewtonianMhd::Tags::DivergenceCleaningSpeed>(box));

        tnsr::i<DataVector, 3, Frame::Inertial> normal_covector =
            get<evolution::dg::Tags::NormalCovector<3>>(
                *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<3>>(
                     box)
                     .at(mortar_id.direction()));
        for (auto& t : normal_covector) {
          t *= -1.0;
        }
        const auto dg_normal_covector = normal_covector;
        for (size_t i = 0; i < 3; ++i) {
          normal_covector.get(i) = evolution::dg::subcell::fd::project(
              dg_normal_covector.get(i),
              dg_mesh.slice_away(mortar_id.direction().dimension()),
              subcell_mesh.extents().slice_away(
                  mortar_id.direction().dimension()));
        }

        // Compute the packaged data
        packaged_data.initialize(num_face_pts);
        using dg_package_data_projected_tags = tmpl::append<
            evolved_vars_tags, fluxes_tags, dg_package_data_temporary_tags,
            typename DerivedCorrection::dg_package_data_primitive_tags>;
        evolution::dg::Actions::detail::dg_package_data<system>(
            make_not_null(&packaged_data),
            dynamic_cast<const DerivedCorrection&>(boundary_correction),
            vars_on_face, normal_covector, {std::nullopt}, box,
            typename DerivedCorrection::dg_package_data_volume_tags{},
            dg_package_data_projected_tags{});

        // Reconstruct the DG solution.
        // Really we should be solving the boundary correction and
        // then reconstructing, but away from a shock this doesn't
        // matter.
        auto dg_packaged_data = evolution::dg::subcell::fd::reconstruct(
            packaged_data,
            dg_mesh.slice_away(mortar_id.direction().dimension()),
            subcell_mesh.extents().slice_away(
                mortar_id.direction().dimension()),
            subcell_options.reconstruction_method());
        // Make a view so we can use iterators with std::copy
        DataVector dg_packaged_data_view{dg_packaged_data.data(),
                                         dg_packaged_data.size()};
        neighbor_package_data[mortar_id] = DataVector{dg_packaged_data.size()};
        std::ranges::copy(dg_packaged_data_view,
                          neighbor_package_data[mortar_id].begin());
      }
    }
  });

  return neighbor_package_data;
}

#define USE_BG(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)               \
  template DirectionalIdMap<3, DataVector>   \
  NeighborPackagedData<USE_BG(data)>::apply( \
      const db::Access& box,                 \
      const std::vector<DirectionalId<3>>& mortars_to_reconstruct_to);

GENERATE_INSTANTIATIONS(INSTANTIATION, (true, false))

#undef INSTANTIATION
}  // namespace NewtonianMhd::subcell

#define INSTANTIATION(r, data)                                                \
  template void evolution::dg::subcell::neighbor_reconstructed_face_solution< \
      3, NewtonianMhd::subcell::NeighborPackagedData<USE_BG(data)>>(          \
      gsl::not_null<db::Access*> box);

GENERATE_INSTANTIATIONS(INSTANTIATION, (true, false))

#undef INSTANTIATION
#undef USE_BG
