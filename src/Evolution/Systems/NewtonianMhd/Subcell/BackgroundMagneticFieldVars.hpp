// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/Systems/NewtonianMhd/AllSolutions.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::subcell {
/*!
 * \brief Evaluates the static background magnetic field \f$B_0\f$ on the
 * active grid and on the face-centred finite-difference grids.
 *
 * The finite-difference scheme needs \f$B_0\f$ on cell faces, where it cannot
 * come from reconstruction: it is not evolved, so it is neither reconstructed
 * nor exchanged as ghost data. Since \f$B_0\f$ and the coordinate map are both
 * static these values are evaluated once and reused.
 *
 * The cell-centred field has to follow the element between the DG and
 * finite-difference grids, so it is re-evaluated whenever the number of grid
 * points no longer matches the active grid. This detects every grid switch
 * because a subcell mesh always has more points than the DG mesh it replaces.
 *
 * \note This mutator is meant to be used with
 * `Initialization::Actions::AddSimpleTags`, and to be re-applied after each
 * change of the active grid.
 */
template <size_t Dim>
struct BackgroundMagneticFieldVars : tt::ConformsTo<db::protocols::Mutator> {
  using background_field = NewtonianMhd::Tags::BackgroundMagneticField<Dim>;
  using volume_tag = NewtonianMhd::Tags::BackgroundMagneticFieldVolume<Dim>;
  using subcell_faces_background_field =
      ::evolution::dg::subcell::Tags::OnSubcellFaces<background_field, Dim>;
  using face_vars = typename subcell_faces_background_field::type::value_type;

  using return_tags = tmpl::list<volume_tag, subcell_faces_background_field>;
  using argument_tags = tmpl::list<
      ::Tags::Time, evolution::dg::subcell::Tags::ActiveGrid,
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Inertial>,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::FunctionsOfTime,
      evolution::initial_data::Tags::InitialData>;

  using simple_tags = return_tags;
  using compute_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<evolution::initial_data::Tags::InitialData>;
  using mutable_global_cache_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          background_magnetic_field,
      const gsl::not_null<std::array<face_vars, Dim>*> face_centered_vars,
      const double time, const evolution::dg::subcell::ActiveGrid active_grid,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& dg_inertial_coords,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& subcell_inertial_coords,
      const Mesh<Dim>& subcell_mesh,
      const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
          grid_to_inertial_map,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const evolution::initial_data::InitialData& initial_data) {
    ASSERT(Mesh<Dim>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                     subcell_mesh.quadrature(0)) == subcell_mesh,
           "The subcell mesh must have isotropic basis, quadrature, and "
           "extents but got "
               << subcell_mesh);

    const auto& active_coords =
        active_grid == evolution::dg::subcell::ActiveGrid::Dg
            ? dg_inertial_coords
            : subcell_inertial_coords;
    if (get<0>(*background_magnetic_field).size() !=
        get<0>(active_coords).size()) {
      *background_magnetic_field = evaluate(active_coords, time, initial_data);
    }

    if (get<0>(gsl::at(*face_centered_vars, 0)).size() != 0) {
      return;
    }
    for (size_t dim = 0; dim < Dim; ++dim) {
      const auto basis = make_array<Dim>(subcell_mesh.basis(0));
      auto quadrature = make_array<Dim>(subcell_mesh.quadrature(0));
      auto extents = make_array<Dim>(subcell_mesh.extents(0));
      gsl::at(extents, dim) = subcell_mesh.extents(0) + 1;
      gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
      const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
      gsl::at(*face_centered_vars, dim) = evaluate(
          grid_to_inertial_map(
              logical_to_grid_map(logical_coordinates(face_centered_mesh)),
              time, functions_of_time),
          time, initial_data);
    }
  }

 private:
  static tnsr::I<DataVector, Dim, Frame::Inertial> evaluate(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const double time,
      const evolution::initial_data::InitialData& initial_data) {
    using tags =
        tmpl::list<NewtonianMhd::Tags::BackgroundMagneticFieldVolume<Dim>>;
    return get<NewtonianMhd::Tags::BackgroundMagneticFieldVolume<Dim>>(
        call_with_dynamic_type<
            tuples::tagged_tuple_from_typelist<tags>,
            NewtonianMhd::InitialData::
                background_magnetic_field_initial_data_list<Dim>>(
            &initial_data, [&coords, &time](const auto* const data) {
              if constexpr (is_analytic_solution_v<
                                std::decay_t<decltype(*data)>>) {
                return data->variables(coords, time, tags{});
              } else {
                return data->variables(coords, tags{});
              }
            }));
  }
};
}  // namespace NewtonianMhd::subcell
