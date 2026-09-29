// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/NewtonianMhd/AllSolutions.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \brief Initialization mutators for the Newtonian MHD system.
 *
 * Analytic solutions and analytic data always report the *total* physical
 * magnetic field in `hydro::Tags::MagneticField`. How that field is divided
 * between the static background \f$B_0\f$ and the evolved perturbation
 * \f$B_1\f$ is a property of the evolution scheme, so the choice is made here,
 * by the executable, rather than by the initial data.
 */
namespace NewtonianMhd::Initialization {
/*!
 * \brief Sets the static background magnetic field \f$B_0\f$ from the initial
 * data.
 *
 * Requires the initial data to provide
 * `NewtonianMhd::Tags::BackgroundMagneticFieldVolume`, and that field to be
 * curl-free and divergence-free; otherwise the flux splitting in
 * `NewtonianMhd::ComputeFluxes` is not equivalent to standard MHD.
 *
 * With a DG-subcell hybrid scheme use
 * `NewtonianMhd::subcell::BackgroundMagneticFieldVars` instead, which also
 * handles the finite-difference grids.
 */
template <size_t Dim>
struct BackgroundMagneticField {
  using return_tags = tmpl::list<Tags::BackgroundMagneticFieldVolume<Dim>>;
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 evolution::initial_data::Tags::InitialData>;

  using simple_tags = tmpl::list<Tags::BackgroundMagneticFieldVolume<Dim>>;
  using compute_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<evolution::initial_data::Tags::InitialData>;
  using mutable_global_cache_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          background_magnetic_field,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const evolution::initial_data::InitialData& initial_data) {
    using tags = tmpl::list<Tags::BackgroundMagneticFieldVolume<Dim>>;
    *background_magnetic_field = get<Tags::BackgroundMagneticFieldVolume<Dim>>(
        call_with_dynamic_type<
            tuples::tagged_tuple_from_typelist<tags>,
            NewtonianMhd::InitialData::
                background_magnetic_field_initial_data_list<Dim>>(
            &initial_data, [&coords](const auto* const data) {
              if constexpr (is_analytic_solution_v<
                                std::decay_t<decltype(*data)>>) {
                return data->variables(coords, 0.0, tags{});
              } else {
                return data->variables(coords, tags{});
              }
            }));
  }
};

/*!
 * \brief Replaces the total magnetic field of the initial data by the evolved
 * perturbation \f$B_1 = B - B_0\f$.
 *
 * Must run after the background field has been set.
 */
template <size_t Dim>
struct SubtractBackgroundMagneticField {
  using return_tags =
      tmpl::list<typename System<Dim, true>::primitive_variables_tag>;
  using argument_tags = tmpl::list<Tags::BackgroundMagneticFieldVolume<Dim>>;

  static void apply(
      const gsl::not_null<
          typename System<Dim, true>::primitive_variables_tag::type*>
          primitive_variables,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          background_magnetic_field) {
    auto& magnetic_field =
        get<hydro::Tags::MagneticField<DataVector, Dim>>(*primitive_variables);
    for (size_t i = 0; i < Dim; ++i) {
      magnetic_field.get(i) -= background_magnetic_field.get(i);
    }
  }
};
}  // namespace NewtonianMhd::Initialization
