// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::subcell {
/*!
 * \brief Zeros out the MHD time derivatives in the elements next to a DG-only
 * block that themselves are not DG-only elements.
 */
template <typename System>
struct ZeroMhdTimeDerivatives {
  // using variables_tags_list =
  // tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
  // hydro::Tags::SpatialVelocity<DataVector, 3>>;
  using return_tags = tmpl::append<
      tmpl::list<::Tags::Variables<db::wrap_tags_in<
          ::Tags::dt, typename System::variables_tag::tags_list>>>,
      tmpl::list<::Tags::Variables<
          typename System::grmhd_system::primitive_variables_tag::tags_list>>>;
  using argument_tags =
      tmpl::list<domain::Tags::Element<3>,
                 evolution::dg::subcell::Tags::SubcellOptions<3>, ::Tags::Time,
                 domain::Tags::Coordinates<3, Frame::Inertial>>;
  // evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>>;

  template <class DtTagsList, class TagsList>
  static void apply(
      const gsl::not_null<Variables<DtTagsList>*> dt_variables,
      const gsl::not_null<Variables<TagsList>*> variables,
      const Element<3>& element,
      const evolution::dg::subcell::SubcellOptions& subcell_options,
      const double time,
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords) {
    const bool bordering_dg_block = alg::any_of(
        element.neighbors(),
        [&subcell_options](const auto& direction_and_neighbor) {
          const size_t first_block_id =
              direction_and_neighbor.second.ids().begin()->block_id();
          return alg::found(subcell_options.only_dg_block_ids(),
                            first_block_id);
        });
    const bool in_dg_only_zone = alg::found(subcell_options.only_dg_block_ids(),
                                            element.id().block_id());
    if (bordering_dg_block and not in_dg_only_zone) {
      tmpl::for_each<
          typename grmhd::ValenciaDivClean::System::variables_tag::tags_list>(
          [&dt_variables]<class Tag>(tmpl::type_<Tag> /*meta*/) {
            auto& var = get<::Tags::dt<Tag>>(*dt_variables);
            for (size_t i = 0; i < var.size(); ++i) {
              var[i] = 0.0;
            }
          });
      auto& rest_mass_density =
          get<hydro::Tags::RestMassDensity<DataVector>>(*variables);
      auto& spatial_velocity =
          get<hydro::Tags::SpatialVelocity<DataVector, 3>>(*variables);
      const auto gamma_time_factor = (time < 250.) ? time / 250. : 1.;
      const auto gamma_mass_factor = (time < 500.) ? time / 500. : 1.;
      const auto gamma_cutoff_factor =
          (time > 2000.) ? (time < 2500.) ? 1. - (time - 2000.) / 500. : 0.
                         : 1.;
      /*    time * 1000. * std::exp(-time / 500.) * std::pow(1. / 500., 2);*/
      /*auto& temperature = db::get<hydro::Tags::Temperature<DataVector>>(box);
      auto& electron_fraction =
      db::get<hydro::Tags::ElectronFraction<DataVector>>(box);*/
      /*const auto equation_of_state =
          db::get<hydro::Tags::GrmhdEquationOfState>(box);*/
      const auto mag = get(magnitude(coords));
      /*std::cout << variables->number_of_grid_points() << "\n";
      std::cout << mag.size() << "\n";
      ASSERT(variables->number_of_grid_points() == mag.size(), "Angry");*/
      for (size_t i = 0; i < mag.size(); ++i) {
        // std::cout << mag[i] << "\n";
        if (mag[i] < 56.4) {//52.4) {
          get(rest_mass_density)[i] =
              (5.e-8 + gamma_mass_factor * 2.5e-8) * gamma_cutoff_factor;
          for (size_t j = 0; j < 3; ++j) {
            spatial_velocity.get(j)[i] = gamma_cutoff_factor *
                                         gamma_time_factor * -0.1 *
                                         coords.get(j)[i] / mag[i];
          }
        }
      }
    }
  }
};
}  // namespace grmhd::GhValenciaDivClean::subcell
