// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean {

namespace OptionTags {
/// \brief Block and group names for blocks in which MHD is not evolved.
struct WaveZoneBlocksAndGroups {
  static constexpr Options::String help = {
      "A list of block and group names in which to skip MHD evolution\n"
      "entirely, evolving only the generalized harmonic system. In these\n"
      "blocks all MHD conserved and primitive variables are set to zero.\n"
      "Set to 'None' to disable wave zones (the default)."};
  using type =
      Options::Auto<std::vector<std::string>, Options::AutoLabel::None>;
};
}  // namespace OptionTags

namespace Tags {
/// \brief Block IDs of the wave-zone blocks in which MHD is not evolved.
///
/// Resolved from block and group names at startup. An empty vector means no
/// wave zones are configured.
struct WaveZoneBlockIds : db::SimpleTag {
  using type = std::vector<size_t>;

  using option_tags = tmpl::list<OptionTags::WaveZoneBlocksAndGroups,
                                 ::domain::OptionTags::DomainCreator<3>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::optional<std::vector<std::string>>& block_and_group_names,
      const std::unique_ptr<DomainCreator<3>>& domain_creator) {
    if (not block_and_group_names.has_value()) {
      return {};
    }
    return domain::block_ids_from_names(*block_and_group_names,
                                        domain_creator->block_names(),
                                        domain_creator->block_groups());
  }
};
}  // namespace Tags

/*!
 * \brief Zeros all MHD conserved and primitive variables in wave zone blocks.
 *
 * This mutator enforces that wave zone blocks carry exactly zero matter.
 * It should be applied after every call to
 * `VariableFixing::Actions::FixVariables<FixToAtmosphere>` and
 * `Actions::UpdateConservatives` so that the variable fixer cannot restore
 * atmosphere floor values in wave zone elements.
 *
 * Wave zone blocks should also appear in
 * `SubcellOptions::OnlyDgBlocksAndGroups` so that FD reconstruction is never
 * attempted there.
 */
template <typename System>
struct ZeroMhdVariablesInWaveZone {
  using return_tags = tmpl::list<typename System::variables_tag,
                                 typename System::primitive_variables_tag>;
  using argument_tags =
      tmpl::list<domain::Tags::Element<3>, Tags::WaveZoneBlockIds>;

  template <typename AllVarsList, typename PrimVarsList>
  static void apply(const gsl::not_null<Variables<AllVarsList>*> all_vars,
                    const gsl::not_null<Variables<PrimVarsList>*> prim_vars,
                    const Element<3>& element,
                    const std::vector<size_t>& wave_zone_block_ids) {
    if (wave_zone_block_ids.empty() or
        not alg::found(wave_zone_block_ids, element.id().block_id())) {
      return;
    }
    // Zero the MHD (Valencia) conserved variables from the combined variables
    tmpl::for_each<
        typename grmhd::ValenciaDivClean::System::variables_tag::tags_list>(
        [&all_vars]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          auto& var = get<Tag>(*all_vars);
          for (size_t i = 0; i < var.size(); ++i) {
            var[i] = 0.0;
          }
        });
    // Zero all primitive MHD variables
    tmpl::for_each<typename grmhd::ValenciaDivClean::System::
                       primitive_variables_tag::tags_list>(
        [&prim_vars]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          auto& var = get<Tag>(*prim_vars);
          for (size_t i = 0; i < var.size(); ++i) {
            var[i] = 0.0;
          }
        });
  }
};

}  // namespace grmhd::GhValenciaDivClean
