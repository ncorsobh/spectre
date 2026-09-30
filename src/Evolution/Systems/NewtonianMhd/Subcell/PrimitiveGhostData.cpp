// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/Subcell/PrimitiveGhostData.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::subcell {
DataVector PrimitiveGhostVariables::apply(const Variables<prim_tags>& prims,
                                          const size_t rdmp_size) {
  DataVector buffer{
      prims.number_of_grid_points() *
          Variables<
              prims_to_reconstruct_tags>::number_of_independent_components +
      rdmp_size};
  Variables<prims_to_reconstruct_tags> prims_for_reconstruction(
      buffer.data(), buffer.size() - rdmp_size);
  tmpl::for_each<prims_to_reconstruct_tags>(
      [&prims, &prims_for_reconstruction](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<tag>(prims_for_reconstruction) = get<tag>(prims);
      });
  return buffer;
}

#define INSTANTIATION(r, data) INSTANTIATION(~, ~)
#undef INSTANTIATION
}  // namespace NewtonianMhd::subcell
