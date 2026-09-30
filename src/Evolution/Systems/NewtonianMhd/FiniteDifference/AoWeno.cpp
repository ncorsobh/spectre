// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/FiniteDifference/AoWeno.hpp"

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/ReconstructWork.tpp"
#include "NumericalAlgorithms/FiniteDifference/AoWeno.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianMhd::fd {
AoWeno53Prim::AoWeno53Prim(const double gamma_hi, const double gamma_lo,
                           const double epsilon,
                           const size_t nonlinear_weight_exponent)
    : gamma_hi_(gamma_hi),
      gamma_lo_(gamma_lo),
      epsilon_(epsilon),
      nonlinear_weight_exponent_(nonlinear_weight_exponent) {
  std::tie(reconstruct_, reconstruct_lower_neighbor_,
           reconstruct_upper_neighbor_) =
      ::fd::reconstruction::aoweno_53_function_pointers<3>(
          nonlinear_weight_exponent_);
}

AoWeno53Prim::AoWeno53Prim(CkMigrateMessage* const msg) : Reconstructor(msg) {}

std::unique_ptr<Reconstructor> AoWeno53Prim::get_clone() const {
  return std::make_unique<AoWeno53Prim>(*this);
}

void AoWeno53Prim::pup(PUP::er& p) {
  Reconstructor::pup(p);
  p | gamma_hi_;
  p | gamma_lo_;
  p | epsilon_;
  p | nonlinear_weight_exponent_;
  if (p.isUnpacking()) {
    std::tie(reconstruct_, reconstruct_lower_neighbor_,
             reconstruct_upper_neighbor_) =
        ::fd::reconstruction::aoweno_53_function_pointers<3>(
            nonlinear_weight_exponent_);
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID AoWeno53Prim::my_PUP_ID = 0;

template <typename TagsList>
void AoWeno53Prim::reconstruct(
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
    const Variables<prims_tags>& volume_prims,
    const EquationsOfState::EquationOfState<false, 2>& eos,
    const Element<3>& element,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<3>& subcell_mesh) const {
  reconstruct_prims_work<prim_tags_for_reconstruction>(
      vars_on_lower_face, vars_on_upper_face,
      [this](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
             const auto& volume_vars, const auto& ghost_cell_vars,
             const auto& subcell_extents, const size_t number_of_variables) {
        reconstruct_(upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
                     ghost_cell_vars, subcell_extents, number_of_variables,
                     gamma_hi_, gamma_lo_, epsilon_);
      },
      volume_prims, eos, element, ghost_data, subcell_mesh, ghost_zone_size(),
      true);
}

template <typename TagsList>
void AoWeno53Prim::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const Variables<prims_tags>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<false, 2>& eos,
    const Element<3>& element,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<3>& subcell_mesh,
    const Direction<3> direction_to_reconstruct) const {
  reconstruct_fd_neighbor_work<prim_tags_for_reconstruction>(
      vars_on_face,
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<3>& subcell_extents,
             const Index<3>& ghost_data_extents,
             const Direction<3>& local_direction_to_reconstruct) {
        reconstruct_lower_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, gamma_hi_, gamma_lo_, epsilon_);
      },
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<3>& subcell_extents,
             const Index<3>& ghost_data_extents,
             const Direction<3>& local_direction_to_reconstruct) {
        reconstruct_upper_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, gamma_hi_, gamma_lo_, epsilon_);
      },
      subcell_volume_prims, eos, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size(), true);
}

bool operator==(const AoWeno53Prim& lhs, const AoWeno53Prim& rhs) {
  // Don't check function pointers since they are set from
  // nonlinear_weight_exponent_
  return lhs.gamma_hi_ == rhs.gamma_hi_ and lhs.gamma_lo_ == rhs.gamma_lo_ and
         lhs.epsilon_ == rhs.epsilon_ and
         lhs.nonlinear_weight_exponent_ == rhs.nonlinear_weight_exponent_;
}

#define TAGS_LIST(data)                                                        \
  tmpl::list<                                                                  \
      Tags::MassDensityCons, Tags::MomentumDensity<>, Tags::EnergyDensity,     \
      Tags::MagneticFieldCons<>, Tags::DivergenceCleaningFieldCons,            \
      hydro::Tags::RestMassDensity<DataVector>,                                \
      hydro::Tags::SpatialVelocity<DataVector, 3>,                             \
      hydro::Tags::SpecificInternalEnergy<DataVector>,                         \
      hydro::Tags::Pressure<DataVector>,                                       \
      hydro::Tags::MagneticField<DataVector, 3>,                               \
      hydro::Tags::DivergenceCleaningField<DataVector>,                        \
      ::Tags::Flux<Tags::MassDensityCons, tmpl::size_t<3>, Frame::Inertial>,   \
      ::Tags::Flux<Tags::MomentumDensity<>, tmpl::size_t<3>, Frame::Inertial>, \
      ::Tags::Flux<Tags::EnergyDensity, tmpl::size_t<3>, Frame::Inertial>,     \
      ::Tags::Flux<Tags::MagneticFieldCons<>, tmpl::size_t<3>,                 \
                   Frame::Inertial>,                                           \
      ::Tags::Flux<Tags::DivergenceCleaningFieldCons, tmpl::size_t<3>,         \
                   Frame::Inertial>>

#define TAGS_LIST_BACKGROUND(data) \
  tmpl::push_back<TAGS_LIST(data), Tags::BackgroundMagneticField<>>

#define INSTANTIATION(r, data) INSTANTIATION(~, ~)
#undef INSTANTIATION

#define INSTANTIATION_IMPL(TAGS, data)                                   \
  template void AoWeno53Prim::reconstruct(                               \
      gsl::not_null<std::array<Variables<TAGS>, 3>*> vars_on_lower_face, \
      gsl::not_null<std::array<Variables<TAGS>, 3>*> vars_on_upper_face, \
      const Variables<prims_tags>& volume_prims,                         \
      const EquationsOfState::EquationOfState<false, 2>& eos,            \
      const Element<3>& element,                                         \
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&      \
          ghost_data,                                                    \
      const Mesh<3>& subcell_mesh) const;                                \
  template void AoWeno53Prim::reconstruct_fd_neighbor(                   \
      gsl::not_null<Variables<TAGS>*> vars_on_face,                      \
      const Variables<prims_tags>& subcell_volume_prims,                 \
      const EquationsOfState::EquationOfState<false, 2>& eos,            \
      const Element<3>& element,                                         \
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&      \
          ghost_data,                                                    \
      const Mesh<3>& subcell_mesh,                                       \
      const Direction<3> direction_to_reconstruct) const;

#define INSTANTIATION(r, data)              \
  INSTANTIATION_IMPL(TAGS_LIST(data), data) \
  INSTANTIATION_IMPL(TAGS_LIST_BACKGROUND(data), data)

INSTANTIATION(~, ~)

#undef INSTANTIATION
#undef INSTANTIATION_IMPL
#undef TAGS_LIST
#undef TAGS_LIST_BACKGROUND
}  // namespace NewtonianMhd::fd
