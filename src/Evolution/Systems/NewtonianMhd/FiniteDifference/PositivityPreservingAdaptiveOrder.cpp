// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <tuple>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/ReconstructWork.tpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianMhd::fd {

template <size_t Dim>
PositivityPreservingAdaptiveOrderPrim<Dim>::
    PositivityPreservingAdaptiveOrderPrim(
        const double alpha_5, const std::optional<double> alpha_7,
        const std::optional<double> alpha_9,
        const FallbackReconstructorType low_order_reconstructor,
        const Options::Context& context)
    : four_to_the_alpha_5_(pow(4.0, alpha_5)),
      low_order_reconstructor_(low_order_reconstructor) {
  if (low_order_reconstructor_ == FallbackReconstructorType::None) {
    PARSE_ERROR(context, "None is not an allowed low-order reconstructor.");
  }
  if (alpha_7.has_value()) {
    six_to_the_alpha_7_ = pow(6.0, alpha_7.value());
  }
  if (alpha_9.has_value()) {
    eight_to_the_alpha_9_ = pow(8.0, alpha_9.value());
  }
  set_function_pointers();
}

template <size_t Dim>
PositivityPreservingAdaptiveOrderPrim<
    Dim>::PositivityPreservingAdaptiveOrderPrim(CkMigrateMessage* const msg)
    : Reconstructor<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<Reconstructor<Dim>>
PositivityPreservingAdaptiveOrderPrim<Dim>::get_clone() const {
  return std::make_unique<PositivityPreservingAdaptiveOrderPrim>(*this);
}

template <size_t Dim>
void PositivityPreservingAdaptiveOrderPrim<Dim>::set_function_pointers() {
  std::tie(reconstruct_, reconstruct_lower_neighbor_,
           reconstruct_upper_neighbor_) = ::fd::reconstruction::
      positivity_preserving_adaptive_order_function_pointers<Dim, false>(
          false, eight_to_the_alpha_9_.has_value(),
          six_to_the_alpha_7_.has_value(), low_order_reconstructor_);
  std::tie(pp_reconstruct_, pp_reconstruct_lower_neighbor_,
           pp_reconstruct_upper_neighbor_) = ::fd::reconstruction::
      positivity_preserving_adaptive_order_function_pointers<Dim, false>(
          true, eight_to_the_alpha_9_.has_value(),
          six_to_the_alpha_7_.has_value(), low_order_reconstructor_);
}

template <size_t Dim>
void PositivityPreservingAdaptiveOrderPrim<Dim>::pup(PUP::er& p) {
  Reconstructor<Dim>::pup(p);
  p | four_to_the_alpha_5_;
  p | six_to_the_alpha_7_;
  p | eight_to_the_alpha_9_;
  p | low_order_reconstructor_;
  if (p.isUnpacking()) {
    set_function_pointers();
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID PositivityPreservingAdaptiveOrderPrim<Dim>::my_PUP_ID = 0;

template <size_t Dim>
template <typename TagsList>
void PositivityPreservingAdaptiveOrderPrim<Dim>::reconstruct(
    const gsl::not_null<std::array<Variables<TagsList>, Dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, Dim>*>
        vars_on_upper_face,
    const Variables<prims_tags>& volume_prims,
    const EquationsOfState::EquationOfState<false, 2>& eos,
    const Element<Dim>& element,
    const DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<Dim>& subcell_mesh) const {
  const double six_to_the_alpha_7 = six_to_the_alpha_7_.value_or(
      std::numeric_limits<double>::signaling_NaN());
  const double eight_to_the_alpha_9 = eight_to_the_alpha_9_.value_or(
      std::numeric_limits<double>::signaling_NaN());
  reconstruct_prims_work<positivity_preserving_tags>(
      vars_on_lower_face, vars_on_upper_face,
      [this, six_to_the_alpha_7, eight_to_the_alpha_9](
          auto upper_face_vars_ptr, auto lower_face_vars_ptr,
          const auto& volume_vars, const auto& ghost_cell_vars,
          const auto& subcell_extents, const size_t number_of_variables) {
        pp_reconstruct_(upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
                        ghost_cell_vars, subcell_extents, number_of_variables,
                        four_to_the_alpha_5_, six_to_the_alpha_7,
                        eight_to_the_alpha_9);
      },
      volume_prims, eos, element, ghost_data, subcell_mesh, ghost_zone_size(),
      false);
  reconstruct_prims_work<non_positive_tags>(
      vars_on_lower_face, vars_on_upper_face,
      [this, six_to_the_alpha_7, eight_to_the_alpha_9](
          auto upper_face_vars_ptr, auto lower_face_vars_ptr,
          const auto& volume_vars, const auto& ghost_cell_vars,
          const auto& subcell_extents, const size_t number_of_variables) {
        reconstruct_(upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
                     ghost_cell_vars, subcell_extents, number_of_variables,
                     four_to_the_alpha_5_, six_to_the_alpha_7,
                     eight_to_the_alpha_9);
      },
      volume_prims, eos, element, ghost_data, subcell_mesh, ghost_zone_size(),
      true);
}

template <size_t Dim>
template <typename TagsList>
void PositivityPreservingAdaptiveOrderPrim<Dim>::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const Variables<prims_tags>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<false, 2>& eos,
    const Element<Dim>& element,
    const DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<Dim>& subcell_mesh,
    const Direction<Dim> direction_to_reconstruct) const {
  const double six_to_the_alpha_7 = six_to_the_alpha_7_.value_or(
      std::numeric_limits<double>::signaling_NaN());
  const double eight_to_the_alpha_9 = eight_to_the_alpha_9_.value_or(
      std::numeric_limits<double>::signaling_NaN());
  // The neighbour reconstruction fills one face at a time, and the two groups
  // differ only in which function pointer they use.
  reconstruct_fd_neighbor_work<positivity_preserving_tags>(
      vars_on_face,
      [this, six_to_the_alpha_7, eight_to_the_alpha_9](
          const auto tensor_component_on_face_ptr,
          const auto& tensor_component_volume,
          const auto& tensor_component_neighbor,
          const Index<Dim>& subcell_extents,
          const Index<Dim>& ghost_data_extents,
          const Direction<Dim>& local_direction_to_reconstruct) {
        pp_reconstruct_lower_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7, eight_to_the_alpha_9);
      },
      [this, six_to_the_alpha_7, eight_to_the_alpha_9](
          const auto tensor_component_on_face_ptr,
          const auto& tensor_component_volume,
          const auto& tensor_component_neighbor,
          const Index<Dim>& subcell_extents,
          const Index<Dim>& ghost_data_extents,
          const Direction<Dim>& local_direction_to_reconstruct) {
        pp_reconstruct_upper_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7, eight_to_the_alpha_9);
      },
      subcell_volume_prims, eos, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size(), false);
  reconstruct_fd_neighbor_work<non_positive_tags>(
      vars_on_face,
      [this, six_to_the_alpha_7, eight_to_the_alpha_9](
          const auto tensor_component_on_face_ptr,
          const auto& tensor_component_volume,
          const auto& tensor_component_neighbor,
          const Index<Dim>& subcell_extents,
          const Index<Dim>& ghost_data_extents,
          const Direction<Dim>& local_direction_to_reconstruct) {
        reconstruct_lower_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7, eight_to_the_alpha_9);
      },
      [this, six_to_the_alpha_7, eight_to_the_alpha_9](
          const auto tensor_component_on_face_ptr,
          const auto& tensor_component_volume,
          const auto& tensor_component_neighbor,
          const Index<Dim>& subcell_extents,
          const Index<Dim>& ghost_data_extents,
          const Direction<Dim>& local_direction_to_reconstruct) {
        reconstruct_upper_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7, eight_to_the_alpha_9);
      },
      subcell_volume_prims, eos, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size(), true);
}

template <size_t Dim>
bool operator==(const PositivityPreservingAdaptiveOrderPrim<Dim>& lhs,
                const PositivityPreservingAdaptiveOrderPrim<Dim>& rhs) {
  // Don't check function pointers since they are set from the other values.
  return lhs.four_to_the_alpha_5_ == rhs.four_to_the_alpha_5_ and
         lhs.six_to_the_alpha_7_ == rhs.six_to_the_alpha_7_ and
         lhs.eight_to_the_alpha_9_ == rhs.eight_to_the_alpha_9_ and
         lhs.low_order_reconstructor_ == rhs.low_order_reconstructor_;
}

template <size_t Dim>
bool operator!=(const PositivityPreservingAdaptiveOrderPrim<Dim>& lhs,
                const PositivityPreservingAdaptiveOrderPrim<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAGS_LIST(data)                                                   \
  tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<DIM(data)>,     \
             Tags::EnergyDensity, Tags::MagneticFieldCons<DIM(data)>,     \
             Tags::DivergenceCleaningFieldCons,                           \
             hydro::Tags::RestMassDensity<DataVector>,                    \
             hydro::Tags::SpatialVelocity<DataVector, DIM(data)>,         \
             hydro::Tags::SpecificInternalEnergy<DataVector>,             \
             hydro::Tags::Pressure<DataVector>,                           \
             hydro::Tags::MagneticField<DataVector, DIM(data)>,           \
             hydro::Tags::DivergenceCleaningField<DataVector>,            \
             ::Tags::Flux<Tags::MassDensityCons, tmpl::size_t<DIM(data)>, \
                          Frame::Inertial>,                               \
             ::Tags::Flux<Tags::MomentumDensity<DIM(data)>,               \
                          tmpl::size_t<DIM(data)>, Frame::Inertial>,      \
             ::Tags::Flux<Tags::EnergyDensity, tmpl::size_t<DIM(data)>,   \
                          Frame::Inertial>,                               \
             ::Tags::Flux<Tags::MagneticFieldCons<DIM(data)>,             \
                          tmpl::size_t<DIM(data)>, Frame::Inertial>,      \
             ::Tags::Flux<Tags::DivergenceCleaningFieldCons,              \
                          tmpl::size_t<DIM(data)>, Frame::Inertial>>

#define TAGS_LIST_BACKGROUND(data) \
  tmpl::push_back<TAGS_LIST(data), Tags::BackgroundMagneticField<DIM(data)>>

#define INSTANTIATION(r, data)                                      \
  template class PositivityPreservingAdaptiveOrderPrim<DIM(data)>;  \
  template bool operator==(                                         \
      const PositivityPreservingAdaptiveOrderPrim<DIM(data)>& lhs,  \
      const PositivityPreservingAdaptiveOrderPrim<DIM(data)>& rhs); \
  template bool operator!=(                                         \
      const PositivityPreservingAdaptiveOrderPrim<DIM(data)>& lhs,  \
      const PositivityPreservingAdaptiveOrderPrim<DIM(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION

#define INSTANTIATION_IMPL(TAGS, data)                                         \
  template void PositivityPreservingAdaptiveOrderPrim<DIM(data)>::reconstruct( \
      gsl::not_null<std::array<Variables<TAGS>, DIM(data)>*>                   \
          vars_on_lower_face,                                                  \
      gsl::not_null<std::array<Variables<TAGS>, DIM(data)>*>                   \
          vars_on_upper_face,                                                  \
      const Variables<prims_tags>& volume_prims,                               \
      const EquationsOfState::EquationOfState<false, 2>& eos,                  \
      const Element<DIM(data)>& element,                                       \
      const DirectionalIdMap<DIM(data), evolution::dg::subcell::GhostData>&    \
          ghost_data,                                                          \
      const Mesh<DIM(data)>& subcell_mesh) const;                              \
  template void                                                                \
  PositivityPreservingAdaptiveOrderPrim<DIM(data)>::reconstruct_fd_neighbor(   \
      gsl::not_null<Variables<TAGS>*> vars_on_face,                            \
      const Variables<prims_tags>& subcell_volume_prims,                       \
      const EquationsOfState::EquationOfState<false, 2>& eos,                  \
      const Element<DIM(data)>& element,                                       \
      const DirectionalIdMap<DIM(data), evolution::dg::subcell::GhostData>&    \
          ghost_data,                                                          \
      const Mesh<DIM(data)>& subcell_mesh,                                     \
      const Direction<DIM(data)> direction_to_reconstruct) const;

#define INSTANTIATION(r, data)              \
  INSTANTIATION_IMPL(TAGS_LIST(data), data) \
  INSTANTIATION_IMPL(TAGS_LIST_BACKGROUND(data), data)

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef INSTANTIATION_IMPL
#undef TAGS_LIST
#undef TAGS_LIST_BACKGROUND
#undef DIM
}  // namespace NewtonianMhd::fd
