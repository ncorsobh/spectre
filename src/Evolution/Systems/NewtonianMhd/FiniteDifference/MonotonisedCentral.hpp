// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
namespace evolution::dg::subcell {
class GhostData;
}  // namespace evolution::dg::subcell
/// \endcond

namespace NewtonianMhd::fd {
/*!
 * \brief Monotonised central reconstruction. See
 * `::fd::reconstruction::monotonised_central()` for details.
 */
class MonotonisedCentralPrim : public Reconstructor {
 private:
  // Conservative vars tags
  using MassDensityCons = NewtonianMhd::Tags::MassDensityCons;
  using EnergyDensity = NewtonianMhd::Tags::EnergyDensity;
  using MomentumDensity = NewtonianMhd::Tags::MomentumDensity<>;
  using MagneticFieldCons = NewtonianMhd::Tags::MagneticFieldCons<>;
  using DivergenceCleaningFieldCons =
      NewtonianMhd::Tags::DivergenceCleaningFieldCons;

  // Primitive vars tags
  using MassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
  using SpecificInternalEnergy =
      hydro::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;

  using prims_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure,
                 MagneticField, DivergenceCleaningField>;
  using cons_tags = tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity,
                               MagneticFieldCons, DivergenceCleaningFieldCons>;
  using flux_tags = db::wrap_tags_in<::Tags::Flux, cons_tags, tmpl::size_t<3>,
                                     Frame::Inertial>;
  // The background field B0 is smooth by construction and is not limited, so
  // only the evolved perturbation is reconstructed.
  using prim_tags_for_reconstruction =
      tmpl::list<MassDensity, Velocity, Pressure, MagneticField,
                 DivergenceCleaningField>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Monotonised central reconstruction scheme using primitive variables."};

  MonotonisedCentralPrim() = default;
  MonotonisedCentralPrim(MonotonisedCentralPrim&&) = default;
  MonotonisedCentralPrim& operator=(MonotonisedCentralPrim&&) = default;
  MonotonisedCentralPrim(const MonotonisedCentralPrim&) = default;
  MonotonisedCentralPrim& operator=(const MonotonisedCentralPrim&) = default;
  ~MonotonisedCentralPrim() override = default;

  explicit MonotonisedCentralPrim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor, MonotonisedCentralPrim);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 2; }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<prims_tags>,
                 hydro::Tags::EquationOfState<false, 2>,
                 domain::Tags::Element<3>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
                 evolution::dg::subcell::Tags::Mesh<3>>;

  template <typename TagsList>
  void reconstruct(
      gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
      gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
      const Variables<prims_tags>& volume_prims,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const Element<3>& element,
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
      const Mesh<3>& subcell_mesh) const;

  /// Called by an element doing DG when the neighbor is doing subcell.
  ///
  /// This is used to reconstruct the fluxes on the mortar that the subcell
  /// neighbor would have sent had we instead used a two a two-communication
  /// subcell solver (first communication for reconstruction, second for
  /// fluxes).
  template <typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<prims_tags>& subcell_volume_prims,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const Element<3>& element,
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
      const Mesh<3>& subcell_mesh, Direction<3> direction_to_reconstruct) const;
};

inline bool operator==(const MonotonisedCentralPrim& /*lhs*/,
                       const MonotonisedCentralPrim& /*rhs*/) {
  return true;
}

inline bool operator!=(const MonotonisedCentralPrim& lhs,
                       const MonotonisedCentralPrim& rhs) {
  return not(lhs == rhs);
}
}  // namespace NewtonianMhd::fd
