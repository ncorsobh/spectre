// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
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
template <size_t Dim>
class Index;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace gsl {
template <typename T>
class not_null;
template <typename T, std::ptrdiff_t Extent>
class span;
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
 * \brief Positivity-preserving adaptive-order reconstruction.
 *
 * Attempts unlimited fifth- (optionally seventh- and ninth-) order
 * reconstruction, measuring smoothness with a Persson-type indicator, and
 * falls back to the `LowOrderReconstructor` where that is not smooth.  See
 * `::fd::reconstruction::positivity_preserving_adaptive_order()` for the
 * algorithm.
 *
 * The mass density and the pressure are reconstructed with the
 * positivity-preserving limiter, since a negative value of either makes the
 * primitive variables unrecoverable. The velocity, magnetic field and
 * divergence-cleaning field take either sign, so they use the plain
 * adaptive-order scheme.
 */
template <size_t Dim>
class PositivityPreservingAdaptiveOrderPrim : public Reconstructor<Dim> {
 private:
  using MassDensityCons = NewtonianMhd::Tags::MassDensityCons;
  using EnergyDensity = NewtonianMhd::Tags::EnergyDensity;
  using MomentumDensity = NewtonianMhd::Tags::MomentumDensity<Dim>;
  using MagneticFieldCons = NewtonianMhd::Tags::MagneticFieldCons<Dim>;
  using DivergenceCleaningFieldCons =
      NewtonianMhd::Tags::DivergenceCleaningFieldCons;

  using MassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      hydro::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, Dim>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;

  using prims_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure,
                 MagneticField, DivergenceCleaningField>;
  using prim_tags_for_reconstruction =
      tmpl::list<MassDensity, Velocity, Pressure, MagneticField,
                 DivergenceCleaningField>;
  using positivity_preserving_tags = tmpl::list<MassDensity, Pressure>;
  using non_positive_tags =
      tmpl::list<Velocity, MagneticField, DivergenceCleaningField>;

 public:
  using FallbackReconstructorType =
      ::fd::reconstruction::FallbackReconstructorType;

  struct Alpha5 {
    using type = double;
    static constexpr Options::String help = {
        "The alpha parameter in the Persson convergence measurement. 4 is the "
        "right value, but anything in the range of 3-5 is 'reasonable'. "
        "Smaller values allow for more oscillations."};
  };
  struct Alpha7 {
    using type = Options::Auto<double, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "The alpha parameter in the Persson convergence measurement. 4 is the "
        "right value, but anything in the range of 3-5 is 'reasonable'. "
        "Smaller values allow for more oscillations. If specified to None, "
        "then 7th-order reconstruction is not used."};
  };
  struct Alpha9 {
    using type = Options::Auto<double, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "The alpha parameter in the Persson convergence measurement. 4 is the "
        "right value, but anything in the range of 3-5 is 'reasonable'. "
        "Smaller values allow for more oscillations. If specified to None, "
        "then 9th-order reconstruction is not used."};
  };
  struct LowOrderReconstructor {
    using type = FallbackReconstructorType;
    static constexpr Options::String help = {
        "The 2nd/3rd-order reconstruction scheme to use if unlimited 5th-order "
        "isn't okay."};
  };

  using options = tmpl::list<Alpha5, Alpha7, Alpha9, LowOrderReconstructor>;

  static constexpr Options::String help{
      "Positivity-preserving adaptive-order reconstruction."};

  PositivityPreservingAdaptiveOrderPrim() = default;
  PositivityPreservingAdaptiveOrderPrim(
      PositivityPreservingAdaptiveOrderPrim&&) = default;
  PositivityPreservingAdaptiveOrderPrim& operator=(
      PositivityPreservingAdaptiveOrderPrim&&) = default;
  PositivityPreservingAdaptiveOrderPrim(
      const PositivityPreservingAdaptiveOrderPrim&) = default;
  PositivityPreservingAdaptiveOrderPrim& operator=(
      const PositivityPreservingAdaptiveOrderPrim&) = default;
  ~PositivityPreservingAdaptiveOrderPrim() override = default;

  PositivityPreservingAdaptiveOrderPrim(
      double alpha_5, std::optional<double> alpha_7,
      std::optional<double> alpha_9,
      FallbackReconstructorType low_order_reconstructor,
      const Options::Context& context = {});

  explicit PositivityPreservingAdaptiveOrderPrim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor<Dim>,
                                     PositivityPreservingAdaptiveOrderPrim);

  auto get_clone() const -> std::unique_ptr<Reconstructor<Dim>> override;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override {
    if (eight_to_the_alpha_9_.has_value()) {
      return 5;
    }
    return six_to_the_alpha_7_.has_value() ? 4 : 3;
  }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<prims_tags>,
                 hydro::Tags::EquationOfState<false, 2>,
                 domain::Tags::Element<Dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>>;

  template <typename TagsList>
  void reconstruct(
      gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_lower_face,
      gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_upper_face,
      const Variables<prims_tags>& volume_prims,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const Element<Dim>& element,
      const DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>&
          ghost_data,
      const Mesh<Dim>& subcell_mesh) const;

  /// Called by an element doing DG when the neighbor is doing subcell.
  template <typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<prims_tags>& subcell_volume_prims,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const Element<Dim>& element,
      const DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>&
          ghost_data,
      const Mesh<Dim>& subcell_mesh,
      Direction<Dim> direction_to_reconstruct) const;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(
      const PositivityPreservingAdaptiveOrderPrim<LocalDim>& lhs,
      const PositivityPreservingAdaptiveOrderPrim<LocalDim>& rhs);

  void set_function_pointers();

  double four_to_the_alpha_5_ = std::numeric_limits<double>::signaling_NaN();
  std::optional<double> six_to_the_alpha_7_;
  std::optional<double> eight_to_the_alpha_9_;
  FallbackReconstructorType low_order_reconstructor_ =
      FallbackReconstructorType::None;

  using PointerRecons =
      void (*)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
               gsl::not_null<std::array<gsl::span<double>, Dim>*>,
               const gsl::span<const double>&,
               const DirectionMap<Dim, gsl::span<const double>>&,
               const Index<Dim>&, size_t, double, double, double);
  PointerRecons reconstruct_ = nullptr;
  PointerRecons pp_reconstruct_ = nullptr;

  using PointerNeighbor = void (*)(gsl::not_null<DataVector*>,
                                   const DataVector&, const DataVector&,
                                   const Index<Dim>&, const Index<Dim>&,
                                   const Direction<Dim>&, const double&,
                                   const double&, const double&);
  PointerNeighbor reconstruct_lower_neighbor_ = nullptr;
  PointerNeighbor reconstruct_upper_neighbor_ = nullptr;
  PointerNeighbor pp_reconstruct_lower_neighbor_ = nullptr;
  PointerNeighbor pp_reconstruct_upper_neighbor_ = nullptr;
};

template <size_t Dim>
bool operator!=(const PositivityPreservingAdaptiveOrderPrim<Dim>& lhs,
                const PositivityPreservingAdaptiveOrderPrim<Dim>& rhs);
}  // namespace NewtonianMhd::fd
