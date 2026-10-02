// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace NewtonianMhd::fd {
/*!
 * \brief Adaptive-order WENO reconstruction hybridizing orders 5 and 3. See
 * ::fd::reconstruction::aoweno_53() for details.
 */
class AoWeno53Prim : public Reconstructor {
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
  struct GammaHi {
    using type = double;
    static constexpr Options::String help = {
        "The linear weight for the 5th-order stencil."};
  };
  struct GammaLo {
    using type = double;
    static constexpr Options::String help = {
        "The linear weight for the central 3rd-order stencil."};
  };
  struct Epsilon {
    using type = double;
    static constexpr Options::String help = {
        "The parameter added to the oscillation indicators to avoid division "
        "by zero"};
  };
  struct NonlinearWeightExponent {
    using type = size_t;
    static constexpr Options::String help = {
        "The exponent q to which the oscillation indicators are raised"};
  };

  using options =
      tmpl::list<GammaHi, GammaLo, Epsilon, NonlinearWeightExponent>;
  static constexpr Options::String help{
      "Adaptive-order WENO reconstruction hybridizing orders 5 and 3 using "
      "primitive variables."};

  AoWeno53Prim() = default;
  AoWeno53Prim(AoWeno53Prim&&) = default;
  AoWeno53Prim& operator=(AoWeno53Prim&&) = default;
  AoWeno53Prim(const AoWeno53Prim&) = default;
  AoWeno53Prim& operator=(const AoWeno53Prim&) = default;
  ~AoWeno53Prim() override = default;

  AoWeno53Prim(double gamma_hi, double gamma_lo, double epsilon,
               size_t nonlinear_weight_exponent);

  explicit AoWeno53Prim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor, AoWeno53Prim);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 3; }

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
  template <typename TagsList>
  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<TagsList>*> vars_on_face,
      const Variables<prims_tags>& subcell_volume_prims,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const Element<3>& element,
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
      const Mesh<3>& subcell_mesh, Direction<3> direction_to_reconstruct) const;

 private:
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const AoWeno53Prim& lhs, const AoWeno53Prim& rhs);

  double gamma_hi_ = std::numeric_limits<double>::signaling_NaN();
  double gamma_lo_ = std::numeric_limits<double>::signaling_NaN();
  double epsilon_ = std::numeric_limits<double>::signaling_NaN();
  size_t nonlinear_weight_exponent_ = 0;

  void (*reconstruct_)(gsl::not_null<std::array<gsl::span<double>, 3>*>,
                       gsl::not_null<std::array<gsl::span<double>, 3>*>,
                       const gsl::span<const double>&,
                       const DirectionMap<3, gsl::span<const double>>&,
                       const Index<3>&, size_t, double, double,
                       double) = nullptr;
  void (*reconstruct_lower_neighbor_)(gsl::not_null<DataVector*>,
                                      const DataVector&, const DataVector&,
                                      const Index<3>&, const Index<3>&,
                                      const Direction<3>&, const double&,
                                      const double&, const double&) = nullptr;
  void (*reconstruct_upper_neighbor_)(gsl::not_null<DataVector*>,
                                      const DataVector&, const DataVector&,
                                      const Index<3>&, const Index<3>&,
                                      const Direction<3>&, const double&,
                                      const double&, const double&) = nullptr;
};

inline bool operator!=(const AoWeno53Prim& lhs, const AoWeno53Prim& rhs) {
  return not(lhs == rhs);
}
}  // namespace NewtonianMhd::fd
