// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianMhd/TagsDeclarations.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianMhd {
/*!
 * \brief Restores the conserved variables to a state from which the primitive
 * variables can be recovered.
 *
 * `PrimitiveFromConservative` computes the specific internal energy as
 *
 * \f{align*}
 *   \epsilon = \frac{1}{\rho}\left(e - \frac{|B|^2}{2}\right)
 *              - \frac{|S|^2}{2\rho^2} ,
 * \f}
 *
 * which is negative, and the pressure with it, whenever the magnetic or
 * kinetic energy exceeds the total energy density. An under-resolved shock can
 * produce exactly that, and the recovery then yields a NaN rather than a
 * recoverable state.  Three bounds are imposed pointwise, in order:
 *
 * - \f$\rho \geq \rho_{\min}\f$ wherever \f$\rho\f$ falls below
 *   `CutoffDensity`;
 * - \f$|B|^2 \leq 2(1 - \epsilon_B)\,e\f$, enforced by rescaling \f$B\f$;
 * - \f$|S|^2 \leq 2(1 - \epsilon_S)\,\rho\left(e - |B|^2/2\right)\f$, enforced
 *   by rescaling \f$S\f$, which leaves a non-negative internal energy.
 *
 * Rescaling rather than clipping keeps the direction of \f$B\f$ and \f$S\f$,
 * so the fix does not introduce a preferred axis.
 *
 * \note This is a last resort that violates conservation. It is applied
 * pointwise, so it cannot tell an under-resolved shock from a genuinely
 * unphysical state; the troubled-cell indicator should be catching such cells
 * first.
 */
template <size_t Dim>
class FixConservatives {
 public:
  /// \brief Minimum value of the mass density.
  struct MinimumValueOfDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Minimum value of the mass density."};
  };
  /// \brief Cutoff below which the mass density is set to
  /// `MinimumValueOfDensity`.
  struct CutoffDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Cutoff below which the mass density is set to MinimumValueOfDensity."};
  };
  /// \brief Safety factor \f$\epsilon_B\f$ in \f$|B|^2 \leq 2(1-\epsilon_B)e\f$
  struct SafetyFactorForB {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Safety factor for the magnetic field bound."};
  };
  /// \brief Safety factor \f$\epsilon_S\f$ in the momentum density bound
  struct SafetyFactorForS {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Safety factor for the momentum density bound."};
  };
  /// \brief If false the fixing is skipped entirely.
  struct Enable {
    using type = bool;
    static constexpr Options::String help = {
        "If true then the limiting is applied."};
  };

  using options = tmpl::list<MinimumValueOfDensity, CutoffDensity,
                             SafetyFactorForB, SafetyFactorForS, Enable>;
  static constexpr Options::String help = {
      "Restore the conserved variables to a recoverable state."};

  FixConservatives(double minimum_density, double cutoff_density,
                   double safety_factor_for_magnetic_field,
                   double safety_factor_for_momentum_density, bool enable,
                   const Options::Context& context = {});

  FixConservatives() = default;
  FixConservatives(const FixConservatives& /*rhs*/) = default;
  FixConservatives& operator=(const FixConservatives& /*rhs*/) = default;
  FixConservatives(FixConservatives&& /*rhs*/) = default;
  FixConservatives& operator=(FixConservatives&& /*rhs*/) = default;
  ~FixConservatives() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  using return_tags = tmpl::list<NewtonianMhd::Tags::MassDensityCons,
                                 NewtonianMhd::Tags::MomentumDensity<Dim>,
                                 NewtonianMhd::Tags::EnergyDensity,
                                 NewtonianMhd::Tags::MagneticFieldCons<Dim>>;
  using argument_tags = tmpl::list<>;

  /// Returns `true` if any variables were fixed.
  bool operator()(gsl::not_null<Scalar<DataVector>*> mass_density_cons,
                  gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
                      momentum_density,
                  gsl::not_null<Scalar<DataVector>*> energy_density,
                  gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
                      magnetic_field_cons) const;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const FixConservatives<LocalDim>& lhs,
                         const FixConservatives<LocalDim>& rhs);

  double minimum_density_{std::numeric_limits<double>::signaling_NaN()};
  double cutoff_density_{std::numeric_limits<double>::signaling_NaN()};
  double one_minus_safety_factor_for_magnetic_field_{
      std::numeric_limits<double>::signaling_NaN()};
  double one_minus_safety_factor_for_momentum_density_{
      std::numeric_limits<double>::signaling_NaN()};
  bool enable_{true};
};

template <size_t Dim>
bool operator!=(const FixConservatives<Dim>& lhs,
                const FixConservatives<Dim>& rhs);
}  // namespace NewtonianMhd
