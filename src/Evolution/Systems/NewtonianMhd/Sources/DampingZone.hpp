// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/Source.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianMhd::Sources {
/*!
 * \brief A sponge layer that relaxes the solution towards a uniform magnetized
 * wind near the outer boundary.
 *
 * Within \f$r_\text{sponge} < r < r_\text{outer}\f$ each conservative variable
 * \f$U\f$ is driven towards a target \f$U_\text{target}\f$ at a rate
 *
 * \f{align*}
 * S(U) = -\lambda(r) \left(U - U_\text{target}\right) ,
 * \qquad \lambda(r) = \frac{f(x)}{\tau_\text{damp}} ,
 * \f}
 *
 * where \f$f\f$ is the cubic Hermite `smoothstep<1>` between
 * \f$r_\text{sponge}\f$ and \f$r_\text{outer}\f$,
 *
 * \f{align*}
 * x &= \frac{r - r_\text{sponge}}{r_\text{outer} - r_\text{sponge}}
 *      \quad \text{clamped to } [0, 1] , \\
 * f(x) &= 3x^2 - 2x^3 ,
 * \f}
 *
 * so that \f$\lambda\f$ vanishes smoothly at \f$r_\text{sponge}\f$ and reaches
 * \f$1/\tau_\text{damp}\f$ at \f$r_\text{outer}\f$.
 *
 * The target state is the undisturbed wind: uniform density \f$\rho_0\f$ and
 * pressure \f$P_0\f$, velocity \f$V_0\f$ along the last coordinate axis
 * (\f$+z\f$ in 3D), and vanishing \f$B_1\f$ and \f$\psi\f$.
 */
template <bool UseBackgroundMagneticField = false>
class DampingZone : public Source<UseBackgroundMagneticField> {
 public:
  /// Radius at which the sponge begins.
  struct SpongeInnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius at which the damping profile starts to rise from zero."};
  };
  /// Radius at which the sponge reaches full strength.
  struct SpongeOuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius at which the damping profile reaches 1 / DampingTimescale."};
  };
  /// The damping timescale \f$\tau_\text{damp}\f$.
  struct DampingTimescale {
    using type = double;
    static constexpr Options::String help = {
        "Timescale of the damping at full strength."};
  };
  /// The asymptotic wind speed \f$V_0\f$.
  struct AsymptoticVelocity {
    using type = double;
    static constexpr Options::String help = {
        "Speed of the undisturbed wind, directed along the last coordinate "
        "axis."};
  };
  /// The undisturbed mass density \f$\rho_0\f$.
  struct BackgroundDensity {
    using type = double;
    static constexpr Options::String help = {
        "Mass density of the undisturbed wind."};
  };
  /// The undisturbed pressure \f$P_0\f$.
  struct BackgroundPressure {
    using type = double;
    static constexpr Options::String help = {
        "Pressure of the undisturbed wind."};
  };

  using options =
      tmpl::list<SpongeInnerRadius, SpongeOuterRadius, DampingTimescale,
                 AsymptoticVelocity, BackgroundDensity, BackgroundPressure>;

  static constexpr Options::String help = {
      "Relaxes the solution towards a uniform wind near the outer boundary."};

  DampingZone() = default;
  DampingZone(const DampingZone& /*rhs*/) = default;
  DampingZone& operator=(const DampingZone& /*rhs*/) = default;
  DampingZone(DampingZone&& /*rhs*/) = default;
  DampingZone& operator=(DampingZone&& /*rhs*/) = default;
  ~DampingZone() override = default;

  DampingZone(double sponge_inner_radius, double sponge_outer_radius,
              double damping_timescale, double asymptotic_velocity,
              double background_density, double background_pressure,
              const Options::Context& context = {});

  /// \cond
  explicit DampingZone(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(DampingZone);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  auto get_clone() const
      -> std::unique_ptr<Source<UseBackgroundMagneticField>> override;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
      gsl::not_null<Scalar<DataVector>*> source_energy_density,
      gsl::not_null<tnsr::I<DataVector, 3>*> source_magnetic_field,
      gsl::not_null<Scalar<DataVector>*> source_divergence_cleaning_field,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, 3>& magnetic_field,
      const Scalar<DataVector>& divergence_cleaning_field,
      const tnsr::I<DataVector, 3>& velocity,
      const Scalar<DataVector>& pressure,
      BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
          background_magnetic_field,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const tnsr::I<DataVector, 3>& coords, double time) const override;

  using sourced_variables = tmpl::list<>;
  using argument_tags = tmpl::list<>;

 private:
  template <bool LocalUseBackgroundMagneticField>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(
      const DampingZone<LocalUseBackgroundMagneticField>& lhs,
      const DampingZone<LocalUseBackgroundMagneticField>& rhs);

  double sponge_inner_radius_ = std::numeric_limits<double>::signaling_NaN();
  double sponge_outer_radius_ = std::numeric_limits<double>::signaling_NaN();
  double damping_timescale_ = std::numeric_limits<double>::signaling_NaN();
  double asymptotic_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double background_density_ = std::numeric_limits<double>::signaling_NaN();
  double background_pressure_ = std::numeric_limits<double>::signaling_NaN();
};

template <bool UseBackgroundMagneticField>
bool operator!=(const DampingZone<UseBackgroundMagneticField>& lhs,
                const DampingZone<UseBackgroundMagneticField>& rhs);
}  // namespace NewtonianMhd::Sources
