// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// Initial data for the Newtonian MHD system
namespace NewtonianMhd::AnalyticData {
/*!
 * \brief Uniform magnetized flow past a perfectly conducting sphere.
 *
 * The fluid is uniform, \f$\rho = \rho_0\f$ and \f$P = P_0\f$, and the velocity
 * is the steady incompressible flow past a sphere of radius \f$R_0\f$.  With
 * \f$\theta\f$ measured from the \f$+z\f$ axis, along which the wind blows with
 * asymptotic speed \f$V_0\f$, the unmagnetized (potential-flow) profile is
 *
 * \f{align*}
 * v_r &= V_0 \cos\theta \left(1 - (R_0/r)^3\right) , \\
 * v_\theta &= -V_0 \sin\theta \left(1 + \tfrac{1}{2}(R_0/r)^3\right) ,
 * \f}
 *
 * while the magnetized case uses the no-slip (Stokes) profile
 *
 * \f{align*}
 * v_r &= V_0 \cos\theta
 *        \left(1 - \tfrac{3}{2}(R_0/r) + \tfrac{1}{2}(R_0/r)^3\right) , \\
 * v_\theta &= -V_0 \sin\theta
 *        \left(1 - \tfrac{3}{4}(R_0/r) - \tfrac{1}{4}(R_0/r)^3\right) .
 * \f}
 *
 * The velocity vanishes inside the sphere.
 *
 * The static background field is the sum of three curl-free, divergence-free
 * pieces: a uniform field \f$B_\text{mag}\hat{x}\f$, the reaction dipole that
 * cancels its radial component at \f$r = R_0\f$, and the conductor's own dipole
 * of strength \f$B_\text{dip}\f$ whose axis is set by \f$\theta_m\f$ and
 * \f$\phi_m\f$,
 *
 * \f{align*}
 * \vec{m}_\text{react} &= -\tfrac{1}{2}R_0^3 B_\text{mag}\, \hat{x} , \\
 * \vec{m}_\text{cond} &= \tfrac{1}{2}R_0^3 B_\text{dip}
 *   \left(\cos\theta_m,\, -\sin\theta_m\sin\phi_m,\,
 *         \sin\theta_m\cos\phi_m\right) , \\
 * \vec{B}_\text{dip}(\vec{m}) &=
 *   \frac{3(\vec{m}\cdot\hat{r})\hat{r} - \vec{m}}{r^3} .
 * \f}
 *
 * Inside the sphere the field is the uniform interior field of a uniformly
 * magnetized sphere, \f$2\vec{m}_\text{cond}/R_0^3\f$.
 *
 * `hydro::Tags::MagneticField` reports the *total* physical field. Whether the
 * evolution carries all of it in \f$B_1\f$ or splits off the static piece into
 * \f$B_0\f$ is decided at initialization by the executable, not here; see
 * `NewtonianMhd::Initialization::BackgroundMagneticField`.
 */
class ConductorFlow : public evolution::initial_data::InitialData,
                      public MarkAsAnalyticData {
 public:
  static constexpr size_t volume_dim = 3;
  using equation_of_state_type = EquationsOfState::IdealFluid<false>;

  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the ideal fluid."};
  };
  struct BackgroundDensity {
    using type = double;
    static constexpr Options::String help = {"The uniform mass density."};
    static type lower_bound() { return 0.0; }
  };
  struct BackgroundPressure {
    using type = double;
    static constexpr Options::String help = {"The uniform pressure."};
    static type lower_bound() { return 0.0; }
  };
  struct AsymptoticVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The wind speed far from the conductor, directed along +z."};
  };
  struct MagneticFieldStrength {
    using type = double;
    static constexpr Options::String help = {
        "Strength of the uniform background field, directed along +x."};
  };
  struct ConductorInternalField {
    using type = double;
    static constexpr Options::String help = {
        "Strength of the conductor's own dipole field."};
  };
  struct MomentTiltAngle {
    using type = double;
    static constexpr Options::String help = {
        "Tilt of the conductor's dipole moment away from +x, in radians."};
  };
  struct MomentAzimuthal {
    using type = double;
    static constexpr Options::String help = {
        "Azimuth of the conductor's dipole moment, in radians."};
  };
  struct ConductorRadius {
    using type = double;
    static constexpr Options::String help = {"The radius of the conductor."};
    static type lower_bound() { return 0.0; }
  };
  struct Magnetized {
    using type = bool;
    static constexpr Options::String help = {
        "Whether to use the no-slip Stokes flow profile (true) or the "
        "free-slip potential flow profile (false)."};
  };
  using options =
      tmpl::list<AdiabaticIndex, BackgroundDensity, BackgroundPressure,
                 AsymptoticVelocity, MagneticFieldStrength,
                 ConductorInternalField, MomentTiltAngle, MomentAzimuthal,
                 ConductorRadius, Magnetized>;

  static constexpr Options::String help = {
      "Uniform magnetized flow past a perfectly conducting sphere."};

  ConductorFlow() = default;
  ConductorFlow(const ConductorFlow& /*rhs*/) = default;
  ConductorFlow& operator=(const ConductorFlow& /*rhs*/) = default;
  ConductorFlow(ConductorFlow&& /*rhs*/) = default;
  ConductorFlow& operator=(ConductorFlow&& /*rhs*/) = default;
  ~ConductorFlow() override = default;

  ConductorFlow(double adiabatic_index, double background_density,
                double background_pressure, double asymptotic_velocity,
                double magnetic_field_strength, double conductor_internal_field,
                double moment_tilt_angle, double moment_azimuthal,
                double conductor_radius, bool magnetized,
                const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit ConductorFlow(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ConductorFlow);
  /// \endcond

  /// Retrieve a collection of variables at position x
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const {
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  const equation_of_state_type& equation_of_state() const {
    return equation_of_state_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// @{
  /// Retrieve a single variable at position x.
  ///
  /// These are public so that a single tag can be requested; the variadic
  /// overload above is a better match only for two or more tags.
  tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataVector>> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>> /*meta*/) const;

  tuples::TaggedTuple<hydro::Tags::Pressure<DataVector>> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      tmpl::list<hydro::Tags::Pressure<DataVector>> /*meta*/) const;

  tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataVector>>
  variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataVector>> /*meta*/)
      const;

  tuples::TaggedTuple<
      hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>
  variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
            tmpl::list<hydro::Tags::SpatialVelocity<
                DataVector, 3, Frame::Inertial>> /*meta*/) const;

  tuples::TaggedTuple<
      hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>
  variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
            tmpl::list<hydro::Tags::MagneticField<
                DataVector, 3, Frame::Inertial>> /*meta*/) const;

  static tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataVector>>
  variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataVector>> /*meta*/);

  tuples::TaggedTuple<NewtonianMhd::Tags::BackgroundMagneticFieldVolume<>>
  variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      tmpl::list<NewtonianMhd::Tags::BackgroundMagneticFieldVolume<>> /*meta*/)
      const;

  /// @}

 private:
  /// The total static magnetic field, whichever tag it is reported under.
  tnsr::I<DataVector, 3, Frame::Inertial> static_magnetic_field(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const;

  friend bool operator==(const ConductorFlow& lhs, const ConductorFlow& rhs);

  double background_density_ = std::numeric_limits<double>::signaling_NaN();
  double background_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double asymptotic_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double magnetic_field_strength_ =
      std::numeric_limits<double>::signaling_NaN();
  double conductor_internal_field_ =
      std::numeric_limits<double>::signaling_NaN();
  double moment_tilt_angle_ = std::numeric_limits<double>::signaling_NaN();
  double moment_azimuthal_ = std::numeric_limits<double>::signaling_NaN();
  double conductor_radius_ = std::numeric_limits<double>::signaling_NaN();
  bool magnetized_ = false;
  equation_of_state_type equation_of_state_;
};

bool operator!=(const ConductorFlow& lhs, const ConductorFlow& rhs);
}  // namespace NewtonianMhd::AnalyticData
