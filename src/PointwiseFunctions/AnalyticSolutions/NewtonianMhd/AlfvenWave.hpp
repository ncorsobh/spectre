// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// Analytic solutions of the Newtonian MHD system
namespace NewtonianMhd::Solutions {
/*!
 * \brief A circularly polarized Alfven wave.
 *
 * This is the standard smooth convergence test for MHD schemes
 * \cite Toth2000. Let \f$\hat{k}\f$ be the propagation direction and
 * \f$(\hat{k}, \hat{e}_1, \hat{e}_2)\f$ a right-handed orthonormal triad. With
 * the phase \f$\varphi = \vec{k}\cdot\vec{x} - \omega t\f$ and the Alfven speed
 * \f$v_A = B_\parallel/\sqrt{\rho_0}\f$, \f$\omega = |\vec{k}| v_A\f$,
 *
 * \f{align*}
 * \rho &= \rho_0 , \qquad P = P_0 , \\
 * \vec{B} &= B_\parallel \hat{k}
 *     + A B_\parallel \left(\cos\varphi\, \hat{e}_1
 *                           + \sin\varphi\, \hat{e}_2\right) , \\
 * \vec{v} &= -A v_A \left(\cos\varphi\, \hat{e}_1
 *                         + \sin\varphi\, \hat{e}_2\right) .
 * \f}
 *
 * Because the polarization is circular, \f$|\vec{B}|^2 =
 * B_\parallel^2(1+A^2)\f$ and \f$|\vec{v}|^2 = A^2v_A^2\f$ are both uniform, so
 * the magnetic pressure is uniform and the density and pressure stay constant:
 * this is an *exact* solution of the nonlinear equations for any amplitude
 * \f$A\f$, not merely a linearization. The wave travels along \f$+\hat{k}\f$;
 * reversing the sign of \f$B_\parallel\f$ reverses the direction.
 *
 * The wavevector is given in units of \f$2\pi\f$, so integer entries give a
 * solution periodic on the unit cube.
 */
class AlfvenWave : public evolution::initial_data::InitialData,
                   public MarkAsAnalyticSolution {
 public:
  static constexpr size_t volume_dim = 3;
  using equation_of_state_type = EquationsOfState::IdealFluid<false>;

  struct WaveVector {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The wavevector of the wave, in units of 2 pi."};
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
  struct ParallelMagneticField {
    using type = double;
    static constexpr Options::String help = {
        "The magnetic field along the propagation direction. Its sign sets the "
        "direction of propagation."};
  };
  struct Amplitude {
    using type = double;
    static constexpr Options::String help = {
        "The amplitude of the transverse perturbation, relative to the "
        "parallel magnetic field."};
  };
  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the ideal fluid."};
  };

  using options = tmpl::list<WaveVector, BackgroundDensity, BackgroundPressure,
                             ParallelMagneticField, Amplitude, AdiabaticIndex>;

  static constexpr Options::String help = {
      "A circularly polarized Alfven wave, an exact solution of the nonlinear "
      "MHD equations."};

  AlfvenWave() = default;
  AlfvenWave(const AlfvenWave& /*rhs*/) = default;
  AlfvenWave& operator=(const AlfvenWave& /*rhs*/) = default;
  AlfvenWave(AlfvenWave&& /*rhs*/) = default;
  AlfvenWave& operator=(AlfvenWave&& /*rhs*/) = default;
  ~AlfvenWave() override = default;

  AlfvenWave(const std::array<double, 3>& wavevector, double background_density,
             double background_pressure, double parallel_magnetic_field,
             double amplitude, double adiabatic_index,
             const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit AlfvenWave(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(AlfvenWave);
  /// \endcond

  /// Retrieve a collection of variables at `(x, t)`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const double t,
      tmpl::list<Tags...> /*meta*/) const {
    return {tuples::get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  const equation_of_state_type& equation_of_state() const {
    return equation_of_state_;
  }

  /// The Alfven speed \f$v_A = B_\parallel/\sqrt{\rho_0}\f$.
  double alfven_speed() const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// @{
  /// Retrieve a single variable at `(x, t)`.
  tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataVector>> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, double t,
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>> /*meta*/) const;

  tuples::TaggedTuple<hydro::Tags::Pressure<DataVector>> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, double t,
      tmpl::list<hydro::Tags::Pressure<DataVector>> /*meta*/) const;

  tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataVector>>
  variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, double t,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataVector>> /*meta*/)
      const;

  tuples::TaggedTuple<
      hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>
  variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x, double t,
            tmpl::list<hydro::Tags::SpatialVelocity<
                DataVector, 3, Frame::Inertial>> /*meta*/) const;

  tuples::TaggedTuple<
      hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>
  variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x, double t,
            tmpl::list<hydro::Tags::MagneticField<
                DataVector, 3, Frame::Inertial>> /*meta*/) const;

  static tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataVector>>
  variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, double t,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataVector>> /*meta*/);
  /// @}

 private:
  /// The phase \f$\vec{k}\cdot\vec{x} - \omega t\f$ at every point.
  DataVector phase(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                   double t) const;

  friend bool operator==(const AlfvenWave& lhs, const AlfvenWave& rhs);

  std::array<double, 3> wavevector_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  std::array<double, 3> first_transverse_direction_{};
  std::array<double, 3> second_transverse_direction_{};
  double background_density_ = std::numeric_limits<double>::signaling_NaN();
  double background_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double parallel_magnetic_field_ =
      std::numeric_limits<double>::signaling_NaN();
  double amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double angular_frequency_ = std::numeric_limits<double>::signaling_NaN();
  equation_of_state_type equation_of_state_;
};

bool operator!=(const AlfvenWave& lhs, const AlfvenWave& rhs);
}  // namespace NewtonianMhd::Solutions
