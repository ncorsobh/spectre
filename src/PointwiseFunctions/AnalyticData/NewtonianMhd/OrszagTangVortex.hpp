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

namespace NewtonianMhd::AnalyticData {
/*!
 * \brief The Orszag-Tang vortex.
 *
 * A smooth initial state on the periodic square \f$[0, 1]^2\f$ that develops
 * supersonic turbulence and interacting shocks, \cite OrszagTang1979,
 *
 * \f{align*}
 * \rho &= \gamma^2 P_0 , \qquad P = \gamma P_0 , \\
 * v^x &= -\sin(2\pi y) , \qquad v^y = \sin(2\pi x) , \qquad v^z = 0 , \\
 * B^x &= -B_0\sin(2\pi y) , \qquad B^y = B_0\sin(4\pi x) , \qquad B^z = 0 ,
 * \f}
 *
 * with \f$\gamma = 5/3\f$, \f$P_0 = 1/(4\pi)\f$ and \f$B_0 = 1/\sqrt{4\pi}\f$
 * by default. The initial data are smooth, so the shocks that form are produced
 * by the evolution rather than imposed, which makes this a test of the
 * shock-capturing scheme and of the divergence cleaning together: the initial
 * field is divergence-free, and \f$\nabla\cdot B\f$ should stay small.
 *
 * \note The problem is two dimensional, but the evolution executables are three
 * dimensional, so it is run in a domain that is thin and periodic in \f$z\f$.
 */
class OrszagTangVortex : public evolution::initial_data::InitialData,
                         public MarkAsAnalyticData {
 public:
  static constexpr size_t volume_dim = 3;
  using equation_of_state_type = EquationsOfState::IdealFluid<false>;

  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the ideal fluid."};
    static type suggested_value() { return 5.0 / 3.0; }
  };
  struct Pressure {
    using type = double;
    static constexpr Options::String help = {
        "The uniform pressure, divided by the adiabatic index."};
    static type lower_bound() { return 0.0; }
    static type suggested_value();
  };
  struct MagneticFieldAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "The amplitude of the initial magnetic field."};
    static type suggested_value();
  };

  using options = tmpl::list<AdiabaticIndex, Pressure, MagneticFieldAmplitude>;

  static constexpr Options::String help = {"The Orszag-Tang vortex."};

  OrszagTangVortex() = default;
  OrszagTangVortex(const OrszagTangVortex& /*rhs*/) = default;
  OrszagTangVortex& operator=(const OrszagTangVortex& /*rhs*/) = default;
  OrszagTangVortex(OrszagTangVortex&& /*rhs*/) = default;
  OrszagTangVortex& operator=(OrszagTangVortex&& /*rhs*/) = default;
  ~OrszagTangVortex() override = default;

  OrszagTangVortex(double adiabatic_index, double pressure,
                   double magnetic_field_amplitude,
                   const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit OrszagTangVortex(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(OrszagTangVortex);
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

  static tuples::TaggedTuple<
      hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>
  variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
            tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3,
                                                    Frame::Inertial>> /*meta*/);

  tuples::TaggedTuple<
      hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>
  variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
            tmpl::list<hydro::Tags::MagneticField<
                DataVector, 3, Frame::Inertial>> /*meta*/) const;

  static tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataVector>>
  variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataVector>> /*meta*/);
  /// @}

 private:
  friend bool operator==(const OrszagTangVortex& lhs,
                         const OrszagTangVortex& rhs);

  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double pressure_ = std::numeric_limits<double>::signaling_NaN();
  double magnetic_field_amplitude_ =
      std::numeric_limits<double>::signaling_NaN();
  equation_of_state_type equation_of_state_;
};

bool operator!=(const OrszagTangVortex& lhs, const OrszagTangVortex& rhs);
}  // namespace NewtonianMhd::AnalyticData
