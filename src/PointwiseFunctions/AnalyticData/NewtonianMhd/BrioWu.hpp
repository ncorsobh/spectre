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
 * \brief The Brio-Wu magnetohydrodynamic shock tube.
 *
 * A planar Riemann problem with the discontinuity at \f$x = 0\f$, at rest on
 * both sides, with a magnetic field that is uniform along the normal and
 * reverses across the discontinuity,
 *
 * \f{align*}
 * (\rho, P, B^y)_{x<0} &= (\rho_L, P_L, B^y_L) , \\
 * (\rho, P, B^y)_{x>0} &= (\rho_R, P_R, B^y_R) , \\
 * B^x &= B^x_0 , \qquad B^z = 0 , \qquad v^i = 0 .
 * \f}
 *
 * The defaults are the values of \cite BrioWu1988: \f$\gamma = 2\f$,
 * \f$(\rho, P, B^y)_L = (1, 1, 1)\f$, \f$(\rho, P, B^y)_R =
 * (0.125, 0.1, -1)\f$ and \f$B^x_0 = 0.75\f$. The solution develops a fast
 * rarefaction, a slow compound wave, a contact discontinuity, a slow shock and
 * a fast rarefaction, and so exercises the shock-capturing scheme; there is no
 * closed-form solution to compare against.
 *
 * \note The problem is one dimensional, but the evolution executables are three
 * dimensional, so it is run in a domain that is thin and periodic in \f$y\f$
 * and \f$z\f$.
 */
class BrioWu : public evolution::initial_data::InitialData,
               public MarkAsAnalyticData {
 public:
  static constexpr size_t volume_dim = 3;
  using equation_of_state_type = EquationsOfState::IdealFluid<false>;

  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the ideal fluid."};
    static type suggested_value() { return 2.0; }
  };
  struct LeftDensity {
    using type = double;
    static constexpr Options::String help = {"The mass density for x < 0."};
    static type lower_bound() { return 0.0; }
    static type suggested_value() { return 1.0; }
  };
  struct LeftPressure {
    using type = double;
    static constexpr Options::String help = {"The pressure for x < 0."};
    static type lower_bound() { return 0.0; }
    static type suggested_value() { return 1.0; }
  };
  struct LeftTransverseMagneticField {
    using type = double;
    static constexpr Options::String help = {
        "The y component of the magnetic field for x < 0."};
    static type suggested_value() { return 1.0; }
  };
  struct RightDensity {
    using type = double;
    static constexpr Options::String help = {"The mass density for x > 0."};
    static type lower_bound() { return 0.0; }
    static type suggested_value() { return 0.125; }
  };
  struct RightPressure {
    using type = double;
    static constexpr Options::String help = {"The pressure for x > 0."};
    static type lower_bound() { return 0.0; }
    static type suggested_value() { return 0.1; }
  };
  struct RightTransverseMagneticField {
    using type = double;
    static constexpr Options::String help = {
        "The y component of the magnetic field for x > 0."};
    static type suggested_value() { return -1.0; }
  };
  struct ParallelMagneticField {
    using type = double;
    static constexpr Options::String help = {
        "The x component of the magnetic field, uniform across the tube."};
    static type suggested_value() { return 0.75; }
  };

  using options =
      tmpl::list<AdiabaticIndex, LeftDensity, LeftPressure,
                 LeftTransverseMagneticField, RightDensity, RightPressure,
                 RightTransverseMagneticField, ParallelMagneticField>;

  static constexpr Options::String help = {
      "The Brio-Wu magnetohydrodynamic shock tube."};

  BrioWu() = default;
  BrioWu(const BrioWu& /*rhs*/) = default;
  BrioWu& operator=(const BrioWu& /*rhs*/) = default;
  BrioWu(BrioWu&& /*rhs*/) = default;
  BrioWu& operator=(BrioWu&& /*rhs*/) = default;
  ~BrioWu() override = default;

  BrioWu(double adiabatic_index, double left_density, double left_pressure,
         double left_transverse_magnetic_field, double right_density,
         double right_pressure, double right_transverse_magnetic_field,
         double parallel_magnetic_field, const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit BrioWu(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BrioWu);
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
  friend bool operator==(const BrioWu& lhs, const BrioWu& rhs);

  double left_density_ = std::numeric_limits<double>::signaling_NaN();
  double left_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double left_transverse_magnetic_field_ =
      std::numeric_limits<double>::signaling_NaN();
  double right_density_ = std::numeric_limits<double>::signaling_NaN();
  double right_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double right_transverse_magnetic_field_ =
      std::numeric_limits<double>::signaling_NaN();
  double parallel_magnetic_field_ =
      std::numeric_limits<double>::signaling_NaN();
  equation_of_state_type equation_of_state_;
};

bool operator!=(const BrioWu& lhs, const BrioWu& rhs);
}  // namespace NewtonianMhd::AnalyticData
