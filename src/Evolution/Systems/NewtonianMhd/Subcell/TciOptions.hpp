// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianMhd::subcell {
/*!
 * \brief Class holding options used by the MHD-specific parts of the
 * troubled-cell indicator.
 */
struct TciOptions {
 private:
  struct DoNotCheckMagneticField {};

 public:
  /// \brief Minimum mass density before we switch to subcell.
  ///
  /// Identifies places where the density has suddenly become negative.
  struct MinimumValueOfDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Minimum mass density before we switch to subcell."};
  };
  /// \brief Minimum pressure before we switch to subcell.
  ///
  /// Identifies places where the pressure has suddenly become negative.
  struct MinimumValueOfPressure {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Minimum pressure before we switch to subcell."};
  };
  /// \brief Safety factor \f$\epsilon_B\f$ bounding the magnetic energy by the
  /// total energy density.
  ///
  /// The internal energy recovered by `PrimitiveFromConservative` is
  /// \f$\epsilon = (e - |B|^2/2)/\rho - v^2/2\f$, so a cell in which
  /// \f$|B|^2/2\f$ approaches \f$e\f$ is about to yield a negative internal
  /// energy.  Such a cell is flagged when
  /// \f$|B|^2 > 2(1 - \epsilon_B) e\f$.
  struct SafetyFactorForB {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Safety factor for the magnetic field bound."};
  };
  /// \brief The cutoff below which the Persson TCI is not applied to the
  /// magnetic field.
  struct MagneticFieldCutoff {
    using type = Options::Auto<double, DoNotCheckMagneticField>;
    static constexpr Options::String help = {
        "The cutoff where if the maximum of the magnetic field in an element "
        "is below this value we do not apply the Persson TCI to the magnetic "
        "field. This is to avoid switching to subcell in regions where there's "
        "no magnetic field.\n"
        "To disable the magnetic field check, set to "
        "'DoNotCheckMagneticField'."};
  };

  using options = tmpl::list<MinimumValueOfDensity, MinimumValueOfPressure,
                             SafetyFactorForB, MagneticFieldCutoff>;
  static constexpr Options::String help = {
      "Options for the troubled-cell indicator."};

  TciOptions();
  TciOptions(double minimum_density_in, double minimum_pressure_in,
             double safety_factor_for_magnetic_field_in,
             std::optional<double> magnetic_field_cutoff_in);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  double minimum_density{std::numeric_limits<double>::signaling_NaN()};
  double minimum_pressure{std::numeric_limits<double>::signaling_NaN()};
  double safety_factor_for_magnetic_field{
      std::numeric_limits<double>::signaling_NaN()};
  // The signaling_NaN default is chosen so that users hit an error/FPE if the
  // cutoff is not specified, rather than silently defaulting to ignoring the
  // magnetic field.
  std::optional<double> magnetic_field_cutoff{
      std::numeric_limits<double>::signaling_NaN()};
};

namespace OptionTags {
struct TciOptions {
  using type = subcell::TciOptions;
  static constexpr Options::String help = "MHD-specific options for the TCI.";
  using group = ::dg::OptionTags::DiscontinuousGalerkinGroup;
};
}  // namespace OptionTags

namespace Tags {
struct TciOptions : db::SimpleTag {
  using type = subcell::TciOptions;

  using option_tags = tmpl::list<OptionTags::TciOptions>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& tci_options) {
    return tci_options;
  }
};
}  // namespace Tags
}  // namespace NewtonianMhd::subcell
