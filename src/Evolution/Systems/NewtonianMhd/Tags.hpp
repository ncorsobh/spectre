// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/Source.hpp"
#include "Evolution/Systems/NewtonianMhd/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/String.hpp"

/// \cond
class DataVector;
/// \endcond

namespace NewtonianMhd {

/// %OptionTags for the Newtonian MHD system
namespace OptionTags {
template <size_t Dim>
struct SourceTerm {
  using type = std::unique_ptr<NewtonianMhd::Sources::Source<Dim>>;
  static constexpr Options::String help = "The volume source term to be used.";
  using group = ::evolution::OptionTags::SystemGroup;
};

struct GlmCleaningSpeed {
  using type = double;
  static constexpr Options::String help =
      "GLM divergence-cleaning propagation speed c_h.";
  using group = ::evolution::OptionTags::SystemGroup;
};

struct GlmConstraintDampingFactor {
  using type = double;
  static constexpr Options::String help =
      "Dimensionless GLM constraint damping factor alpha "
      "(source term = - alpha * c_h * psi).";
  using group = ::evolution::OptionTags::SystemGroup;
};
}  // namespace OptionTags

/// %Tags for the Newtonian MHD system
namespace Tags {

/// The mass density of the fluid (as a conservative variable).
struct MassDensityCons : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The momentum density of the fluid.
template <size_t Dim, typename Fr>
struct MomentumDensity : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Fr>;
  static std::string name() { return Frame::prefix<Fr>() + "MomentumDensity"; }
};

/// The energy density of the fluid.
struct EnergyDensity : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The evolved (perturbation) magnetic field \f$B_1^i\f$.
///
/// When the background-field splitting is enabled the total magnetic field is
/// \f$B^i = B_0^i + B_1^i\f$ where \f$B_0\f$ is the static curl-free /
/// divergence-free background stored in `BackgroundMagneticField`.  When it is
/// disabled the "perturbation" holds the entire physical field.
template <size_t Dim, typename Fr>
struct MagneticField : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Fr>;
  static std::string name() { return Frame::prefix<Fr>() + "MagneticField"; }
};

/// The static background magnetic field \f$B_0^i\f$.  Non-evolved; set once
/// during initialization from an analytic function.  Zero when the
/// background-field splitting is disabled.
template <size_t Dim, typename Fr>
struct BackgroundMagneticField : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Fr>;
  static std::string name() {
    return Frame::prefix<Fr>() + "BackgroundMagneticField";
  }
};

/// The GLM divergence-cleaning field \f$\psi\f$.
struct DivergenceCleaningField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The GLM cleaning propagation speed \f$c_h\f$ (a scalar constant per
/// element).
struct GlmCleaningSpeed : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::GlmCleaningSpeed>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const double value) { return value; }
};

/// Dimensionless GLM constraint damping factor \f$\alpha\f$.
struct GlmConstraintDampingFactor : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::GlmConstraintDampingFactor>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const double value) { return value; }
};

/// The characteristic speeds (9 in 3D: +/-c_h, v_n +/- c_f, v_n +/- c_A,
/// v_n +/- c_slow, v_n).
template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 2 * Dim + 3>;
};

/// The source term in the evolution equations.
template <size_t Dim>
struct SourceTerm : db::SimpleTag {
  using type = std::unique_ptr<NewtonianMhd::Sources::Source<Dim>>;
  using option_tags = tmpl::list<OptionTags::SourceTerm<Dim>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& source_term) {
    return source_term->get_clone();
  }
};

}  // namespace Tags
}  // namespace NewtonianMhd
