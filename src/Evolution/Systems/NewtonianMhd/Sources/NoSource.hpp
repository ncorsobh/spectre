// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/Source.hpp"
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
 * \brief Used to mark that the initial data do not require source terms in the
 * evolution equations.
 */
template <bool UseBackgroundMagneticField = false>
class NoSource : public Source<UseBackgroundMagneticField> {
 public:
  using options = tmpl::list<>;

  static constexpr Options::String help = {"No source terms added."};

  NoSource() = default;
  NoSource(const NoSource& /*rhs*/) = default;
  NoSource& operator=(const NoSource& /*rhs*/) = default;
  NoSource(NoSource&& /*rhs*/) = default;
  NoSource& operator=(NoSource&& /*rhs*/) = default;
  ~NoSource() override = default;

  /// \cond
  explicit NoSource(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NoSource);
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
};
}  // namespace NewtonianMhd::Sources
