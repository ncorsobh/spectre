// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <pup_stl.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

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
/// \endcond

namespace NewtonianMhd {
/// Volume-source terms for the Newtonian MHD system.
namespace Sources {

/// \brief Base class for a NewtonianMhd volume source term.
///
/// The source term modifies the RHS of the mass, momentum, energy,
/// perturbation-magnetic-field, and GLM-cleaning-field equations.  It is
/// invoked from `TimeDerivativeTerms::apply` after fluxes and prior to
/// integration.
template <bool UseBackgroundMagneticField = false>
class Source : public PUP::able {
 protected:
  Source() = default;

 public:
  ~Source() override = default;

  /// \cond
  explicit Source(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(Source);
  /// \endcond

  virtual auto get_clone() const -> std::unique_ptr<Source> = 0;

  virtual void operator()(
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
      const tnsr::I<DataVector, 3>& coords, double time) const = 0;
};
}  // namespace Sources
}  // namespace NewtonianMhd
