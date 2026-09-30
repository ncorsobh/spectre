// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianMhd::BoundaryCorrections {
/*!
 * \brief An HLL (Harten-Lax-van Leer) Riemann solver for the NewtonianMhd
 * system.
 *
 * Let \f$U\f$ be an evolved variable, \f$F^i\f$ its flux, and \f$n_i\f$ the
 * outward directed unit normal to the interface.  Denoting \f$F := n_i F^i\f$,
 * the HLL boundary correction is \cite Harten1983
 *
 * \f{align*}
 * G_\text{HLL} = \frac{\lambda_\text{max} F_\text{int} +
 * \lambda_\text{min} F_\text{ext}}{\lambda_\text{max} - \lambda_\text{min}}
 * - \frac{\lambda_\text{min}\lambda_\text{max}}{\lambda_\text{max} -
 *   \lambda_\text{min}} \left(U_\text{int} - U_\text{ext}\right)
 * \f}
 *
 * with
 *
 * \f{align*}
 * \lambda_\text{min} &=
 * \text{min}\left(\lambda^{-}_\text{int},-\lambda^{+}_\text{ext}, 0\right) \\
 * \lambda_\text{max} &=
 * \text{max}\left(\lambda^{+}_\text{int},-\lambda^{-}_\text{ext}, 0\right)
 * \f}
 *
 * For Newtonian MHD with GLM divergence cleaning the extreme characteristic
 * speeds are
 *
 * \f{align*}
 * \lambda^{+} &= \max\left(v^in_i + c_f,\; c_h\right) \\
 * \lambda^{-} &= \min\left(v^in_i - c_f,\; -c_h\right)
 * \f}
 *
 * where \f$c_f = \sqrt{c_s^2 + |B_0 + B_1|^2/\rho}\f$ is the fast magnetosonic
 * speed built from the *total* magnetic field and \f$c_h\f$ is the GLM cleaning
 * speed, which bounds the speed of the \f$\psi\f$ waves.
 */
template <bool UseBackgroundMagneticField = false>
class Hll final : public evolution::BoundaryCorrection {
 public:
  struct LargestOutgoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct LargestIngoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the HLL boundary correction term for the Newtonian MHD "
      "system."};

  Hll() = default;
  Hll(const Hll&) = default;
  Hll& operator=(const Hll&) = default;
  Hll(Hll&&) = default;
  Hll& operator=(Hll&&) = default;
  ~Hll() override = default;

  /// \cond
  explicit Hll(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Hll);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<>,
                 Tags::EnergyDensity, Tags::MagneticFieldCons<>,
                 Tags::DivergenceCleaningFieldCons,
                 ::Tags::NormalDotFlux<Tags::MassDensityCons>,
                 ::Tags::NormalDotFlux<Tags::MomentumDensity<>>,
                 ::Tags::NormalDotFlux<Tags::EnergyDensity>,
                 ::Tags::NormalDotFlux<Tags::MagneticFieldCons<>>,
                 ::Tags::NormalDotFlux<Tags::DivergenceCleaningFieldCons>,
                 LargestOutgoingCharSpeed, LargestIngoingCharSpeed>;
  using dg_package_data_temporary_tags =
      background_magnetic_field_tag_list<Tags::BackgroundMagneticField<>,
                                         UseBackgroundMagneticField>;
  using dg_package_data_primitive_tags =
      tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>>;
  using dg_package_data_volume_tags =
      tmpl::list<hydro::Tags::EquationOfState<false, 2>,
                 Tags::DivergenceCleaningSpeed>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  /// @{
  /// The background-field overload is selected by
  /// `dg_package_data_temporary_tags`: with the splitting disabled that
  /// list is empty, so the framework calls the shorter one and \f$B_0\f$ is
  /// never projected onto faces.
  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_mass_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_momentum_density,
      gsl::not_null<Scalar<DataVector>*> packaged_energy_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_magnetic_field,
      gsl::not_null<Scalar<DataVector>*> packaged_divergence_cleaning_field,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_mass_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_momentum_density,
      gsl::not_null<Scalar<DataVector>*>
          packaged_normal_dot_flux_energy_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_magnetic_field,
      gsl::not_null<Scalar<DataVector>*>
          packaged_normal_dot_flux_divergence_cleaning_field,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_outgoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_ingoing_char_speed,

      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
      const Scalar<DataVector>& divergence_cleaning_field,

      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_mass_density,
      const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_momentum_density,
      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_energy_density,
      const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_magnetic_field,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          flux_divergence_cleaning_field,

      const tnsr::I<DataVector, 3, Frame::Inertial>& velocity,
      const Scalar<DataVector>& specific_internal_energy,

      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const EquationsOfState::EquationOfState<false, 2>& equation_of_state,
      double divergence_cleaning_speed) const;

  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_mass_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_momentum_density,
      gsl::not_null<Scalar<DataVector>*> packaged_energy_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_magnetic_field,
      gsl::not_null<Scalar<DataVector>*> packaged_divergence_cleaning_field,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_mass_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_momentum_density,
      gsl::not_null<Scalar<DataVector>*>
          packaged_normal_dot_flux_energy_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_magnetic_field,
      gsl::not_null<Scalar<DataVector>*>
          packaged_normal_dot_flux_divergence_cleaning_field,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_outgoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_ingoing_char_speed,

      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
      const Scalar<DataVector>& divergence_cleaning_field,

      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_mass_density,
      const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_momentum_density,
      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_energy_density,
      const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_magnetic_field,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          flux_divergence_cleaning_field,

      BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
          background_magnetic_field,

      const tnsr::I<DataVector, 3, Frame::Inertial>& velocity,
      const Scalar<DataVector>& specific_internal_energy,

      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const EquationsOfState::EquationOfState<false, 2>& equation_of_state,
      double divergence_cleaning_speed) const;
  /// @}

  void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> boundary_correction_mass_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          boundary_correction_momentum_density,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_energy_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          boundary_correction_magnetic_field,
      gsl::not_null<Scalar<DataVector>*>
          boundary_correction_divergence_cleaning_field,
      const Scalar<DataVector>& mass_density_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density_int,
      const Scalar<DataVector>& energy_density_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field_int,
      const Scalar<DataVector>& divergence_cleaning_field_int,
      const Scalar<DataVector>& normal_dot_flux_mass_density_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_momentum_density_int,
      const Scalar<DataVector>& normal_dot_flux_energy_density_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_magnetic_field_int,
      const Scalar<DataVector>& normal_dot_flux_divergence_cleaning_field_int,
      const Scalar<DataVector>& largest_outgoing_char_speed_int,
      const Scalar<DataVector>& largest_ingoing_char_speed_int,
      const Scalar<DataVector>& mass_density_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density_ext,
      const Scalar<DataVector>& energy_density_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field_ext,
      const Scalar<DataVector>& divergence_cleaning_field_ext,
      const Scalar<DataVector>& normal_dot_flux_mass_density_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_momentum_density_ext,
      const Scalar<DataVector>& normal_dot_flux_energy_density_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_magnetic_field_ext,
      const Scalar<DataVector>& normal_dot_flux_divergence_cleaning_field_ext,
      const Scalar<DataVector>& largest_outgoing_char_speed_ext,
      const Scalar<DataVector>& largest_ingoing_char_speed_ext,
      dg::Formulation dg_formulation) const;
};
}  // namespace NewtonianMhd::BoundaryCorrections
