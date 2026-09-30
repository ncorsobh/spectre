// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianMhd::BoundaryConditions {
/*!
 * \brief A boundary condition that verifies that the fluid characteristic
 * speeds are directed out of the domain; no boundary data is altered.
 *
 * The check is \f$v^in_i - c_f \ge 0\f$, with \f$c_f\f$ the fast magnetosonic
 * speed built from the total magnetic field \f$B_0 + B_1\f$.
 *
 * \warning The GLM divergence-cleaning waves travel at \f$\pm c_h\f$ and are
 * therefore *always* partly ingoing; they are deliberately excluded from the
 * check, which would otherwise be impossible to satisfy for any \f$c_h > 0\f$.
 * The \f$\psi\f$ and normal-\f$B_1\f$ data flowing in through the boundary are
 * whatever the interior flux supplies. This is only appropriate when the
 * solution is driven to a clean, uniform state before the boundary is reached,
 * e.g. by a damping zone.
 */
template <size_t Dim, bool UseBackgroundMagneticField = false>
class DemandOutgoingCharSpeeds final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "A boundary condition that only verifies the fluid characteristic speeds "
      "are all directed out of the domain. The divergence-cleaning waves are "
      "not checked."};

  DemandOutgoingCharSpeeds() = default;
  DemandOutgoingCharSpeeds(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds& operator=(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds(const DemandOutgoingCharSpeeds&) = default;
  DemandOutgoingCharSpeeds& operator=(const DemandOutgoingCharSpeeds&) =
      default;
  ~DemandOutgoingCharSpeeds() override = default;

  explicit DemandOutgoingCharSpeeds(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DemandOutgoingCharSpeeds);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::DemandOutgoingCharSpeeds;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::MagneticFieldCons<Dim>>;
  using dg_interior_temporary_tags =
      background_magnetic_field_tag_list<Tags::BackgroundMagneticField<Dim>,
                                         UseBackgroundMagneticField>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>>;
  using dg_gridless_tags = tmpl::list<hydro::Tags::EquationOfState<false, 2>>;

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::append<tmpl::list<evolution::dg::subcell::Tags::Mesh<Dim>>,
                   background_magnetic_field_tag_list<
                       Tags::BackgroundMagneticFieldVolume<Dim>,
                       UseBackgroundMagneticField>>;
  using fd_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::MagneticField<DataVector, Dim>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;
  using fd_gridless_tags = tmpl::list<hydro::Tags::EquationOfState<false, 2>,
                                      fd::Tags::Reconstructor<Dim>>;

  /// \brief Copies the outermost cells into the ghost zone, after checking
  /// that no fluid characteristic enters the domain.
  ///
  /// This is a piecewise-constant (lowest-order) reconstruction at the
  /// external boundary. As in `dg_demand_outgoing_char_speeds`, only the
  /// fluid speed \f$v_n - c_f\f$ is checked: the GLM cleaning waves travel at
  /// \f$\pm c_h\f$ whatever the state, so one of them always enters and
  /// requiring otherwise would reject every boundary.
  static void fd_demand_outgoing_char_speeds(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

      const Direction<Dim>& direction,

      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          outward_directed_normal_covector,

      // fd_interior_temporary_tags
      const Mesh<Dim>& subcell_mesh,
      BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
          interior_background_magnetic_field,

      // fd_interior_primitive_variables_tags
      const Scalar<DataVector>& interior_mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const Scalar<DataVector>& interior_pressure,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_magnetic_field,
      const Scalar<DataVector>& interior_divergence_cleaning_field,

      // fd_gridless_tags
      const EquationsOfState::EquationOfState<false, 2>& equation_of_state,
      const fd::Reconstructor<Dim>& reconstructor);

  /// \brief Overload selected when the background-field splitting is
  /// disabled, where `fd_interior_temporary_tags` holds no \f$B_0\f$.
  static void fd_demand_outgoing_char_speeds(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

      const Direction<Dim>& direction,

      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          outward_directed_normal_covector,

      const Mesh<Dim>& subcell_mesh,

      const Scalar<DataVector>& interior_mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const Scalar<DataVector>& interior_pressure,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_magnetic_field,
      const Scalar<DataVector>& interior_divergence_cleaning_field,

      const EquationsOfState::EquationOfState<false, 2>& equation_of_state,
      const fd::Reconstructor<Dim>& reconstructor);

  /// @{
  /// The background-field overload is selected by
  /// `dg_interior_temporary_tags`: with the splitting disabled that list is
  /// empty, so the framework calls the shorter one and \f$B_0\f$ is never
  /// projected onto element faces.
  template <size_t ThermodynamicDim>
  static std::optional<std::string> dg_demand_outgoing_char_speeds(
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          outward_directed_normal_covector,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& magnetic_field,
      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
      const Scalar<DataVector>& specific_internal_energy,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state);

  template <size_t ThermodynamicDim>
  static std::optional<std::string> dg_demand_outgoing_char_speeds(
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          outward_directed_normal_covector,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& magnetic_field,
      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
      const Scalar<DataVector>& specific_internal_energy,
      BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
          background_magnetic_field,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state);
  /// @}
};
}  // namespace NewtonianMhd::BoundaryConditions
