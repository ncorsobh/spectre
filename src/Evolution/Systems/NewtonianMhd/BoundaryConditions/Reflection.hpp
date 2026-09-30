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
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianMhd::BoundaryConditions {

namespace detail {
/// \brief Shared ghost-state construction for `Reflection` and
/// `ConductorReflection`.
///
/// `no_slip` selects whether the tangential velocity is kept (free-slip,
/// `Reflection`) or also reversed (`ConductorReflection`).
template <bool UseBackgroundMagneticField = false>
void reflection_dg_ghost(
    gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> momentum_density,
    gsl::not_null<Scalar<DataVector>*> energy_density,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field_cons,
    gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> flux_mass_density,
    gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
        flux_momentum_density,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> flux_energy_density,
    gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
        flux_magnetic_field,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        flux_divergence_cleaning_field,
    BackgroundMagneticFieldOutput<UseBackgroundMagneticField>
        background_magnetic_field,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
    gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_pressure,
    double divergence_cleaning_speed, bool no_slip,
    BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
        interior_background_magnetic_field);

/// \brief Shared finite-difference ghost-zone fill for `Reflection` and
/// `ConductorReflection`.
///
/// The cells adjacent to the boundary are mirrored into every ghost cell,
/// reversing the normal component of the velocity (the whole velocity when
/// `no_slip`) and of the magnetic field, and reversing \f$\psi\f$, so that the
/// reconstructed interface values carry the same conditions the DG ghost state
/// imposes.
void reflection_fd_ghost(
    gsl::not_null<Scalar<DataVector>*> mass_density,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
    gsl::not_null<Scalar<DataVector>*> pressure,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
    gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    const Direction<3>& direction, const Mesh<3>& subcell_mesh,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_pressure,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    size_t ghost_zone_size, bool no_slip);
}  // namespace detail

/*!
 * \brief Reflecting (free-slip, perfectly conducting) boundary conditions for
 * Newtonian MHD.
 *
 * Ghost (exterior) data 'mirrors' interior volume data with respect to the
 * boundary interface: the normal component of the velocity and of the evolved
 * magnetic field is reversed while tangential components and scalars are
 * copied.  Taking the mesh velocity \f$v_m\f$ into account,
 *
 * \f{align*}
 * v_\text{ghost}^i &= v_\text{int}^i - 2[(v_\text{int}^j-v_m^j)n_j]n^i \\
 * B_{1,\text{ghost}}^i &= B_{1,\text{int}}^i - 2(B_{1,\text{int}}^jn_j)n^i \\
 * \psi_\text{ghost} &= -\psi_\text{int} ,
 * \f}
 *
 * with \f$\rho\f$, \f$\epsilon\f$ and \f$P\f$ copied unchanged.  Reversing the
 * normal component of \f$B_1\f$ makes the interface value of \f$B_1^in_i\f$
 * vanish, and the anti-symmetric \f$\psi\f$ is what the GLM subsystem needs
 * for \f$B^in_i = 0\f$ to be preserved by the cleaning waves.
 *
 * \note This condition is only admissible where the *total* normal field
 * vanishes.  A perfect conductor requires the tangential electric field to
 * vanish in its rest frame, and in ideal MHD \f$E = -v\times B\f$ gives
 *
 * \f{align*}
 *   E_t = -v_n\,(n\times B_t) + B_n\,(n\times v_t) ,
 * \f}
 *
 * which at an impenetrable wall (\f$v_n = 0\f$) leaves \f$B_n(n\times v_t)\f$.
 * Free slip keeps \f$v_t\f$ arbitrary, so it is consistent only when
 * \f$B_n = 0\f$; a wall threaded by normal flux instead needs
 * \f$v_t = 0\f$, which is what `ConductorReflection` imposes.
 *
 * The reflection drives \f$B_1^in_i\f$ to zero, so \f$B^in_i = B_0^in_i\f$
 * at the interface.  With the background-field splitting enabled this class
 * therefore checks that \f$B_0^in_i\f$ vanishes and errors out otherwise,
 * rather than assuming a particular background.
 *
 * The background field \f$B_0\f$ is copied to the exterior unchanged: it is
 * smooth and is only used by the boundary correction to evaluate wave speeds.
 */
template <bool UseBackgroundMagneticField = false>
class Reflection final : public BoundaryCondition {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Free-slip reflecting boundary conditions for Newtonian MHD."};

  Reflection() = default;
  Reflection(Reflection&&) = default;
  Reflection& operator=(Reflection&&) = default;
  Reflection(const Reflection&) = default;
  Reflection& operator=(const Reflection&) = default;
  ~Reflection() override = default;

  explicit Reflection(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Reflection);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::MagneticFieldCons<>, Tags::DivergenceCleaningFieldCons>;
  using dg_interior_temporary_tags =
      background_magnetic_field_tag_list<Tags::BackgroundMagneticField<>,
                                         UseBackgroundMagneticField>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>>;
  using dg_gridless_tags = tmpl::list<Tags::DivergenceCleaningSpeed>;

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;
  using fd_gridless_tags = tmpl::list<fd::Tags::Reconstructor>;

  void fd_ghost(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

      const Direction<3>& direction,

      // interior temporary tags
      const Mesh<3>& subcell_mesh,

      // interior primitive variables tags
      const Scalar<DataVector>& interior_mass_density,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,
      const Scalar<DataVector>& interior_pressure,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
      const Scalar<DataVector>& interior_divergence_cleaning_field,

      // gridless tags
      const fd::Reconstructor& reconstructor) const;

  /// @{
  /// The background-field overload is selected by the boundary correction's
  /// `dg_package_data_temporary_tags`: with the splitting disabled that list
  /// is empty, so neither the ghost nor the interior \f$B_0\f$ exists.
  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          magnetic_field_cons,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> flux_mass_density,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
          flux_momentum_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          flux_energy_density,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
          flux_magnetic_field,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          flux_divergence_cleaning_field,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          outward_directed_normal_covector,

      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
      const Scalar<DataVector>& interior_divergence_cleaning_field,
      const Scalar<DataVector>& interior_mass_density,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const Scalar<DataVector>& interior_pressure,
      double divergence_cleaning_speed) const;

  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          magnetic_field_cons,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> flux_mass_density,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
          flux_momentum_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          flux_energy_density,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
          flux_magnetic_field,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          flux_divergence_cleaning_field,

      BackgroundMagneticFieldOutput<UseBackgroundMagneticField>
          background_magnetic_field,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          outward_directed_normal_covector,

      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
      const Scalar<DataVector>& interior_divergence_cleaning_field,
      const Scalar<DataVector>& interior_mass_density,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_velocity,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const Scalar<DataVector>& interior_pressure,
      BackgroundMagneticFieldArgument<UseBackgroundMagneticField>
          interior_background_magnetic_field,
      double divergence_cleaning_speed) const;
  /// @}
};

}  // namespace NewtonianMhd::BoundaryConditions
