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
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace NewtonianMhd::BoundaryConditions {
/*!
 * \brief Sets Dirichlet boundary conditions from the analytic solution or
 * analytic data.
 *
 * The prescription reports the total physical magnetic field, so the ghost
 * perturbation is \f$B_1 = B - B_0\f$ with \f$B_0\f$ copied from the interior
 * face. Taking \f$B_0\f$ from the interior rather than re-evaluating it keeps
 * this boundary condition agnostic to whether the executable splits off a
 * background field at all.
 */
template <size_t Dim, bool UseBackgroundMagneticField = false>
class DirichletAnalytic final : public BoundaryCondition<Dim> {
 public:
  /// \brief What analytic solution/data to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };
  using options = tmpl::list<AnalyticPrescription>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions using either an analytic solution "
      "or analytic data."};

  DirichletAnalytic() = default;
  DirichletAnalytic(DirichletAnalytic&&) = default;
  DirichletAnalytic& operator=(DirichletAnalytic&&) = default;
  DirichletAnalytic(const DirichletAnalytic&);
  DirichletAnalytic& operator=(const DirichletAnalytic&);
  ~DirichletAnalytic() override = default;

  explicit DirichletAnalytic(CkMigrateMessage* msg);

  explicit DirichletAnalytic(
      std::unique_ptr<evolution::initial_data::InitialData>
          analytic_prescription);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletAnalytic);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::append<
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>,
      background_magnetic_field_tag_list<Tags::BackgroundMagneticField<Dim>,
                                         UseBackgroundMagneticField>>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, Tags::DivergenceCleaningSpeed>;

  /// @{
  /// The background-field overload is selected by the boundary correction's
  /// `dg_package_data_temporary_tags`: with the splitting disabled that list
  /// is empty, so neither the ghost nor the interior \f$B_0\f$ exists.
  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          magnetic_field_cons,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,

      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_mass_density,
      gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_momentum_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_energy_density,
      gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_magnetic_field,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_divergence_cleaning_field,

      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

      const std::optional<
          tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords, double time,
      double divergence_cleaning_speed) const;

  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          magnetic_field_cons,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field_cons,

      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_mass_density,
      gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_momentum_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_energy_density,
      gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_magnetic_field,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_divergence_cleaning_field,

      BackgroundMagneticFieldOutput<Dim, UseBackgroundMagneticField>
          background_magnetic_field,

      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

      const std::optional<
          tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      BackgroundMagneticFieldArgument<Dim, UseBackgroundMagneticField>
          interior_background_magnetic_field,
      double time, double divergence_cleaning_speed) const;
  /// @}

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
};
}  // namespace NewtonianMhd::BoundaryConditions
