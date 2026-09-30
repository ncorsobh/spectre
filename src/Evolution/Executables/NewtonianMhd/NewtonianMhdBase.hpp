// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/Conservative/UpdateConservatives.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/Actions/TciAndRollback.hpp"
#include "Evolution/DgSubcell/Actions/TciAndSwitchToDg.hpp"
#include "Evolution/DgSubcell/GetTciDecision.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/NeighborTciDecision.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/SetInterpolators.hpp"
#include "Evolution/DgSubcell/SubcellEqualRateRegion.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ObserverMesh.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/ChangeFixedLtsRatio.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/FixedLtsRatio.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/NonconformingEqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SetupEqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/NewtonianMhd/AllSolutions.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/Characteristics.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianMhd/FixConservatives.hpp"
#include "Evolution/Systems/NewtonianMhd/Initialization/BackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/OptionalBackgroundMagneticField.hpp"
#include "Evolution/Systems/NewtonianMhd/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianMhd/Sources/Factory.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/BackgroundMagneticFieldVars.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/PrimsAfterRollback.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/ResizeAndComputePrimitives.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/SetInitialRdmpData.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TciOnFdGrid.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TciOptions.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Evolution/VariableFixing/Actions.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Factory.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/SpectralFilter.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/ChangeFixedLtsRatio.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/ChangeStepSize.hpp"
#include "Time/ChangeTimeStepperOrder.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Time/UpdateU.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

/*!
 * \brief Metavariables for the Newtonian MHD system.
 *
 * `UseBackgroundMagneticField` picks how the initial magnetic field is divided
 * between the static background \f$B_0\f$ and the evolved perturbation
 * \f$B_1\f$: with it disabled the whole field is evolved, which is standard
 * MHD, while with it enabled only the perturbation is. The latter needs
 * initial data that supply a background field, so `InitialDataList` is
 * restricted accordingly.
 */
template <typename InitialDataList, bool UseBackgroundMagneticField>
struct NewtonianMhdMetavars {
  using metavariables = NewtonianMhdMetavars;
  static constexpr size_t volume_dim = 3;
  // Controls whether to use unlimited DG (false) or a DG-FD hybrid scheme
  // (true).
  static constexpr bool use_dg_subcell = true;

  using system = NewtonianMhd::System<UseBackgroundMagneticField>;

  using temporal_id = Tags::TimeStepId;

  static constexpr bool use_dg_element_collection = false;

  using initial_data_tag = evolution::initial_data::Tags::InitialData;
  using initial_data_list = InitialDataList;

  using equation_of_state_tag = hydro::Tags::EquationOfState<false, 2>;

  using analytic_variables_tags =
      typename system::primitive_variables_tag::tags_list;

  // Note that with background-field splitting the evolved magnetic field is
  // the perturbation, so its error against an analytic solution's total field
  // is not meaningful.
  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_variables_tags, use_dg_subcell, initial_data_list>;
  using error_compute = Tags::ErrorsCompute<analytic_variables_tags>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_variables_tags>;

  using observe_fields = tmpl::push_back<
      tmpl::append<
          typename system::variables_tag::tags_list,
          typename system::primitive_variables_tag::tags_list, error_tags,
          NewtonianMhd::background_magnetic_field_tag_list<
              NewtonianMhd::Tags::BackgroundMagneticFieldVolume<>,
              UseBackgroundMagneticField>,
          tmpl::conditional_t<use_dg_subcell,
                              tmpl::list<evolution::dg::subcell::Tags::
                                             TciStatusCompute<volume_dim>>,
                              tmpl::list<>>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              volume_dim, Frame::ElementLogical>,
          ::Events::Tags::ObserverCoordinatesCompute<volume_dim,
                                                     Frame::ElementLogical>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<volume_dim,
                                                                   Frame::Grid>,
          domain::Tags::Coordinates<volume_dim, Frame::Grid>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              volume_dim, Frame::Inertial>,
          domain::Tags::Coordinates<volume_dim, Frame::Inertial>>>;
  using non_tensor_compute_tags = tmpl::append<
      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Tags::ObserverMeshCompute<volume_dim>,
              evolution::dg::subcell::Tags::ObserverInverseJacobianCompute<
                  volume_dim, Frame::ElementLogical, Frame::Inertial>,
              evolution::dg::subcell::Tags::
                  ObserverJacobianAndDetInvJacobianCompute<
                      volume_dim, Frame::ElementLogical, Frame::Inertial>>,
          tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                     ::Events::Tags::ObserverInverseJacobianCompute<
                         volume_dim, Frame::ElementLogical, Frame::Inertial>,
                     ::Events::Tags::ObserverJacobianCompute<
                         volume_dim, Frame::ElementLogical, Frame::Inertial>,
                     ::Events::Tags::ObserverDetInvJacobianCompute<
                         Frame::ElementLogical, Frame::Inertial>>>,
      tmpl::list<analytic_compute, error_compute>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            NewtonianMhd::Sources::Source<UseBackgroundMagneticField>,
            NewtonianMhd::Sources::all_sources<UseBackgroundMagneticField>>,
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, non_tensor_compute_tags>,
                       Events::time_events<system>,
                       dg::Events::ChangeFixedLtsRatio<volume_dim>>>>,
        tmpl::pair<
            evolution::BoundaryCorrection,
            NewtonianMhd::BoundaryCorrections::standard_boundary_corrections<
                UseBackgroundMagneticField>>,
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<
            NewtonianMhd::BoundaryConditions::BoundaryCondition,
            NewtonianMhd::BoundaryConditions::standard_boundary_conditions<
                UseBackgroundMagneticField>>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::push_back<
                       StepChoosers::standard_slab_choosers<system>,
                       evolution::dg::StepChoosers::FixedLtsRatio<volume_dim>>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>,
        tmpl::pair<Filters::Filter<volume_dim,
                                   typename system::variables_tag::tags_list>,
                   Filters::all_filters<
                       volume_dim, typename system::variables_tag::tags_list>>>;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>>>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using equal_rate_regions = tmpl::flatten<
      tmpl::list<evolution::dg::NonconformingEqualRateRegions<volume_dim>,
                 tmpl::conditional_t<
                     use_dg_subcell,
                     evolution::dg::subcell::SubcellEqualRateRegion<volume_dim>,
                     tmpl::list<>>>>;

  // Sets the static background field on the active grid, then replaces the
  // initial data's total magnetic field by the evolved perturbation.
  using background_magnetic_field_actions = tmpl::conditional_t<
      UseBackgroundMagneticField,
      tmpl::list<Initialization::Actions::InitializeItems<tmpl::conditional_t<
                     use_dg_subcell,
                     NewtonianMhd::subcell::BackgroundMagneticFieldVars,
                     NewtonianMhd::Initialization::BackgroundMagneticField>>,
                 Actions::MutateApply<NewtonianMhd::Initialization::
                                          SubtractBackgroundMagneticField>>,
      tmpl::list<>>;

  // Re-evaluates the background field after the active grid has changed. The
  // perturbation is not touched: it is projected or reconstructed along with
  // the other evolved variables.
  using background_magnetic_field_update = tmpl::conditional_t<
      UseBackgroundMagneticField and use_dg_subcell,
      Actions::MutateApply<NewtonianMhd::subcell::BackgroundMagneticFieldVars>,
      tmpl::list<>>;

  using initialization_actions = tmpl::flatten<tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<metavariables, TimeStepper, false, true>,
          evolution::dg::Initialization::Domain<metavariables>,
          Initialization::TimeStepperHistory<system>>,
      Initialization::Actions::ConservativeSystem<system>,
      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Actions::SetSubcellGrid<volume_dim,
                                                              system, false>,
              Actions::MutateApply<evolution::dg::subcell::SetInterpolators<
                  volume_dim, NewtonianMhd::fd::Tags::Reconstructor>>,
              background_magnetic_field_actions, Actions::UpdateConservatives,
              evolution::dg::subcell::Actions::SetAndCommunicateInitialRdmpData<
                  volume_dim, NewtonianMhd::subcell::SetInitialRdmpData>,
              evolution::dg::subcell::Actions::ComputeAndSendTciOnInitialGrid<
                  volume_dim, system, NewtonianMhd::subcell::TciOnFdGrid>,
              evolution::dg::subcell::Actions::SetInitialGridFromTciData<
                  volume_dim, system>,
              Actions::MutateApply<
                  NewtonianMhd::subcell::ResizeAndComputePrims>,
              background_magnetic_field_update, Actions::UpdateConservatives>,
          tmpl::list<
              evolution::Initialization::Actions::SetVariables<
                  domain::Tags::Coordinates<volume_dim, Frame::ElementLogical>>,
              background_magnetic_field_actions, Actions::UpdateConservatives>>,
      Initialization::Actions::AddComputeTags<
          tmpl::list<NewtonianMhd::Tags::SoundSpeedSquaredCompute<DataVector>,
                     NewtonianMhd::Tags::FastMagnetosonicSpeedCompute<
                         UseBackgroundMagneticField>>>,
      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<metavariables>>,
      ::evolution::dg::Initialization::Mortars<volume_dim>,
      evolution::dg::Initialization::Actions::SetupEqualRateRegions<
          metavariables, volume_dim, equal_rate_regions>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Initialization::Actions::InitializeItems<
          evolution::dg::Initialization::SpectralFilters<
              volume_dim, typename system::variables_tag::tags_list>>,
      Parallel::Actions::TerminatePhase>>;

  using events_and_dense_triggers_postprocessors = tmpl::list<
      AlwaysReadyPostprocessor<typename system::primitive_from_conservative>>;

  using dg_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, use_dg_element_collection>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          volume_dim, use_dg_element_collection>,
      Actions::MutateApply<RecordTimeStepperData<system>>,
      evolution::Actions::RunEventsAndDenseTriggers<tmpl::push_front<
          events_and_dense_triggers_postprocessors,
          evolution::dg::ApplyLtsDenseBoundaryCorrections<metavariables>>>,
      Actions::MutateApply<UpdateU<system>>,
      evolution::dg::Actions::ApplyLtsBoundaryCorrections<
          volume_dim, use_dg_element_collection>,
      Actions::MutateApply<ChangeTimeStepperOrder<system>>,
      VariableFixing::Actions::FixVariables<NewtonianMhd::FixConservatives>,
      tmpl::conditional_t<
          use_dg_subcell,
          // The primitive variables are computed as part of the TCI.
          tmpl::list<evolution::dg::subcell::Actions::TciAndRollback<
                         NewtonianMhd::subcell::TciOnDgGrid>,
                     background_magnetic_field_update>,
          Actions::MutateApply<typename system::primitive_from_conservative>>,
      Actions::MutateApply<CleanHistory<system>>,
      Actions::MutateApply<evolution::dg::CleanMortarHistory<volume_dim>>,
      dg::Actions::SpectralFilter<volume_dim,
                                  typename system::variables_tag::tags_list>>>;

  struct SubcellOptions {
    static constexpr bool subcell_enabled = use_dg_subcell;
    static constexpr bool subcell_enabled_at_external_boundary = true;

    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<NewtonianMhd::fd::Tags::Reconstructor>(box)
          .ghost_zone_size();
    }

    using DgComputeSubcellNeighborPackagedData =
        NewtonianMhd::subcell::NeighborPackagedData<UseBackgroundMagneticField>;

    using GhostVariables = NewtonianMhd::subcell::PrimitiveGhostVariables;
  };

  using dg_subcell_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      dg_step_actions,
      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      // This is just to adjust for FixedLtsRatio, so we can pass an empty list
      // of StepChoosers.
      Actions::MutateApply<ChangeStepSize<tmpl::list<>>>,
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim, NewtonianMhd::subcell::PrimitiveGhostVariables,
          use_dg_element_collection>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,
      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,
      Actions::MutateApply<NewtonianMhd::subcell::PrimsAfterRollback>,
      background_magnetic_field_update,
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          NewtonianMhd::subcell::TimeDerivative>,
      Actions::MutateApply<RecordTimeStepperData<system>>,
      evolution::Actions::RunEventsAndDenseTriggers<
          events_and_dense_triggers_postprocessors>,
      Actions::MutateApply<UpdateU<system>>,
      Actions::MutateApply<CleanHistory<system>>,
      Actions::MutateApply<evolution::dg::CleanMortarHistory<volume_dim>>,
      VariableFixing::Actions::FixVariables<NewtonianMhd::FixConservatives>,
      Actions::MutateApply<typename system::primitive_from_conservative>,
      evolution::dg::subcell::Actions::TciAndSwitchToDg<
          NewtonianMhd::subcell::TciOnFdGrid>,
      Actions::MutateApply<NewtonianMhd::subcell::ResizeAndComputePrims>,
      background_magnetic_field_update,

      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  using step_actions =
      tmpl::conditional_t<use_dg_subcell, dg_subcell_step_actions,
                          dg_step_actions>;

  using dg_element_array = DgElementArray<
      metavariables,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<Parallel::Phase::Restart,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::WriteCheckpoint,
              tmpl::list<evolution::Actions::RunEventsAndTriggers<
                             Triggers::WhenToCheck::AtCheckpoints>,
                         Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::flatten<
                  tmpl::list<evolution::Actions::RunEventsAndTriggers<
                                 Triggers::WhenToCheck::AtSteps>,
                             evolution::Actions::RunEventsAndTriggers<
                                 Triggers::WhenToCheck::AtSlabs>,
                             Actions::ChangeSlabSize,
                             evolution::dg::Actions::ChangeFixedLtsRatio,
                             step_actions, Actions::MutateApply<AdvanceTime<>>,
                             PhaseControl::Actions::ExecutePhaseChange>>>>>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array, dg_registration_list>>;
  };

  using component_list =
      tmpl::list<observers::Observer<metavariables>,
                 observers::ObserverWriter<metavariables>, dg_element_array>;

  using const_global_cache_tags = tmpl::push_back<
      tmpl::conditional_t<use_dg_subcell,
                          tmpl::list<NewtonianMhd::fd::Tags::Reconstructor,
                                     NewtonianMhd::subcell::Tags::TciOptions>,
                          tmpl::list<>>,
      initial_data_tag, equation_of_state_tag,
      NewtonianMhd::Tags::SourceTerm<UseBackgroundMagneticField>,
      NewtonianMhd::Tags::DivergenceCleaningSpeed,
      NewtonianMhd::Tags::ConstraintDampingParameter,
      ::Tags::VariableFixer<NewtonianMhd::FixConservatives>>;

  static constexpr Options::String help{
      "Evolve the Newtonian MHD system in conservative form.\n\n"};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
