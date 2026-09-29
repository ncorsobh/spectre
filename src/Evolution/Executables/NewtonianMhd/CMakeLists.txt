# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Actions
  Charmxx::main
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  FiniteDifference
  Hydro
  Informer
  LinearOperators
  MathFunctions
  NewtonianMhd
  NewtonianMhdAnalyticData
  NewtonianMhdSolutions
  NewtonianMhdSources
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  )

function(add_newtonian_mhd_executable EXECUTABLE)
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    ${EXECUTABLE}.cpp
    VolumeTermsInstantiation3D.cpp
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_newtonian_mhd_executable(EvolveNewtonianMhd)
add_newtonian_mhd_executable(EvolveNewtonianMhdBgField)
