# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GhValenciaDivClean)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ApplyTensorYlmFilter.cpp
  StressEnergy.cpp
  TimeDerivativeTerms.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AllSolutions.hpp
  ApplyTensorYlmFilter.hpp
  Characteristics.hpp
  Constraints.hpp
  NeutrinoSystems.hpp
  SetPiAndPhiFromConstraints.hpp
  StressEnergy.hpp
  System.hpp
  SystemM1.hpp
  Tags.hpp
  TimeDerivativeTerms.hpp
  WaveZone.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  DomainBoundaryConditions
  DomainCreators
  DomainStructure
  FiniteDifference
  GeneralRelativity
  GeneralizedHarmonic
  GhGrMhdAnalyticData
  GhGrMhdSolutions
  GhRelativisticEulerSolutions
  Hydro
  LinearOperators
  MonteCarloSolutions
  Serialization
  Utilities
  ValenciaDivClean
  INTERFACE
  Options
  Parallel
  )

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Instantiations)
add_subdirectory(Subcell)
