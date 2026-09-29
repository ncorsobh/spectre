# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianMhd")

set(LIBRARY_SOURCES
  BoundaryConditions/Test_BoundaryConditions.cpp
  BoundaryCorrections/Test_BoundaryCorrections.cpp
  FiniteDifference/Test_AoWeno.cpp
  FiniteDifference/Test_MonotonisedCentral.cpp
  FiniteDifference/Test_Tag.cpp
  Sources/Test_Sources.cpp
  Test_Characteristics.cpp
  Test_ConservativeFromPrimitive.cpp
  Test_Fluxes.cpp
  Test_SoundSpeedSquared.cpp
  Test_System.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainBoundaryConditionsHelpers
  Framework
  Hydro
  NewtonianMhd
  NewtonianMhdSources
  Utilities
  )
