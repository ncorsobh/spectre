# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianMhdAnalyticData")

set(LIBRARY_SOURCES
  Test_BrioWu.cpp
  Test_ConductorFlow.cpp
  Test_OrszagTangVortex.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Framework
  Hydro
  NewtonianMhd
  NewtonianMhdAnalyticData
  Options
  Utilities
  )
