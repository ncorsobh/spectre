# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianMhdAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BrioWu.cpp
  OrszagTangVortex.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BrioWu.hpp
  OrszagTangVortex.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Hydro
  InitialDataUtilities
  NewtonianMhd
  Options
  Serialization
  Utilities
  )
