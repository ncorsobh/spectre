# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianMhdSources)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DampingZone.cpp
  NoSource.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DampingZone.hpp
  Factory.hpp
  NoSource.hpp
  Source.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  Utilities
  )
