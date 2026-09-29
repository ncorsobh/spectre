# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianMhd)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  ConservativeFromPrimitive.cpp
  Fluxes.cpp
  PrimitiveFromConservative.cpp
  SoundSpeedSquared.cpp
  TimeDerivativeTerms.cpp
  SpectralFilter.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AllSolutions.hpp
  OptionalBackgroundMagneticField.hpp
  Characteristics.hpp
  ConservativeFromPrimitive.hpp
  Fluxes.hpp
  PrimitiveFromConservative.hpp
  SoundSpeedSquared.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivativeTerms.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DgSubcell
  Domain
  ErrorHandling
  Events
  Evolution
  Hydro
  InitialDataUtilities
  Options
  Serialization
  PRIVATE
  FiniteDifference
  NewtonianMhdAnalyticData
  NewtonianMhdSolutions
  NewtonianMhdSources
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Initialization)
add_subdirectory(Instantiations)
add_subdirectory(Sources)
