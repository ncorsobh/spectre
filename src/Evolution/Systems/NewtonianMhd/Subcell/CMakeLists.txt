# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  NeighborPackagedData.cpp
  PrimitiveGhostData.cpp
  PrimsAfterRollback.cpp
  ResizeAndComputePrimitives.cpp
  SetInitialRdmpData.cpp
  TciOnDgGrid.cpp
  TciOnFdGrid.cpp
  TciOptions.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BackgroundMagneticFieldVars.hpp
  ComputeFluxes.hpp
  NeighborPackagedData.hpp
  PrimitiveGhostData.hpp
  PrimsAfterRollback.hpp
  ResizeAndComputePrimitives.hpp
  SetInitialRdmpData.hpp
  Subcell.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  TciOptions.hpp
  TimeDerivative.hpp
  )
