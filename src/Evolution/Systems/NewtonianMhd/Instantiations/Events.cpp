// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define USE_BG(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class Events::ObserveTimeStep<NewtonianMhd::System<USE_BG(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (true, false))

#undef INSTANTIATION
#undef USE_BG
