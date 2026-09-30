// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Time/ChangeTimeStepperOrder.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define USE_BG(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                               \
  template class ChangeTimeStepperOrder<NewtonianMhd::System<USE_BG(data)>>; \
  template class CleanHistory<NewtonianMhd::System<USE_BG(data)>>;           \
  template class RecordTimeStepperData<NewtonianMhd::System<USE_BG(data)>>;  \
  template class UpdateU<NewtonianMhd::System<USE_BG(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (true, false))

#undef INSTANTIATION
#undef USE_BG
