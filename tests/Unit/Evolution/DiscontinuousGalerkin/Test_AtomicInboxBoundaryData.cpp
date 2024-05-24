// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/AtomicInboxBoundaryData.hpp"

namespace evolution::dg {
namespace {
void test_3d_helper(const Direction<3>& dir) {
  constexpr size_t Dim = 3;
  const size_t offset =
      (dir.side() == Side::Lower ? 0 : 4) + 8 * dir.dimension();
  CAPTURE(offset);
  const auto swap_segments = [&dir](std::array<SegmentId, Dim> segment_ids)
      -> std::array<SegmentId, Dim> {
    if (dir.dimension() == 0) {
      return segment_ids;
    } else if (dir.dimension() == 1) {
      return {{segment_ids[1], segment_ids[0], segment_ids[2]}};
    } else {
      return {{segment_ids[1], segment_ids[2], segment_ids[0]}};
    }
  };
  for (size_t i = 0; i < 4; ++i) {
    // Loop over the grid index in the dimension normal to the interface.

    // 4 neighbors
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 0}, {2, 0}}})}}) ==
          0 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 1}, {2, 0}}})}}) ==
          1 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 0}, {2, 1}}})}}) ==
          2 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 1}, {2, 1}}})}}) ==
          3 + offset);

    // 3 neighbors
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 0}, {2, 0}}})}}) ==
          0 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 0}, {2, 1}}})}}) ==
          2 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 1}, {1, 0}}})}}) ==
          1 + offset);

    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 0}, {1, 0}}})}}) ==
          0 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 1}, {2, 0}}})}}) ==
          1 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 1}, {2, 1}}})}}) ==
          3 + offset);

    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 0}, {2, 0}}})}}) ==
          0 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 1}, {2, 0}}})}}) ==
          1 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {1, 0}, {2, 1}}})}}) ==
          2 + offset);

    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {1, 0}, {2, 0}}})}}) ==
          0 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 0}, {2, 1}}})}}) ==
          2 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 1}, {2, 1}}})}}) ==
          3 + offset);

    // 2 neighbors
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 0}, {1, 0}}})}}) ==
          0 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {2, 1}, {1, 0}}})}}) ==
          1 + offset);

    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {1, 0}, {2, 0}}})}}) ==
          0 + offset);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {1, 0}, {2, 1}}})}}) ==
          2 + offset);

    // 1 neighbor
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              dir,
              ElementId<Dim>{2, swap_segments({{{2, i}, {1, 0}, {1, 0}}})}}) ==
          0 + offset);
  }
}

template <size_t Dim>
void test() {
  static_assert(Dim < 4);
  static_assert(evolution::dg::is_atomic_inbox_boundary_data_v<
                AtomicInboxBoundaryData<Dim>>);
  CAPTURE(Dim);
  if constexpr (Dim == 1) {
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::lower_xi(), ElementId<Dim>{2}}) == 0);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::upper_xi(), ElementId<Dim>{2}}) == 1);
  } else if constexpr (Dim == 2) {
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::lower_xi(), ElementId<Dim>{2}}) == 0);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::upper_xi(), ElementId<Dim>{2}}) == 2);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::lower_eta(), ElementId<Dim>{2}}) == 4 + 0);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::upper_eta(), ElementId<Dim>{2}}) == 4 + 2);

    for (size_t i = 0; i < 4; ++i) {
      // Loop over the grid index in the dimension normal to the interface.
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::lower_xi(),
                ElementId<Dim>{2, {{{2, i}, {1, 0}}}}}) == 0);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::lower_xi(),
                ElementId<Dim>{2, {{{2, i}, {2, 0}}}}}) == 0);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::lower_xi(),
                ElementId<Dim>{2, {{{2, i}, {2, 1}}}}}) == 1);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::lower_xi(),
                ElementId<Dim>{2, {{{2, i}, {2, 2}}}}}) == 0);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::lower_xi(),
                ElementId<Dim>{2, {{{2, i}, {2, 3}}}}}) == 1);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::upper_xi(),
                ElementId<Dim>{2, {{{2, i}, {1, 0}}}}}) == 2 + 0);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::upper_xi(),
                ElementId<Dim>{2, {{{2, i}, {2, 0}}}}}) == 2 + 0);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::upper_xi(),
                ElementId<Dim>{2, {{{2, i}, {2, 1}}}}}) == 2 + 1);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::upper_xi(),
                ElementId<Dim>{2, {{{2, i}, {2, 2}}}}}) == 2 + 0);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::upper_xi(),
                ElementId<Dim>{2, {{{2, i}, {2, 3}}}}}) == 2 + 1);

      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::lower_eta(),
                ElementId<Dim>{2, {{{2, 0}, {2, i}}}}}) == 4 + 0);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::lower_eta(),
                ElementId<Dim>{2, {{{2, 1}, {2, i}}}}}) == 4 + 1);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::lower_eta(),
                ElementId<Dim>{2, {{{2, 2}, {2, i}}}}}) == 4 + 0);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::lower_eta(),
                ElementId<Dim>{2, {{{2, 3}, {2, i}}}}}) == 4 + 1);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::upper_eta(),
                ElementId<Dim>{2, {{{2, 0}, {2, i}}}}}) == 4 + 2 + 0);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::upper_eta(),
                ElementId<Dim>{2, {{{2, 1}, {2, i}}}}}) == 4 + 2 + 1);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::upper_eta(),
                ElementId<Dim>{2, {{{2, 2}, {2, i}}}}}) == 4 + 2 + 0);
      CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
                Direction<Dim>::upper_eta(),
                ElementId<Dim>{2, {{{2, 3}, {2, i}}}}}) == 4 + 2 + 1);
    }
  } else {
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::lower_xi(), ElementId<Dim>{2}}) == 0);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::upper_xi(), ElementId<Dim>{2}}) == 4);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::lower_eta(), ElementId<Dim>{2}}) == 8 + 0);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::upper_eta(), ElementId<Dim>{2}}) == 8 + 4);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::lower_zeta(), ElementId<Dim>{2}}) == 16 + 0);
    CHECK(AtomicInboxBoundaryData<Dim>::index(DirectionalId<Dim>{
              Direction<Dim>::upper_zeta(), ElementId<Dim>{2}}) == 16 + 4);

    test_3d_helper(Direction<Dim>::lower_xi());
    test_3d_helper(Direction<Dim>::upper_xi());
    test_3d_helper(Direction<Dim>::lower_eta());
    test_3d_helper(Direction<Dim>::upper_eta());
    test_3d_helper(Direction<Dim>::lower_zeta());
    test_3d_helper(Direction<Dim>::upper_zeta());
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.AtomicInboxBoundaryData",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace evolution::dg