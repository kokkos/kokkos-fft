// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <random>
#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_transpose.hpp"
#include "Test_Types.hpp"
#include "Test_Utils.hpp"

template <typename InViewType, typename OutViewType>
void test_managed_prep_transpose_view() {
  constexpr std::size_t DIMS = InViewType::rank();
  static_assert(InViewType::rank() == OutViewType::rank(),
                "input and output have different ranks");

  using InManagedViewType =
      typename KokkosFFT::Impl::managable_view_type<InViewType>::type;
  using OutManagedViewType =
      typename KokkosFFT::Impl::managable_view_type<OutViewType>::type;
  static_assert(!InManagedViewType::memory_traits::is_unmanaged,
                "Unable to get managed input view type");
  static_assert(!OutManagedViewType::memory_traits::is_unmanaged,
                "Unable to get managed output view type");

  static_assert(!InViewType::memory_traits::is_unmanaged,
                "Unable to get managed input view type");
  static_assert(!OutViewType::memory_traits::is_unmanaged,
                "Unable to get managed output view type");
  using LayoutType = typename InViewType::array_layout;

  // no need to allocate
  {
    LayoutType layout;
    KokkosFFT::axis_type<DIMS> map;
    for (int i = 0; i < DIMS; ++i) {
      layout.dimension[i] = 5;
      map[i]              = i;
    }
    InViewType in("in", layout);
    OutViewType out("out", layout);
    auto data_prev = out.data();
    KokkosFFT::Impl::_prep_transpose_view(in, out, map);
    // ensure no allocation
    EXPECT_EQ(data_prev, out.data());
    // check shape
    for (int i = 0; i < DIMS; ++i) {
      EXPECT_EQ(out.extent(i), 5);
    }
  }
  // allocate
  {
    LayoutType layout;
    KokkosFFT::axis_type<DIMS> map;
    for (int i = 0; i < DIMS; ++i) {
      layout.dimension[i] = 5;
      map[i]              = i;
    }
    InViewType in("in", layout);
    OutViewType out;
    KokkosFFT::Impl::_prep_transpose_view(in, out, map);
    // check shape
    for (int i = 0; i < DIMS; ++i) {
      EXPECT_EQ(out.extent(i), 5);
    }
  }
}

template <typename InViewType, typename OutViewType>
void test_unmanaged_prep_transpose_view() {
  constexpr std::size_t DIMS = InViewType::rank();
  static_assert(InViewType::rank() == OutViewType::rank(),
                "input and output have different ranks");

  using InManagedViewType =
      typename KokkosFFT::Impl::managable_view_type<InViewType>::type;
  using OutManagedViewType =
      typename KokkosFFT::Impl::managable_view_type<OutViewType>::type;
  static_assert(!InManagedViewType::memory_traits::is_unmanaged,
                "Unable to get managed input view type");
  static_assert(!OutManagedViewType::memory_traits::is_unmanaged,
                "Unable to get managed output view type");
  using LayoutType = typename InViewType::array_layout;

  // no need to reshape
  {
    LayoutType layout;
    KokkosFFT::axis_type<DIMS> map;
    for (int i = 0; i < DIMS; ++i) {
      layout.dimension[i] = 5;
      map[i]              = i;
    }
    InManagedViewType in("in", layout);
    OutManagedViewType out("out", layout);
    OutViewType u_out(out.data(), layout);
    auto data_prev = out.data();
    KokkosFFT::Impl::_prep_transpose_view(in, u_out, map);
    EXPECT_EQ(data_prev, u_out.data());
    // check shape
    for (int i = 0; i < DIMS; ++i) {
      EXPECT_EQ(u_out.extent(i), 5);
    }
  }
  // reshape success
  {
    LayoutType layout;
    KokkosFFT::axis_type<DIMS> map;
    for (int i = 0; i < DIMS; ++i) {
      layout.dimension[i] = 5;
      map[i]              = i;
    }

    LayoutType layout_orig;
    layout_orig.dimension[0] = 5;
    if (DIMS == 0) {
      // give the 1D version a larger original shape so it will reshape down to
      // a portion of the allocation
      layout_orig.dimension[0] = 10;
    }
    for (int i = 1; i < DIMS; ++i) {
      layout_orig.dimension[0] *= 5;
      layout_orig.dimension[i] = 1;
    }
    InManagedViewType in("in", layout);
    OutManagedViewType out("out", layout_orig);
    OutViewType u_out(out.data(), layout_orig);
    KokkosFFT::Impl::_prep_transpose_view(in, u_out, map);
    // check shape
    for (int i = 0; i < DIMS; ++i) {
      EXPECT_EQ(u_out.extent(i), 5);
    }
  }
  // reshape failure
  {
    LayoutType layout;
    KokkosFFT::axis_type<DIMS> map;
    for (int i = 0; i < DIMS; ++i) {
      layout.dimension[i] = 5;
      map[i]              = i;
    }

    LayoutType layout_orig;
    layout_orig.dimension[0] = 1;
    InManagedViewType in("in", layout);
    OutManagedViewType out("out", layout_orig);
    OutViewType u_out(out.data(), layout_orig);
    EXPECT_THROW(KokkosFFT::Impl::_prep_transpose_view(in, u_out, map),
                 std::runtime_error);
  }
}

TEST(prep_transpose_view, 1DManaged) {
  test_managed_prep_transpose_view<
      Kokkos::View<double*, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double*, Kokkos::DefaultExecutionSpace>>();
}

TEST(prep_transpose_view, 1DUnmanaged) {
  test_unmanaged_prep_transpose_view<
      Kokkos::View<double*, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double*, Kokkos::DefaultExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>>();
}

TEST(prep_transpose_view, 2DManaged) {
  test_managed_prep_transpose_view<
      Kokkos::View<double**, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double**, Kokkos::DefaultExecutionSpace>>();
}

TEST(prep_transpose_view, 2DUnmanaged) {
  test_unmanaged_prep_transpose_view<
      Kokkos::View<double**, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double**, Kokkos::DefaultExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>>();
}

TEST(prep_transpose_view, 3DManaged) {
  test_managed_prep_transpose_view<
      Kokkos::View<double***, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double***, Kokkos::DefaultExecutionSpace>>();
}

TEST(prep_transpose_view, 3DUnmanaged) {
  test_unmanaged_prep_transpose_view<
      Kokkos::View<double***, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double***, Kokkos::DefaultExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>>();
}

TEST(prep_transpose_view, 4DManaged) {
  test_managed_prep_transpose_view<
      Kokkos::View<double****, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double****, Kokkos::DefaultExecutionSpace>>();
}

TEST(prep_transpose_view, 4DUnmanaged) {
  test_unmanaged_prep_transpose_view<
      Kokkos::View<double****, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double****, Kokkos::DefaultExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>>();
}

TEST(prep_transpose_view, 5DManaged) {
  test_managed_prep_transpose_view<
      Kokkos::View<double*****, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double*****, Kokkos::DefaultExecutionSpace>>();
}

TEST(prep_transpose_view, 5DUnmanaged) {
  test_unmanaged_prep_transpose_view<
      Kokkos::View<double*****, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double*****, Kokkos::DefaultExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>>();
}

TEST(prep_transpose_view, 6DManaged) {
  test_managed_prep_transpose_view<
      Kokkos::View<double******, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double******, Kokkos::DefaultExecutionSpace>>();
}

TEST(prep_transpose_view, 6DUnmanaged) {
  test_unmanaged_prep_transpose_view<
      Kokkos::View<double******, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double******, Kokkos::DefaultExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>>();
}

TEST(prep_transpose_view, 7DManaged) {
  test_managed_prep_transpose_view<
      Kokkos::View<double*******, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double*******, Kokkos::DefaultExecutionSpace>>();
}

TEST(prep_transpose_view, 7DUnmanaged) {
  test_unmanaged_prep_transpose_view<
      Kokkos::View<double*******, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double*******, Kokkos::DefaultExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>>();
}

TEST(prep_transpose_view, 8DManaged) {
  test_managed_prep_transpose_view<
      Kokkos::View<double********, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double********, Kokkos::DefaultExecutionSpace>>();
}

TEST(prep_transpose_view, 8DUnmanaged) {
  test_unmanaged_prep_transpose_view<
      Kokkos::View<double********, Kokkos::DefaultExecutionSpace>,
      Kokkos::View<double********, Kokkos::DefaultExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>>();
}
