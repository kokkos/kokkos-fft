#ifndef KOKKOSFFT_BATCHED_ACCESSORS_HPP
#define KOKKOSFFT_BATCHED_ACCESSORS_HPP

#include "KokkosFFT_Batched_Levels.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

// Lines are accessed through load(i) / store(i, v) so that kernels work with
// both StridedLine and RealPairAsComplex.

/// \brief A 1D line of a View: base pointer and stride. Works for any layout
/// (LayoutLeft, LayoutRight, LayoutStride subviews).
template <typename ValueType>
struct StridedLine {
  using value_type = ValueType;

  ValueType *m_data;
  std::size_t m_stride;

  KOKKOS_INLINE_FUNCTION ValueType load(std::size_t i) const {
    return m_data[i * m_stride];
  }
  KOKKOS_INLINE_FUNCTION void store(std::size_t i, const ValueType &v) const {
    m_data[i * m_stride] = v;
  }
};

/// \brief A real line of length 2h read as h complex values:
///   z[k] = x[2k] + i x[2k+1]
/// load() constructs a Kokkos::complex from the two reals; store() writes
/// them back. The two reals are generally not adjacent in memory (strided
/// slices), so the line cannot hand out a Kokkos::complex reference.
template <typename RealType>
struct RealPairAsComplex {
  using value_type = Kokkos::complex<RealType>;

  RealType *m_data;
  std::size_t m_stride;

  KOKKOS_INLINE_FUNCTION value_type load(std::size_t k) const {
    return value_type(m_data[2 * k * m_stride], m_data[(2 * k + 1) * m_stride]);
  }
  KOKKOS_INLINE_FUNCTION void store(std::size_t k, const value_type &z) const {
    m_data[2 * k * m_stride]       = z.real();
    m_data[(2 * k + 1) * m_stride] = z.imag();
  }
};

/// \brief A complex line of length m read as 2m reals:
///   r[2i] = Re z[i], r[2i+1] = Im z[i]
/// Used as a real work buffer by odd-n R2C/C2R.
template <typename RealType>
struct ComplexAsReal {
  using value_type = RealType;

  Kokkos::complex<RealType> *m_data;
  std::size_t m_stride;

  KOKKOS_INLINE_FUNCTION RealType load(std::size_t i) const {
    const auto &z = m_data[(i / 2) * m_stride];
    return i % 2 == 0 ? z.real() : z.imag();
  }
  KOKKOS_INLINE_FUNCTION void store(std::size_t i, RealType v) const {
    auto &z = m_data[(i / 2) * m_stride];
    if (i % 2 == 0) {
      z.real(v);
    } else {
      z.imag(v);
    }
  }
};

/// \brief Line of a rank-1 View
template <typename ViewType>
KOKKOS_INLINE_FUNCTION auto make_line(const ViewType &view) {
  static_assert(ViewType::rank() == 1, "make_line: a rank-1 View is required");
  return StridedLine<typename ViewType::value_type>{
      view.data(), static_cast<std::size_t>(view.stride(0))};
}

/// \brief Rank-1 complex View of length m read as 2m reals
template <typename ViewType>
KOKKOS_INLINE_FUNCTION auto make_real_line(const ViewType &view) {
  static_assert(ViewType::rank() == 1,
                "make_real_line: a rank-1 View is required");
  using real_type = typename ViewType::value_type::value_type;
  return ComplexAsReal<real_type>{view.data(),
                                  static_cast<std::size_t>(view.stride(0))};
}

/// \brief Rank-1 real View of even length 2h read as h complex values
template <typename ViewType>
KOKKOS_INLINE_FUNCTION auto make_pair_line(const ViewType &view) {
  static_assert(ViewType::rank() == 1,
                "make_pair_line: a rank-1 View is required");
  return RealPairAsComplex<typename ViewType::value_type>{
      view.data(), static_cast<std::size_t>(view.stride(0))};
}

// Lines of an N-D View along dimension Dim. Line l is identified by the
// indices of the other dimensions, decoded from l with the last dimension
// fastest. Views whose extents agree outside Dim (e.g. in/out of an R2C pass
// along its real axis) get the same line for the same l.

/// \brief Number of lines of `view` along dimension Dim
template <int Dim, typename ViewType>
KOKKOS_INLINE_FUNCTION std::size_t line_count(const ViewType &view) {
  std::size_t count = 1;
  for (std::size_t e = 0; e < ViewType::rank(); ++e) {
    if (e != static_cast<std::size_t>(Dim)) count *= view.extent(e);
  }
  return count;
}

/// \brief Offset (in elements, from view.data()) of line `l` along Dim
template <int Dim, typename ViewType>
KOKKOS_INLINE_FUNCTION std::size_t line_offset(const ViewType &view,
                                               std::size_t l) {
  std::size_t offset = 0;
  for (int e = static_cast<int>(ViewType::rank()) - 1; e >= 0; --e) {
    if (e == Dim) continue;
    const std::size_t extent = view.extent(e);
    offset += (l % extent) * view.stride(e);
    l /= extent;
  }
  return offset;
}

/// \brief Offset (in elements, from view.data()) of element `i` in a flat
/// enumeration of `view` (last dimension fastest)
template <typename ViewType>
KOKKOS_INLINE_FUNCTION std::size_t element_offset(const ViewType &view,
                                                  std::size_t i) {
  std::size_t offset = 0;
  for (int e = static_cast<int>(ViewType::rank()) - 1; e >= 0; --e) {
    const std::size_t extent = view.extent(e);
    offset += (i % extent) * view.stride(e);
    i /= extent;
  }
  return offset;
}

/// \brief Line `l` of `view` along Dim
template <int Dim, typename ViewType>
KOKKOS_INLINE_FUNCTION auto line_at(const ViewType &view, std::size_t l) {
  return StridedLine<typename ViewType::value_type>{
      view.data() + line_offset<Dim>(view, l),
      static_cast<std::size_t>(view.stride(Dim))};
}

/// \brief Line `l` of a real `view` along Dim, read as complex pairs
template <int Dim, typename ViewType>
KOKKOS_INLINE_FUNCTION auto pair_line_at(const ViewType &view, std::size_t l) {
  return RealPairAsComplex<typename ViewType::value_type>{
      view.data() + line_offset<Dim>(view, l),
      static_cast<std::size_t>(view.stride(Dim))};
}

// Line addressing for passes that merge lines (team level,
// KokkosFFT_Batched_Levels.hpp). There, every work item finds its line from the
// line index, so the decoding must be cheap. A LinePairMap is built once per
// pass for the two views of the pass (in and out, whose extents agree outside
// Dim):
// - line l is decoded with the first dimension fastest if `first_fastest` is
//   set, with the last one otherwise (as line_offset); dimensions of extent 1
//   are skipped;
// - neighbouring dimensions that are contiguous in both views (the stride of
//   the slower one is the extent times the stride of the faster one) are
//   merged into one, which does not change the line enumeration;
// - offsets(l) gives the offsets of line l in both views with one division
//   per remaining dimension but the last (none for a contiguous slice whose
//   pass runs along its first or last dimension).

/// \brief Line index -> offsets of the line in two views (see above)
template <std::size_t MaxDims>
struct LinePairMap {
  int m_ndims = 0;
  Kokkos::Array<std::size_t, MaxDims> m_extent;
  Kokkos::Array<std::size_t, MaxDims> m_stride_a;
  Kokkos::Array<std::size_t, MaxDims> m_stride_b;

  KOKKOS_INLINE_FUNCTION Kokkos::pair<std::size_t, std::size_t> offsets(
      std::size_t l) const {
    std::size_t offset_a = 0, offset_b = 0;
    for (int d = 0; d + 1 < m_ndims; ++d) {
      const std::size_t q = fast_div(l, m_extent[d]);
      const std::size_t i = l - q * m_extent[d];
      offset_a += i * m_stride_a[d];
      offset_b += i * m_stride_b[d];
      l = q;
    }
    if (m_ndims > 0) {
      offset_a += l * m_stride_a[m_ndims - 1];
      offset_b += l * m_stride_b[m_ndims - 1];
    }
    return {offset_a, offset_b};
  }
};

/// \brief LinePairMap of the lines of views `a` and `b` along Dim
template <int Dim, typename ViewTypeA, typename ViewTypeB>
KOKKOS_INLINE_FUNCTION auto make_line_pair_map(const ViewTypeA &a,
                                               const ViewTypeB &b,
                                               bool first_fastest) {
  constexpr int rank = static_cast<int>(ViewTypeA::rank());
  static_assert(ViewTypeB::rank() == ViewTypeA::rank(),
                "make_line_pair_map: views of different ranks");
  LinePairMap<(rank > 1 ? rank - 1 : 1)> map;
  for (int i = 0; i < rank; ++i) {
    const int e = first_fastest ? i : rank - 1 - i;
    if (e == Dim || a.extent(e) == 1) continue;
    const std::size_t extent   = a.extent(e);
    const std::size_t stride_a = a.stride(e);
    const std::size_t stride_b = b.stride(e);
    if (map.m_ndims > 0) {
      const int d = map.m_ndims - 1;
      if (stride_a == map.m_extent[d] * map.m_stride_a[d] &&
          stride_b == map.m_extent[d] * map.m_stride_b[d]) {
        map.m_extent[d] *= extent;  // contiguous with the previous dimension
        continue;
      }
    }
    map.m_extent[map.m_ndims]   = extent;
    map.m_stride_a[map.m_ndims] = stride_a;
    map.m_stride_b[map.m_ndims] = stride_b;
    ++map.m_ndims;
  }
  return map;
}

/// \brief Line of `view` along Dim starting at `offset` (from LinePairMap)
template <int Dim, typename ViewType>
KOKKOS_INLINE_FUNCTION auto line_from_offset(const ViewType &view,
                                             std::size_t offset) {
  return StridedLine<typename ViewType::value_type>{
      view.data() + offset, static_cast<std::size_t>(view.stride(Dim))};
}

/// \brief Line of a real `view` along Dim starting at `offset`, read as
/// complex pairs
template <int Dim, typename ViewType>
KOKKOS_INLINE_FUNCTION auto pair_line_from_offset(const ViewType &view,
                                                  std::size_t offset) {
  return RealPairAsComplex<typename ViewType::value_type>{
      view.data() + offset, static_cast<std::size_t>(view.stride(Dim))};
}

/// \brief Whether the first dimension of `view` is the closest in memory
/// (LayoutLeft-like): consecutive lines are then neighbours in memory when
/// they are enumerated with the first dimension fastest
template <typename ViewType>
KOKKOS_INLINE_FUNCTION bool first_dim_is_inner(const ViewType &view) {
  constexpr std::size_t rank = ViewType::rank();
  return rank >= 2 && view.stride(0) < view.stride(rank - 1);
}

/// \brief Whether neighbouring lines of `view` along Dim are closer in
/// memory than neighbouring elements of one line, i.e. whether another
/// dimension (of extent > 1) has a smaller stride than Dim
template <int Dim, typename ViewType>
KOKKOS_INLINE_FUNCTION bool lines_are_inner(const ViewType &view) {
  const auto line_stride = view.stride(Dim);
  for (int e = 0; e < static_cast<int>(ViewType::rank()); ++e) {
    if (e != Dim && view.extent(e) > 1 && view.stride(e) < line_stride)
      return true;
  }
  return false;
}

/// \brief Scratch line of complex values stored as pairs of reals in a
/// whole real N-D View: z[k] = (r[2k], r[2k+1]) in a flat enumeration of
/// the View. The enumeration is memory order when the span is contiguous
/// and a decode over extents/strides otherwise; any bijection works since
/// the contents only matter to the line that uses the scratch.
/// Capacity: view.size() / 2 complex values. `m_base` (in complex values)
/// selects a region, so that concurrent lines use disjoint scratch.
template <typename RealType, std::size_t Rank>
struct FlatRealPairLine {
  using value_type = Kokkos::complex<RealType>;

  RealType *m_data;
  bool m_contiguous;
  Kokkos::Array<std::size_t, Rank> m_extents;
  Kokkos::Array<std::size_t, Rank> m_strides;
  std::size_t m_base;

  KOKKOS_INLINE_FUNCTION std::size_t offset(std::size_t f) const {
    if (m_contiguous) return f;
    std::size_t off = 0;
    for (int e = static_cast<int>(Rank) - 1; e >= 0; --e) {
      off += (f % m_extents[e]) * m_strides[e];
      f /= m_extents[e];
    }
    return off;
  }
  KOKKOS_INLINE_FUNCTION value_type load(std::size_t k) const {
    const std::size_t f = 2 * (m_base + k);
    return value_type(m_data[offset(f)], m_data[offset(f + 1)]);
  }
  KOKKOS_INLINE_FUNCTION void store(std::size_t k, const value_type &z) const {
    const std::size_t f   = 2 * (m_base + k);
    m_data[offset(f)]     = z.real();
    m_data[offset(f + 1)] = z.imag();
  }
};

/// \brief Whole real N-D View used as complex scratch, starting at complex
/// index `base`
template <typename ViewType>
KOKKOS_INLINE_FUNCTION auto make_flat_pair_scratch(const ViewType &view,
                                                   std::size_t base = 0) {
  constexpr std::size_t rank = ViewType::rank();
  FlatRealPairLine<typename ViewType::value_type, rank> line{
      view.data(), view.span_is_contiguous(), {}, {}, base};
  for (std::size_t e = 0; e < rank; ++e) {
    line.m_extents[e] = view.extent(e);
    line.m_strides[e] = view.stride(e);
  }
  return line;
}

}  // namespace Impl
}  // namespace Batched
}  // namespace KokkosFFT

#endif
