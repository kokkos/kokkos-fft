#ifndef KOKKOSFFT_BATCHED_PLAN_HPP
#define KOKKOSFFT_BATCHED_PLAN_HPP

#include "KokkosFFT_Batched_AxisData.hpp"
#include "KokkosFFT_Batched_Base_Types.hpp"
#include "KokkosFFT_Batched_Concepts.hpp"
#include "KokkosFFT_Batched_Factorize.hpp"
#include "KokkosFFT_Batched_Traits.hpp"
#include "KokkosFFT_Batched_Twiddles.hpp"
#include "KokkosFFT_Batched_fwd.hpp"
#include <KokkosFFT_Asserts.hpp>
#include <KokkosFFT_Traits.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>
#include <string>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

/// \brief Types, constants and compile-time checks shared by the leaf and the
/// recursive Plan specializations.
template <typename ExecPolicy, typename InViewType, typename OutViewType,
          typename Axes>
struct PlanBase {
  static_assert(
      ExecutionSpaceOrTeamPolicy<ExecPolicy>,
      "KokkosFFT::Batched::Plan: the first template argument must be an "
      "execution space or a Kokkos::TeamPolicy");
  static_assert(AxesSelectable<Axes>,
                "KokkosFFT::Batched::Plan: Axes must be a non-empty AxisTag");
  static_assert(
      Kokkos::is_view_v<InViewType> && Kokkos::is_view_v<OutViewType>,
      "KokkosFFT::Batched::Plan: InViewType and OutViewType must be Views");
  static_assert(
      InViewType::rank() == OutViewType::rank(),
      "KokkosFFT::Batched::Plan: InViewType and OutViewType must have the "
      "same rank");
  static_assert(
      Axes::rank <= InViewType::rank(),
      "KokkosFFT::Batched::Plan: the number of FFT axes must not exceed the "
      "view rank");
  static_assert(
      are_valid_axes<Axes, InViewType::rank()>(),
      "KokkosFFT::Batched::Plan: axes must be distinct and in [0, view rank). "
      "Negative axes are not supported");
  static_assert(
      KokkosFFT::Impl::have_same_base_floating_point_type_v<InViewType,
                                                            OutViewType>,
      "KokkosFFT::Batched::Plan: InViewType and OutViewType must have the "
      "same floating point precision");

  using policy_type = ExecPolicy;
  using execution_space = policy_execution_space_t<ExecPolicy>;
  using memory_space = typename execution_space::memory_space;
  using in_view_type = InViewType;
  using out_view_type = OutViewType;
  using axes_type = Axes;
  using in_value_type = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using lengths_type = Kokkos::Array<std::size_t, Axes::rank>;

  // Checked here so that an invalid Plan type fails on instantiation, not
  // only when `kind` is used.
  static_assert(is_fft_value_v<in_value_type> && is_fft_value_v<out_value_type>,
                "KokkosFFT::Batched::Plan: value types must be float, double, "
                "Kokkos::complex<float> or Kokkos::complex<double>");
  static_assert(
      KokkosFFT::Impl::is_complex_v<in_value_type> ||
          KokkosFFT::Impl::is_complex_v<out_value_type>,
      "KokkosFFT::Batched::Plan: real -> real transforms are not supported");

  static_assert(
      Kokkos::SpaceAccessibility<
          execution_space, typename InViewType::memory_space>::accessible &&
          Kokkos::SpaceAccessibility<
              execution_space, typename OutViewType::memory_space>::accessible,
      "KokkosFFT::Batched::Plan: the views must be accessible from the "
      "execution space");

  /// Serial plan (execution space) or team plan (TeamPolicy)
  static constexpr bool is_team = is_team_policy_v<ExecPolicy>;
  /// FFT rank of this (sub)plan
  static constexpr std::size_t rank = Axes::rank;
  /// Rank of the full batched view
  static constexpr std::size_t view_rank = InViewType::rank();
  static constexpr TransformKind kind =
      transform_kind_v<in_value_type, out_value_type>;
};

/// \brief Logical FFT lengths of the axes `Axes` from the full batched views
/// (host). Checks that the batch extents agree and that the FFT extents are
/// consistent with the transform kind: equal for C2C, and on the last axis
/// out = in/2+1 for R2C, in = out/2+1 for C2R.
template <typename Axes, TransformKind Kind, typename InViewType,
          typename OutViewType>
Kokkos::Array<std::size_t, Axes::rank> fft_lengths(const InViewType &in,
                                                   const OutViewType &out) {
  constexpr auto axes = axis_values<Axes>::value;
  for (std::size_t d = 0; d < InViewType::rank(); ++d) {
    const bool is_fft_axis =
        std::find(axes.begin(), axes.end(), static_cast<int>(d)) != axes.end();
    KOKKOSFFT_THROW_IF(
        !is_fft_axis && in.extent(d) != out.extent(d),
        "KokkosFFT::Batched::Plan: in and out must have the same extents "
        "along the batch dimensions");
  }

  Kokkos::Array<std::size_t, Axes::rank> lengths{};
  for (std::size_t i = 0; i < axes.size(); ++i) {
    const std::size_t in_extent = in.extent(axes[i]);
    const std::size_t out_extent = out.extent(axes[i]);
    const bool is_last = i + 1 == axes.size();
    if (is_last && Kind == TransformKind::R2C) {
      lengths[i] = in_extent;
      KOKKOSFFT_THROW_IF(
          out_extent != in_extent / 2 + 1,
          "KokkosFFT::Batched::Plan: R2C requires out extent n/2+1 along "
          "the last FFT axis");
    } else if (is_last && Kind == TransformKind::C2R) {
      lengths[i] = out_extent;
      KOKKOSFFT_THROW_IF(
          in_extent != out_extent / 2 + 1,
          "KokkosFFT::Batched::Plan: C2R requires in extent n/2+1 along "
          "the last FFT axis");
    } else {
      lengths[i] = in_extent;
      KOKKOSFFT_THROW_IF(
          out_extent != in_extent,
          "KokkosFFT::Batched::Plan: in and out must have the same "
          "extents along the FFT axes");
    }
    KOKKOSFFT_THROW_IF(
        lengths[i] == 0,
        "KokkosFFT::Batched::Plan: FFT lengths must be positive");
  }
  return lengths;
}

} // namespace Impl

/// \brief Leaf: a 1D transform along axis `Axis`.
/// Owns the stage descriptors and twiddle tables of this axis.
/// The sizes of all members are O(n) of this axis, independent of the batch
/// size. They are allocated once on the host at construction.
template <typename ExecPolicy, typename InViewType, typename OutViewType,
          int Axis>
class Plan<ExecPolicy, InViewType, OutViewType, AxisTag<Axis>>
    : public Impl::PlanBase<ExecPolicy, InViewType, OutViewType,
                            AxisTag<Axis>> {
  using base_type =
      Impl::PlanBase<ExecPolicy, InViewType, OutViewType, AxisTag<Axis>>;

public:
  using base_type::is_team;
  using base_type::kind;
  using base_type::rank;
  using base_type::view_rank;
  using typename base_type::axes_type;
  using typename base_type::complex_type;
  using typename base_type::execution_space;
  using typename base_type::float_type;
  using typename base_type::in_value_type;
  using typename base_type::in_view_type;
  using typename base_type::lengths_type;
  using typename base_type::memory_space;
  using typename base_type::out_value_type;
  using typename base_type::out_view_type;
  using typename base_type::policy_type;

  static constexpr int axis = Axis;

  /// Kernel descriptor of this axis: lengths, stage descriptors and twiddle
  /// tables. The kernels are templated on it, not on the Plan type, so all
  /// axes/plans with the same complex type and memory space share them (§9, R2)
  using axis_data_type = Impl::AxisData<complex_type, memory_space>;
  using stage_view_type = typename axis_data_type::stage_view_type;
  using twiddle_view_type = typename axis_data_type::twiddle_view_type;

  /// \brief Plan from the full batched views (host).
  /// \param exec_policy [in] Execution space instance (serial plan) or
  /// TeamPolicy (team plan). Its execution space fills the twiddle tables.
  /// \param in [in] Batched input view (extents are used, not the data)
  /// \param out [in] Batched output view (extents are used, not the data)
  Plan(const ExecPolicy &exec_policy, const InViewType &in,
       const OutViewType &out, AxisTag<Axis>)
      : Plan(exec_policy, Impl::fft_lengths<axes_type, kind>(in, out)) {}

  /// \brief Plan from the logical FFT length (host). Used by the recursive
  /// parent, which has no views of the child types (see development-plan.md
  /// §3.5).
  Plan(const ExecPolicy &exec_policy, const lengths_type &lengths) {
    init(Impl::get_space(exec_policy), lengths[0]);
  }

  /// \brief The kernel descriptor of this axis (what the kernels take)
  KOKKOS_FUNCTION const axis_data_type &kernel_data() const { return m_data; }

  /// \brief Logical length of the FFT axis `i` (only i = 0 for a leaf)
  KOKKOS_FUNCTION std::size_t length(std::size_t i) const {
    return m_data.length(i);
  }

  /// \brief Extent of the input along FFT axis `i`: n/2+1 for C2R, else n
  KOKKOS_FUNCTION std::size_t in_extent([[maybe_unused]] std::size_t i) const {
    KOKKOS_ASSERT(i == 0);
    return kind == TransformKind::C2R ? m_data.m_n / 2 + 1 : m_data.m_n;
  }

  /// \brief Extent of the output along FFT axis `i`: n/2+1 for R2C, else n
  KOKKOS_FUNCTION std::size_t out_extent([[maybe_unused]] std::size_t i) const {
    KOKKOS_ASSERT(i == 0);
    return kind == TransformKind::R2C ? m_data.m_n / 2 + 1 : m_data.m_n;
  }

  /// \brief Team scratch size to be provisioned by the user (host)
  std::size_t get_scratch_size([[maybe_unused]] int level = 0) const {
    return m_scratch_size;
  }

  /// \brief Total number of points of the transform (for normalization)
  KOKKOS_FUNCTION std::size_t fft_size() const { return m_data.m_n; }

  /// \brief Length of the transform run by the stages: n for C2C, n/2 for
  /// even-n R2C/C2R (half-length complex FFT), n for odd-n R2C/C2R (real
  /// half-complex FFT, KokkosFFT_Batched_Kernels_Odd.hpp)
  KOKKOS_FUNCTION std::size_t n_fft() const { return m_data.n_fft(); }

  KOKKOS_FUNCTION int nstages() const { return m_data.nstages(); }

  // The accessors below read Views in memory_space: use them in kernels only.
  // On the host, copy the Views (radices(), twiddles(), ...) with
  // Kokkos::create_mirror_view_and_copy.

  /// \brief Radix of stage `s`
  KOKKOS_FUNCTION std::size_t radix(int s) const { return m_data.radix(s); }

  /// \brief Offset of stage `s` in the twiddle table
  KOKKOS_FUNCTION std::size_t tw_offset(int s) const {
    return m_data.tw_offset(s);
  }

  /// \brief Offset of stage `s` in the generic-radix roots table
  KOKKOS_FUNCTION std::size_t root_offset(int s) const {
    return m_data.root_offset(s);
  }

  KOKKOS_FUNCTION const stage_view_type &radices() const {
    return m_data.radices();
  }

  KOKKOS_FUNCTION const twiddle_view_type &twiddles() const {
    return m_data.twiddles();
  }

  KOKKOS_FUNCTION const twiddle_view_type &roots() const {
    return m_data.roots();
  }

  /// \brief Twiddles of real transforms (empty for C2C):
  ///   even n: W_n^k, k = 0..n/4 (pre/post-processing)
  ///   odd n : W_n^t, t = 0..n-1 (half-complex DIT stages)
  KOKKOS_FUNCTION const twiddle_view_type &real_twiddles() const {
    return m_data.real_twiddles();
  }

private:
  void init(const execution_space &exec, std::size_t n) {
    KOKKOSFFT_THROW_IF(n == 0,
                       "KokkosFFT::Batched::Plan: FFT length must be positive");
    constexpr bool is_real = kind != TransformKind::C2C;
    const bool is_odd_real = is_real && n % 2 == 1;
    m_data.m_n = n;
    // Even-n real transforms run a complex FFT of half length; odd-n real
    // transforms run a real FFT of length n on half-complex data
    m_data.m_n_fft = is_real && !is_odd_real ? n / 2 : n;

    const auto radices = Impl::factorize(m_data.m_n_fft);
    m_data.m_nstages = static_cast<int>(radices.size());

    const std::string prefix = "KokkosFFT::Batched::Plan::";
    const std::string suffix = "_axis" + std::to_string(Axis);
    m_data.m_radix = Impl::to_view(exec, prefix + "radix" + suffix, radices);
    if (is_odd_real) {
      // Only the W_n^t table is needed; the complex stage tables stay empty
      m_data.m_real_twiddles =
          Impl::to_view(exec, prefix + "real_twiddles" + suffix,
                        Impl::make_odd_real_twiddles<complex_type>(n));
      return;
    }

    const auto tables =
        Impl::make_stage_tables<complex_type>(m_data.m_n_fft, radices);
    m_data.m_tw_offset =
        Impl::to_view(exec, prefix + "tw_offset" + suffix, tables.tw_offset);
    m_data.m_root_offset = Impl::to_view(exec, prefix + "root_offset" + suffix,
                                         tables.root_offset);
    m_data.m_twiddles =
        Impl::to_view(exec, prefix + "twiddles" + suffix, tables.twiddles);
    m_data.m_roots =
        Impl::to_view(exec, prefix + "roots" + suffix, tables.roots);
    if constexpr (is_real) {
      m_data.m_real_twiddles =
          Impl::to_view(exec, prefix + "real_twiddles" + suffix,
                        Impl::make_real_twiddles<complex_type>(n));
    }
  }

  //! Lengths, stage descriptors and twiddle tables (what the kernels use)
  axis_data_type m_data;
  //! Team scratch size, computed on the host at construction
  std::size_t m_scratch_size = 0;
};

/// \brief Recursive case: split the plan into a heads subplan and a last-axis
/// plan. It owns no data except its two children.
template <typename ExecPolicy, typename InViewType, typename OutViewType,
          int Axis, int NextAxis, int... Rest>
class Plan<ExecPolicy, InViewType, OutViewType,
           AxisTag<Axis, NextAxis, Rest...>>
    : public Impl::PlanBase<ExecPolicy, InViewType, OutViewType,
                            AxisTag<Axis, NextAxis, Rest...>> {
  using base_type = Impl::PlanBase<ExecPolicy, InViewType, OutViewType,
                                   AxisTag<Axis, NextAxis, Rest...>>;

public:
  using base_type::is_team;
  using base_type::kind;
  using base_type::rank;
  using base_type::view_rank;
  using typename base_type::axes_type;
  using typename base_type::complex_type;
  using typename base_type::execution_space;
  using typename base_type::float_type;
  using typename base_type::in_value_type;
  using typename base_type::in_view_type;
  using typename base_type::lengths_type;
  using typename base_type::memory_space;
  using typename base_type::out_value_type;
  using typename base_type::out_view_type;
  using typename base_type::policy_type;

private:
  using split_type = Impl::split_view_types<execution_space, in_value_type,
                                            out_value_type, view_rank>;

public:
  using heads_axes_type = typename axes_type::heads;
  using last_axes_type = AxisTag<axes_type::last_v>;

  using heads_plan_type =
      Plan<ExecPolicy, typename split_type::heads_in_view_type,
           typename split_type::heads_out_view_type, heads_axes_type>;
  using last_plan_type =
      Plan<ExecPolicy, typename split_type::last_in_view_type,
           typename split_type::last_out_view_type, last_axes_type>;

  /// \brief Plan from the full batched views (host): extracts and validates
  /// the FFT lengths, then delegates to the lengths constructor.
  Plan(const ExecPolicy &exec_policy, const InViewType &in,
       const OutViewType &out, axes_type)
      : Plan(exec_policy, Impl::fft_lengths<axes_type, kind>(in, out)) {}

  /// \brief Plan from the logical FFT lengths (host, in the order of Axes):
  /// the heads child gets lengths[0..rank-1), the last child lengths[rank-1]
  /// (see development-plan.md §3.5).
  Plan(const ExecPolicy &exec_policy, const lengths_type &lengths)
      : m_heads_plan(exec_policy, heads_lengths(lengths)),
        m_last_plan(exec_policy, last_lengths(lengths)) {
    if constexpr (kind != TransformKind::C2C) {
      // The C2C passes of the heads use the real view as scratch: a line of
      // length n_a needs 2 n_a reals, and the real view holds
      // n_last * (product of the other extents) reals (§2.6)
      KOKKOSFFT_THROW_IF(
          lengths[rank - 1] < 2,
          "KokkosFFT::Batched::Plan: N-D R2C/C2R requires a length of at "
          "least 2 along the last FFT axis");
    }
  }

  /// \brief Logical length of the FFT axis `i` (in the order of Axes)
  KOKKOS_FUNCTION std::size_t length(std::size_t i) const {
    return i + 1 < rank ? m_heads_plan.length(i) : m_last_plan.length(0);
  }

  KOKKOS_FUNCTION std::size_t in_extent(std::size_t i) const {
    return i + 1 < rank ? m_heads_plan.in_extent(i) : m_last_plan.in_extent(0);
  }

  KOKKOS_FUNCTION std::size_t out_extent(std::size_t i) const {
    return i + 1 < rank ? m_heads_plan.out_extent(i)
                        : m_last_plan.out_extent(0);
  }

  /// \brief Total number of points of the transform (for normalization)
  KOKKOS_FUNCTION std::size_t fft_size() const {
    return m_heads_plan.fft_size() * m_last_plan.fft_size();
  }

  /// \brief Team scratch size to be provisioned by the user (host)
  std::size_t get_scratch_size(int level = 0) const {
    return std::max(m_heads_plan.get_scratch_size(level),
                    m_last_plan.get_scratch_size(level));
  }

  KOKKOS_FUNCTION const heads_plan_type &heads_plan() const {
    return m_heads_plan;
  }
  KOKKOS_FUNCTION const last_plan_type &last_plan() const {
    return m_last_plan;
  }

private:
  static typename heads_plan_type::lengths_type
  heads_lengths(const lengths_type &lengths) {
    typename heads_plan_type::lengths_type heads{};
    for (std::size_t i = 0; i + 1 < rank; ++i)
      heads[i] = lengths[i];
    return heads;
  }
  static typename last_plan_type::lengths_type
  last_lengths(const lengths_type &lengths) {
    return {lengths[rank - 1]};
  }

  heads_plan_type m_heads_plan;
  last_plan_type m_last_plan;
};

/// \brief CTAD: the first argument picks the level, e.g.
///   Plan plan(exec_space, x, x_hat, AxisTag<0>{});   // serial plan
///   Plan plan(team_policy, x, x_hat, AxisTag<0>{});  // team plan
template <typename ExecPolicy, typename InViewType, typename OutViewType,
          typename Axes>
Plan(const ExecPolicy &, const InViewType &, const OutViewType &, Axes)
    -> Plan<ExecPolicy, InViewType, OutViewType, Axes>;

} // namespace Batched
} // namespace KokkosFFT

#endif
