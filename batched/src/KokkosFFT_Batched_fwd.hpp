#ifndef KOKKOSFFT_BATCHED_FWD_HPP
#define KOKKOSFFT_BATCHED_FWD_HPP

namespace KokkosFFT {
namespace Batched {
template <int... Tags>
struct AxisTag;

template <typename ExecPolicy, typename InViewType, typename OutViewType,
          typename Axes>
class Plan;
}  // namespace Batched
}  // namespace KokkosFFT

#endif
