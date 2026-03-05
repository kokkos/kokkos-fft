// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_BLOCK_HPP
#define KOKKOSFFT_DISTRIBUTED_BLOCK_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_PackUnpack.hpp"
#include "KokkosFFT_Distributed_All2All.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Distributed FFT transposition block
/// 0. Sanity check
/// 1. Pack
/// 2. Fence
/// 3. All2All
/// 4. Unpack
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam DIM Rank of in/out View
template <typename ExecutionSpace, std::size_t DIM>
class TransBlock {
  using execSpace           = ExecutionSpace;
  using buffer_extents_type = KokkosFFT::shape_type<DIM + 1>;
  using extents_type        = KokkosFFT::shape_type<DIM>;

  ExecutionSpace m_exec;
  buffer_extents_type m_buffer_extents;
  extents_type m_src_map, m_dst_map;
  std::size_t m_src_axis, m_dst_axis;

 public:
  /// \brief Constructor for TransBlock
  /// This class does not assume any data type at the construction time
  /// It only describes the shape of the buffer and mappings
  ///
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] buffer_extents The extents of the buffer
  /// \param[in] src_map The map of the source view
  /// \param[in] src_axis The axis to be split
  /// \param[in] dst_map The map of the destination view
  /// \param[in] dst_axis The axis to be merged
  explicit TransBlock(const ExecutionSpace& exec_space,
                      const buffer_extents_type& buffer_extents,
                      const extents_type& src_map, std::size_t src_axis,
                      const extents_type& dst_map, std::size_t dst_axis)
      : m_exec(exec_space),
        m_buffer_extents(buffer_extents),
        m_src_map(src_map),
        m_dst_map(dst_map),
        m_src_axis(src_axis),
        m_dst_axis(dst_axis) {}

  /// \brief Operator for TransBlock
  /// This operator does the following:
  /// 0. Sanity check
  /// 1. Pack
  /// 2. Fence
  /// 3. All2All
  /// 4. Unpack
  /// As send/recv buffers are unmanaged views with 1 extra dimension,
  /// the maximum dimension of the input/output view is 7.
  /// Mappings are reversed for backward direction.
  ///
  /// \tparam CommType Type of communicator wrapper
  /// \tparam ViewType Type of input/output views
  /// \tparam BufferType Type of send/receive buffers
  /// \param[in] comm communicator wrapper
  /// \param[in] in Input view
  /// \param[in] out Output view
  /// \param[in] send Send buffer
  /// \param[in] recv Receive buffer
  /// \param[in] direction Direction of transpose operation used for either
  /// forward or backward direction
  template <typename CommType, typename ViewType, typename BufferType>
  void operator()(const CommType& comm, const ViewType& in, const ViewType& out,
                  const BufferType& send, const BufferType& recv,
                  KokkosFFT::Direction direction) const {
    using value_type = typename ViewType::non_const_value_type;
    using LayoutType = typename ViewType::array_layout;
    using buffer_data_type =
        KokkosFFT::Impl::add_pointer_n_t<value_type, DIM + 1>;
    using buffer_view_type = Kokkos::View<buffer_data_type, LayoutType,
                                          typename ViewType::execution_space>;

    Kokkos::Profiling::ScopedRegion region(
        "KokkosFFT::Distributed::TransBlock");
    // Making unmanaged views from meta data
    buffer_view_type send_buffer(
        reinterpret_cast<value_type*>(send.data()),
        KokkosFFT::Impl::create_layout<LayoutType>(m_buffer_extents)),
        recv_buffer(
            reinterpret_cast<value_type*>(recv.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(m_buffer_extents));

    good(in, out, send_buffer, recv_buffer);

    bool is_forward = direction == KokkosFFT::Direction::forward;
    auto src_map    = is_forward ? m_src_map : m_dst_map;
    auto dst_map    = is_forward ? m_dst_map : m_src_map;
    auto src_axis   = is_forward ? m_src_axis : m_dst_axis;
    auto dst_axis   = is_forward ? m_dst_axis : m_src_axis;

    pack(m_exec, in, send_buffer, src_map, src_axis);
    m_exec.fence();
    all2all(send_buffer, recv_buffer, comm);
    unpack(m_exec, recv_buffer, out, dst_map, dst_axis);
  }

 private:
  /// \brief Sanity check for input/output views and buffers
  /// \tparam ViewType Type of input/output views
  /// \tparam BufferType Type of send/receive buffers
  /// \param[in] in Input view
  /// \param[out] out Output view
  /// \param[in,out] send Send buffer
  /// \param[in,out] recv Receive buffer
  /// \throw std::runtime_error if input and output views have larger size than
  /// send and receive buffers. If send and receive buffers are aliasing with
  /// input and output views, throw std::runtime_error
  template <typename ViewType, typename BufferType>
  void good(const ViewType& in, const ViewType& out, const BufferType& send,
            const BufferType& recv) const {
    using view_value_type   = typename ViewType::non_const_value_type;
    using buffer_value_type = typename BufferType::non_const_value_type;

    static_assert(std::is_same_v<buffer_value_type, view_value_type>,
                  "Input and output views must have same value type");

    auto buffer_size = send.size();
    auto in_size     = in.size();
    auto out_size    = out.size();

    auto error_msg = [](std::string_view details, std::string_view view_name,
                        const ViewType& view, std::string_view buffer_name,
                        const BufferType& buffer) -> std::string {
      std::string message(details);
      message += ": \n";
      message += std::string(view_name);
      message +=
          view.label().empty() ? "" : " (" + std::string(view.label()) + ")";
      message += " with extents(";
      message += std::to_string(view.extent(0));
      for (std::size_t r = 1; r < view.rank(); r++) {
        message += ",";
        message += std::to_string(view.extent(r));
      }
      message += "), ";
      message += " and " + std::string(buffer_name);
      message += buffer.label().empty()
                     ? ""
                     : " (" + std::string(buffer.label()) + ")";
      message += " with extents(";
      message += std::to_string(buffer.extent(0));
      for (std::size_t r = 1; r < buffer.rank(); r++) {
        message += ",";
        message += std::to_string(buffer.extent(r));
      }
      message += ")";

      return message;
    };

    // Check input and output view sizes are smaller than buffer sizes
    KOKKOSFFT_THROW_IF(
        in_size > buffer_size,
        error_msg("Input view size must be smaller than the send buffer size",
                  "input", in, "send_buffer", send));

    KOKKOSFFT_THROW_IF(
        out_size > buffer_size,
        error_msg(
            "Output view size must be smaller than the receive buffer size",
            "output", out, "recv_buffer", recv));

    // Check in and send_buffer are not aliasing
    KOKKOSFFT_THROW_IF(
        KokkosFFT::Impl::are_aliasing(in.data(), send.data()),
        error_msg("Input view must not be aliasing with send buffer", "input",
                  in, "send_buffer", send));

    // Check out and recv_buffer are not aliasing
    KOKKOSFFT_THROW_IF(
        KokkosFFT::Impl::are_aliasing(out.data(), recv.data()),
        error_msg("Output view must not be aliasing with receive buffer",
                  "output", out, "recv_buffer", recv));
  }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
