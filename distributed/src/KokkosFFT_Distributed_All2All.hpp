// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_ALL2ALL_HPP
#define KOKKOSFFT_DISTRIBUTED_ALL2ALL_HPP

#include "KokkosFFT_Distributed_MPI_Types.hpp"
#include "KokkosFFT_Distributed_MPI_Comm.hpp"
#include "KokkosFFT_Distributed_MPI_All2All.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {
template <typename ExecutionSpace>
using TplComm = ScopedMPIComm<ExecutionSpace>;
}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
