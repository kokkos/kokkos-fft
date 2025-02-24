// SPDX-FileCopyrightText: (C) 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
// SPDX-FileCopyrightText: Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  int result = 0;
  Kokkos::initialize(argc, argv);
  result = RUN_ALL_TESTS();
  Kokkos::finalize();

  return result;
}
