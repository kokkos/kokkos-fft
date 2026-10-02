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
