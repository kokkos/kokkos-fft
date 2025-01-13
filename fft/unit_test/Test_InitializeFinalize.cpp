// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Core.hpp"

/**
 * Fixture that checks Kokkos is neither initialized nor finalized before and
 * after each test.
 */
class KokkosFFTExecutionEnvironmentNeverInitialized : public ::testing::Test {
  static void checkNeverInitialized() {
    ASSERT_FALSE(Kokkos::is_initialized());
    ASSERT_FALSE(Kokkos::is_finalized());
    ASSERT_FALSE(KokkosFFT::is_initialized());
    ASSERT_FALSE(KokkosFFT::is_finalized());
  }

 protected:
  void SetUp() override { checkNeverInitialized(); }
  void TearDown() override { checkNeverInitialized(); }
};

namespace {
using InitializeFinalize_DeathTest =
    KokkosFFTExecutionEnvironmentNeverInitialized;
using InitializeFinalize_Test = KokkosFFTExecutionEnvironmentNeverInitialized;

TEST_F(InitializeFinalize_DeathTest, initialize) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  // Valid usage
  EXPECT_EXIT(
      {
        Kokkos::initialize();
        KokkosFFT::initialize();
        KokkosFFT::finalize();
        Kokkos::finalize();
        std::exit(EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");

  // KokkosFFT is initialized twice
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        KokkosFFT::initialize();
        KokkosFFT::initialize();
        Kokkos::finalize();
      },
      "Error: KokkosFFT::initialize\\(\\) has already been called. KokkosFFT "
      "can be "
      "initialized at most once\\.");

  // KokkosFFT is initialized twice after finalize
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        KokkosFFT::initialize();
        KokkosFFT::finalize();
        KokkosFFT::initialize();
        Kokkos::finalize();
      },
      "Error: KokkosFFT::initialize\\(\\) has already been called. KokkosFFT "
      "can be "
      "initialized at most once\\.");

  // KokkosFFT is initialized before Kokkos initialization
  EXPECT_DEATH(
      {
        KokkosFFT::initialize();
        Kokkos::initialize();
        Kokkos::finalize();
      },
      "Error: KokkosFFT::initialize\\(\\) must not be called before "
      "initializing Kokkos\\.");

  // KokkosFFT is initialized after Kokkos finalization
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        KokkosFFT::initialize();
      },
      "Error: KokkosFFT::initialize\\(\\) must not be called after finalizing "
      "Kokkos\\.");
}

TEST_F(InitializeFinalize_DeathTest, finalize) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  // KokkosFFT is not initialized
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        KokkosFFT::finalize();
        Kokkos::finalize();
      },
      "Error: KokkosFFT::finalize\\(\\) may only be called after KokkosFFT has "
      "been initialized\\.");

  // KokkosFFT is finalized twice
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        KokkosFFT::initialize();
        KokkosFFT::finalize();
        KokkosFFT::finalize();
        Kokkos::finalize();
      },
      "Error: KokkosFFT::finalize\\(\\) has already been called\\.");

  // KokkosFFT is finalized before Kokkos initialization
  EXPECT_DEATH(
      {
        KokkosFFT::finalize();
        Kokkos::initialize();
        Kokkos::finalize();
      },
      "Error: KokkosFFT::finalize\\(\\) may only be called after Kokkos has "
      "been initialized\\.");

  // KokkosFFT is finalized after Kokkos finalization
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        KokkosFFT::initialize();
        Kokkos::finalize();
        KokkosFFT::finalize();
      },
      "Error: KokkosFFT::finalize\\(\\) must be called before finalizing "
      "Kokkos\\.");
}

TEST_F(InitializeFinalize_Test, is_initialized) {
  EXPECT_EXIT(
      {
        Kokkos::initialize();
        bool success = true;
        success &= !KokkosFFT::is_initialized();
        std::cout << "success" << success << std::endl;
        KokkosFFT::initialize();
        success &= KokkosFFT::is_initialized();
        std::cout << "success" << success << std::endl;
        KokkosFFT::finalize();
        success &= !KokkosFFT::is_initialized();
        std::cout << "success" << success << std::endl;
        Kokkos::finalize();
        std::exit(success ? EXIT_SUCCESS : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST_F(InitializeFinalize_Test, is_finalized) {
  EXPECT_EXIT(
      {
        Kokkos::initialize();
        bool success = true;
        success &= !KokkosFFT::is_finalized();
        KokkosFFT::initialize();
        success &= !KokkosFFT::is_finalized();
        KokkosFFT::finalize();
        success &= KokkosFFT::is_finalized();
        std::exit(success ? EXIT_SUCCESS : EXIT_FAILURE);
        Kokkos::finalize();
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

}  // namespace
