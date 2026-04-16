<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Testing Instructions for Kokkos-FFT

These instructions apply when creating, modifying, or reviewing test files (`Test_*.cpp`) in the `unit_test/` directories.

## Test Framework

- **Google Test** (gtest) is used for all C++ unit tests.
- Test sources are linked against `GTest::gtest`.
- Google Test is provided via the `tpls/googletest` submodule or an external installation.

## License Headers

Every test file must start with the SPDX license header:

```cpp
// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
```

CI enforces REUSE compliance — omitting this header causes build failure.

## Test Directory Structure

```
common/unit_test/          # Common utility tests
├── Test_Main.cpp          # Kokkos initialization/finalization
├── Test_Common_Utils.cpp  # Utility function tests
├── Test_Traits.cpp        # Type trait tests
├── Test_Layout.cpp        # View layout tests
├── Test_Normalization.cpp # Normalization tests
├── Test_Mapping.cpp       # Index mapping tests
├── Test_Transpose.cpp     # Transpose operation tests
├── Test_Extents.cpp       # Extent manipulation tests
├── Test_Padding.cpp       # Padding tests
└── Test_Helpers.cpp       # Helper function tests

fft/unit_test/             # FFT operation tests
├── Test_Main.cpp          # Kokkos initialization/finalization
├── Test_Plans.cpp         # FFT plan tests
├── Test_Transform.cpp     # FFT transform tests
├── Test_DynPlans.cpp      # Dynamic plan tests
└── Test_Utils.hpp         # Shared test utilities

distributed/unit_test/     # MPI distributed tests
├── Test_Main.cpp          # MPI + Kokkos initialization
├── Test_TplComm.cpp       # MPI communication tests
├── Test_All2All.cpp       # All-to-all tests
└── ...                    # Other distributed tests
```

## Test File Structure

Every test file follows this pattern:

```cpp
// SPDX license header

#include <gtest/gtest.h>
#include "KokkosFFT_ComponentUnderTest.hpp"
// other includes

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;

template <typename T>
using View1D = Kokkos::View<T*, execution_space>;

template <typename T>
using View2D = Kokkos::View<T**, execution_space>;

// Test fixtures and test cases...

}  // namespace
```

- Wrap all test code in an anonymous namespace.
- Define type aliases at the top of the namespace for common view types.
- Use `Kokkos::DefaultExecutionSpace` as the default execution space.

## Test_Main.cpp Pattern

Each test directory has a `Test_Main.cpp` that initializes Kokkos:

```cpp
// SPDX license header

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace testing::internal {
extern bool g_help_flag;
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  int result = 0;
  if (::testing::GTEST_FLAG(list_tests) || ::testing::internal::g_help_flag) {
    result = RUN_ALL_TESTS();
  } else {
    Kokkos::initialize(argc, argv);
    result = RUN_ALL_TESTS();
    Kokkos::finalize();
  }
  return result;
}
```

- Kokkos is initialized only when actually running tests (not when listing or showing help).
- Do not create new `Test_Main.cpp` files — reuse the existing ones per module.

## Test Types

### Simple Tests

```cpp
TEST(TestSuiteName, TestCaseName) {
  // Test logic
  EXPECT_EQ(result, expected);
}
```

### Typed Tests (Parameterized by Type)

```cpp
using test_types = ::testing::Types<float, double>;

template <typename T>
struct TestFixture : public ::testing::Test {
  using value_type = T;
};

TYPED_TEST_SUITE(TestFixture, test_types);

TYPED_TEST(TestFixture, TestCaseName) {
  using value_type = typename TestFixture::value_type;
  // Test logic using value_type
}
```

### Common Type Lists

```cpp
// Scalar types
using real_types = ::testing::Types<float, double>;

// Layout types
using layout_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Paired types (scalar + layout)
using test_types = ::testing::Types<
    std::pair<float, Kokkos::LayoutLeft>,
    std::pair<float, Kokkos::LayoutRight>,
    std::pair<double, Kokkos::LayoutLeft>,
    std::pair<double, Kokkos::LayoutRight>>;

// Execution space types
#if defined(KOKKOS_ENABLE_SERIAL)
using execution_spaces = ::testing::Types<
    Kokkos::Serial,
    Kokkos::DefaultHostExecutionSpace,
    Kokkos::DefaultExecutionSpace>;
#else
using execution_spaces = ::testing::Types<
    Kokkos::DefaultHostExecutionSpace,
    Kokkos::DefaultExecutionSpace>;
#endif
```

### Compile-Time Tests

For testing type traits and static assertions:

```cpp
template <typename T>
struct CompileTestFixture : public ::testing::Test {
  using value_type = T;
  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

TYPED_TEST_SUITE(CompileTestFixture, real_types);

TYPED_TEST(CompileTestFixture, StaticAssertionTest) {
  // Compile-time checks only — runtime is skipped via SetUp()
  static_assert(KokkosFFT::Impl::is_real_v<typename TestFixture::value_type>);
}
```

Use `GTEST_SKIP()` in `SetUp()` to prevent runtime execution for compile-time-only tests.

## Working with Kokkos Views in Tests

### Creating Views

```cpp
const int n = 128;
View1D<double> input("input", n);
View2D<Kokkos::complex<double>> output("output", n, n);
```

### Initializing View Data

```cpp
// On device
Kokkos::parallel_for(
    "init", Kokkos::RangePolicy<execution_space>(0, n),
    KOKKOS_LAMBDA(int i) { input(i) = static_cast<double>(i); });
```

### Comparing Views

Use the test utility functions from `Test_Utils.hpp`:

- `allclose(view1, view2, rtol, atol)` — compare views with tolerance using `Kokkos::parallel_reduce`.
- `multiply(view, scalar)` — scalar multiplication using `Kokkos::parallel_for`.

### Host Access

```cpp
auto h_input = Kokkos::create_mirror_view(input);
Kokkos::deep_copy(h_input, input);
// Access h_input(i) on host
```

## Assertions

- Use `EXPECT_*` macros (non-fatal) for most checks.
- Use `ASSERT_*` macros (fatal) only when subsequent checks depend on the result.
- Common assertions:
  - `EXPECT_EQ(a, b)` — equality.
  - `EXPECT_NEAR(a, b, tolerance)` — floating-point near-equality.
  - `EXPECT_TRUE(condition)` / `EXPECT_FALSE(condition)`.
  - `EXPECT_THROW(statement, exception_type)` — exception checking.
  - `EXPECT_NO_THROW(statement)` — no exception.

## Adding New Tests

1. Create `Test_<ComponentName>.cpp` in the appropriate `unit_test/` directory.
2. Add the SPDX license header.
3. Add the file to the test executable sources in `unit_test/CMakeLists.txt`:
   ```cmake
   add_executable(
       unit-tests-kokkos-fft-<module>
       Test_Main.cpp
       Test_ExistingComponent.cpp
       Test_NewComponent.cpp  # Add here
   )
   ```
4. Follow the test file structure pattern (anonymous namespace, type aliases, fixtures).
5. Use typed tests to cover both `float` and `double` precisions.
6. Test both `Kokkos::LayoutLeft` and `Kokkos::LayoutRight` layouts.

## CMake Test Configuration

```cmake
add_executable(unit-tests-kokkos-fft-<module> Test_Main.cpp Test_Component.cpp)
target_compile_features(unit-tests-kokkos-fft-<module> PUBLIC cxx_std_17)
target_link_libraries(unit-tests-kokkos-fft-<module> PUBLIC KokkosFFT::fft GTest::gtest)

include(GoogleTest)
gtest_discover_tests(
    unit-tests-kokkos-fft-<module>
    PROPERTIES DISCOVERY_TIMEOUT 600
    DISCOVERY_MODE PRE_TEST
)
```

- Use `gtest_discover_tests()` with `DISCOVERY_TIMEOUT 600` and `DISCOVERY_MODE PRE_TEST`.
- Link against `GTest::gtest` (imported target).

## MPI Tests

For distributed tests:

- Use `MPIEXEC_EXECUTABLE` for test execution.
- Test with 1, 2, and 4 MPI processes.
- Set appropriate timeouts (600s for 1-2 processes, 1200s for 4 processes).

## Running Tests

```bash
# Build with tests enabled
cmake -B build -DKokkosFFT_ENABLE_TESTS=ON -DKokkosFFT_ENABLE_INTERNAL_KOKKOS=ON -DKokkos_ENABLE_SERIAL=ON
cmake --build build -j $(nproc)

# Run all tests
cd build && ctest --output-on-failure

# Run specific test executable
./build/common/unit_test/unit-tests-kokkos-fft-common

# Run specific test case
./build/common/unit_test/unit-tests-kokkos-fft-common --gtest_filter="TestSuite.TestCase"
```

## Test Executables

| Executable | Module | Description |
|-----------|--------|-------------|
| `unit-tests-kokkos-fft-common` | common | Common utility tests |
| `unit-tests-kokkos-fft-core` | fft | Core FFT plan and transform tests |
| `unit-tests-kokkos-dynfft` | fft | Dynamic plan API tests |
| `unit-tests-kokkos-fft-distributed` | distributed | Distributed FFT tests (MPI) |

## Things to Avoid

- Do not create new `Test_Main.cpp` files — use the existing one per module.
- Do not use `ASSERT_*` when `EXPECT_*` suffices — keep tests running after non-critical failures.
- Do not skip testing both `float` and `double` precisions.
- Do not skip testing both `LayoutLeft` and `LayoutRight` layouts.
- Do not use raw `new`/`delete` in tests — use Kokkos views.
- Do not hardcode magic numbers — use named constants with descriptive names.
- Do not modify `tpls/googletest/` files.
