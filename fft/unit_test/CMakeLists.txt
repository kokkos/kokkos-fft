add_executable(unit-tests-kokkos-fft-core
    Test_Main.cpp
    Test_Plans.cpp
    Test_Transform.cpp
    Test_Helpers.cpp 
)

target_compile_features(unit-tests-kokkos-fft-core PUBLIC cxx_std_17)
target_compile_options(unit-tests-kokkos-fft-core PUBLIC -std=c++17)

target_link_libraries(unit-tests-kokkos-fft-core PUBLIC Kokkos::fft GTest::gtest)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(unit-tests-kokkos-fft-core PROPERTIES DISCOVERY_TIMEOUT 600)