add_executable(
  unit-tests-fft-batched
  Test_Main.cpp
  Test_Base_Types.cpp
  Test_Traits.cpp
  Test_Concepts.cpp
  Test_Plan.cpp
  Test_Execute1D.cpp
  Test_ExecuteND.cpp
)

target_link_libraries(unit-tests-fft-batched PUBLIC batched GTest::gtest)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(unit-tests-fft-batched PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
