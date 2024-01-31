add_subdirectory(src)
if(BUILD_TESTING)
    add_subdirectory(unit_test)
endif()

if(KokkosFFT_ENABLE_BENCHMARK)
    add_subdirectory(perf_test)
endif()