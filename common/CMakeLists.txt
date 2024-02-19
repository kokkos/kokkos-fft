add_subdirectory(src)
if(KokkosFFT_ENABLE_TESTS)
    add_subdirectory(unit_test)
endif()
