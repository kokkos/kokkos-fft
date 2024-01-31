function(KOKKOSFFT_ADD_BENCHMARK name)
    CMAKE_PARSE_ARGUMENTS(
        BENCHMARK
        ""
        ""
        "SOURCES"
        ${ARGN}
    )

    if(DEFINED BENCHMARK_UNPARSED_ARGUMENTS)
        message(
            WARNING
            "Unexpected arguments when adding a benchmark: "
            ${BENCHMARK_UNPARSED_ARGUMENTS}
        )
    endif()

	set(BENCHMARK_NAME ${PACKAGE_NAME}_${name})

	#Adding BenchmarkMain.cpp to sources
	list(APPEND BENCHMARK_SOURCES
		BenchmarkMain.cpp
	)

    add_executable(
        ${BENCHMARK_NAME}
        ${BENCHMARK_SOURCES}
    )
    target_link_libraries(
        ${BENCHMARK_NAME}
        PRIVATE benchmark::benchmark Kokkos::kokkos KokkosFFT::fft
    )
    target_include_directories(
        ${BENCHMARK_NAME}
        SYSTEM PRIVATE ${benchmark_SOURCE_DIR}/include
    )
	target_include_directories(
        ${BENCHMARK_NAME}
        PRIVATE ${CMAKE_BINARY_DIR}
    )

    foreach(SOURCE_FILE ${BENCHMARK_SOURCES})
        SET_SOURCE_FILES_PROPERTIES(
            ${SOURCE_FILE}
            PROPERTIES LANGUAGE CXX
        )
    endforeach()

    string(TIMESTAMP BENCHMARK_TIME "%Y-%m-%d_T%H-%M-%S" UTC)
    set(
        BENCHMARK_ARGS
        --benchmark_counters_tabular=true
        --benchmark_out=${BENCHMARK_NAME}_${BENCHMARK_TIME}.json
    )

    add_test(
        NAME ${BENCHMARK_NAME}
        COMMAND ${BENCHMARK_NAME} ${BENCHMARK_ARGS}
    )
endfunction()

# Set benchmark targets
set(
	BENCHMARK_SOURCES
	PerfTest_FFT1.cpp
)

KOKKOSFFT_ADD_BENCHMARK(
    PerformanceTest_Benchmark
    SOURCES ${BENCHMARK_SOURCES}
)