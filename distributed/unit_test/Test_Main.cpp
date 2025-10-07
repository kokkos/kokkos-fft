#include <mpi.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

class MpiEnvironment : public ::testing::Environment {
 public:
  ~MpiEnvironment() override {}

  // Override this to define how to set up the environment.
  void SetUp() override {
    m_comm = MPI_COMM_WORLD;

    // Ensure all processes start tests together
    ::MPI_Barrier(m_comm);
  }

  // Override this to define how to tear down the environment.
  void TearDown() override {
    // Ensure all processes finish tests together
    ::MPI_Barrier(m_comm);
  }

  MPI_Comm m_comm;
};

// Custom printer to force output per rank
class MPITestEventListener : public ::testing::EmptyTestEventListener {
 public:
  void OnTestPartResult(const ::testing::TestPartResult& result) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (result.failed()) {
      std::ostringstream oss;
      oss << "[Rank " << rank << "] "
          << (result.fatally_failed() ? "FATAL FAILURE" : "FAILURE") << ": "
          << result.file_name() << ":" << result.line_number() << "\n"
          << result.summary() << std::endl;
      std::cout << oss.str();
    }
  }
};

int main(int argc, char* argv[]) {
// Initialize MPI first
#if defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  MPI_Init(&argc, &argv);
#else
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    throw std::runtime_error("MPI_THREAD_MULTIPLE is needed");
  }
#endif

  // Initialize google test
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MpiEnvironment());

  Kokkos::initialize(argc, argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Keep Gtest print on rank 0 and print errors from other ranks
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  if (rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new MPITestEventListener);
  }

  // run tests
  auto result = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  Kokkos::finalize();
  MPI_Finalize();

  return result;
}
