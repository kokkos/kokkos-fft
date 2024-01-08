#ifndef __TEST_UTILS_HPP__
#define __TEST_UTILS_HPP__

#include "Test_Types.hpp"

template <typename ViewType>
bool allclose(const ViewType& a, const ViewType& b, double rtol = 1.e-5,
              double atol = 1.e-8) {
  constexpr std::size_t rank = ViewType::rank;
  for (std::size_t i = 0; i < rank; i++) {
    assert(a.extent(i) == b.extent(i));
  }
  const auto n = a.size();

  auto* ptr_a = a.data();
  auto* ptr_b = b.data();

  int error = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>{0,
                                                                           n},
      KOKKOS_LAMBDA(const int& i, int& err) {
        auto tmp_a = ptr_a[i];
        auto tmp_b = ptr_b[i];
        bool not_close =
            Kokkos::abs(tmp_a - tmp_b) > (atol + rtol * Kokkos::abs(tmp_b));
        err += static_cast<int>(not_close);
      },
      error);

  return error == 0;
}

template <typename ViewType, typename T>
void multiply(ViewType& x, T a) {
  const auto n = x.size();
  auto* ptr_x  = x.data();

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>{0,
                                                                           n},
      KOKKOS_LAMBDA(const int& i) { ptr_x[i] = ptr_x[i] * a; });
}

template <typename T>
void display(std::string name, std::vector<T>& values) {
  std::cout << name << std::endl;
  for (auto value : values) {
    std::cout << value << std::endl;
  }
}

#endif