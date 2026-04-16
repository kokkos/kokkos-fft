<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# C++ Instructions for Kokkos-FFT

These instructions apply when creating, modifying, or reviewing C++ code (`.hpp`, `.cpp`) in this repository.

## Language Standard

- **Minimum**: C++17. Use C++17 features freely (structured bindings, `if constexpr`, `std::optional`, fold expressions, etc.).
- **C++20**: Supported but not required. Use `#if __cplusplus >= 202002L` guards for C++20-only features (e.g., `std::source_location`).
- Avoid C++23 or later features.

## Header-Only Library

- The library (`common/` and `fft/`) is entirely header-only. All implementation is in `.hpp` files.
- The only `.cpp` files are test files (`Test_*.cpp`) and example programs.
- Do not add `.cpp` files to `common/src/` or `fft/src/`.

## File Naming

- Source headers: `KokkosFFT_<ComponentName>.hpp` (PascalCase component name).
- Backend-specific headers: `KokkosFFT_<Backend>_<Component>.hpp` (e.g., `KokkosFFT_Cuda_types.hpp`).
- Test files: `Test_<ComponentName>.cpp` (e.g., `Test_Padding.cpp`).

## License Headers

Every C++ file must start with the SPDX license header:

```cpp
// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
```

CI enforces REUSE compliance — omitting this header causes build failure.

## Include Guards

Use `#ifndef`/`#define`/`#endif` include guards (not `#pragma once`):

```cpp
#ifndef KOKKOSFFT_COMPONENTNAME_HPP
#define KOKKOSFFT_COMPONENTNAME_HPP

// ... content ...

#endif
```

The guard name follows the pattern `KOKKOSFFT_<FILENAME_UPPER>_HPP`, where the filename is uppercased with underscores.

## Include Ordering

- **Do NOT sort includes.** The `.clang-format` configuration has `SortIncludes: false` intentionally.
- Group includes logically but do not rely on alphabetical ordering.
- Place the component's own header first in test files.

## Formatting

- Code is formatted with `clang-format` using the project's `.clang-format` file (Google style base with customizations).
- Key settings: `AlignConsecutiveAssignments: true`, `AllowShortCaseLabelsOnASingleLine: true`, `AllowShortIfStatementsOnASingleLine: true`, `InsertNewlineAtEOF: true`.
- Run `clang-format -i <file>` before committing.
- CI will reject unformatted code.

## Static Analysis

- `clang-tidy` is enabled with these checks: `modernize-type-traits`, `modernize-use-using`, `modernize-use-nullptr`, `cppcoreguidelines-pro-type-cstyle-cast`, `bugprone-reserved-identifier`.
- Header filter applies to `(common|fft).*\.hpp` — excludes `tpls/`.
- Do not use C-style casts; use `static_cast`, `reinterpret_cast`, or `const_cast`.
- Use `using` declarations instead of `typedef`.
- Use `nullptr` instead of `NULL` or `0` for null pointers.

## Naming Conventions

| Entity | Convention | Examples |
|--------|-----------|---------|
| Types/Classes | PascalCase | `ViewType`, `ExecutionSpace`, `Plan` |
| Functions | snake_case | `get_modified_shape()`, `crop_or_pad()` |
| Variables | snake_case | `in_view`, `out_extents`, `fft_size` |
| Class members | `m_` prefix + snake_case | `m_in_topology`, `m_plan` |
| Template parameters | PascalCase | `ExecutionSpace`, `InViewType`, `DIM` |
| Enum types | PascalCase | `Normalization`, `Direction` |
| Enum values | snake_case | `Normalization::forward`, `Direction::backward` |
| Macros | UPPER_SNAKE_CASE | `KOKKOSFFT_THROW_IF`, `KOKKOSFFT_ENABLE_TPL_CUFFT` |
| Constants | snake_case or UPPER_SNAKE_CASE | `constexpr std::size_t MAX_FFT_DIM = 3` |

## Namespace Organization

- **Public API**: `KokkosFFT::` — all user-facing functions and types.
- **Internal implementation**: `KokkosFFT::Impl::` — not for user access.
- **Distributed**: `KokkosFFT::Distributed::` — MPI-based distributed FFT (experimental).
- **Testing utilities**: `KokkosFFT::Testing::` — test helper functions.
- Always close namespaces with a comment: `} // namespace KokkosFFT`.

## Template Patterns

Public API functions follow this pattern:

```cpp
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void function_name(const ExecutionSpace& exec_space,
                   const InViewType& in, const OutViewType& out, ...);
```

- `ExecutionSpace` is always the first template parameter.
- Use SFINAE (`std::enable_if_t`) and type traits for compile-time validation.
- Use `Kokkos::View<>` for data; never raw pointers in the public API.

## Type Traits

- Use the project's type traits from `KokkosFFT_traits.hpp`:
  - `base_floating_point<T>` — extracts base floating-point type from complex.
  - `is_real_v<T>` — checks if type is `float` or `double`.
  - `is_complex_v<T>` — checks if type is `Kokkos::complex<float/double>`.
  - `is_admissible_value_type_v<T>` — validates type for FFT operations.
  - `is_layout_left_or_right_v<ViewType>` — validates Kokkos view layout.

## Error Handling

- **Runtime errors**: Use `KOKKOSFFT_THROW_IF(condition, "message")` which throws `std::runtime_error`. The condition is **true-to-throw** (throws when expression evaluates to true).
- **Compile-time checks**: Use `KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(InViewType, OutViewType, "name")` for view compatibility validation in binary operations. For other compile time checks, use `static_assert`. Make sure that the function name is appropriately embedded in the error messages.
- **FFT backend errors**: Use `KokkosFFT::Impl::check_fft_call()` for vendor FFT return code checking.
- Do not use raw `throw` or `assert` — use the project macros instead.

## Complex Numbers

- Use `Kokkos::complex<T>` exclusively (not `std::complex`).
- Supported base types: `float` and `double` only.

## Kokkos Patterns

- Use `Kokkos::View<>` for all data containers.
- Supported layouts: `Kokkos::LayoutLeft` and `Kokkos::LayoutRight`.
- Use `Kokkos::parallel_for`, `Kokkos::parallel_reduce` with appropriate policies (`RangePolicy`, `MDRangePolicy`, `TeamPolicy`).
- Use `KOKKOS_LAMBDA` for parallel kernels.
- Pass an execution space instance (`Kokkos::ExecutionSpace`) to functions if available. This encourages the asynchronous mechanism in Kokkos.
- Use `Kokkos::deep_copy()` for data transfer between host and device.
- Use `Kokkos::create_mirror_view()` for host-accessible copies.

## Documentation Style

Use Doxygen-style comments for public API functions:

```cpp
/// \brief Brief description of the function.
///
/// Detailed description if needed.
///
/// \tparam ExecutionSpace The Kokkos execution space.
/// \tparam InViewType The input Kokkos View type.
/// \tparam OutViewType The output Kokkos View type.
///
/// \param exec_space[in] The execution space instance.
/// \param in[in] The input view.
/// \param out[out] The output view.
///
/// \return Description of return value.
///
/// \throws std::runtime_error If preconditions are not met.
```

- Use `///` prefix (not `/** */`).
- Use `\brief`, `\tparam`, `\param`, `\return`, `\throws` tags.
- Internal implementation functions may have shorter comments.

## Backend Abstraction

- Backend selection is controlled by preprocessor defines (`KOKKOSFFT_ENABLE_TPL_CUFFT`, `KOKKOSFFT_ENABLE_TPL_FFTW`, etc.) set via CMake.
- Each backend provides three files: `KokkosFFT_<Backend>_types.hpp`, `KokkosFFT_<Backend>_plans.hpp`, `KokkosFFT_<Backend>_transform.hpp`.
- When adding backend-specific code, wrap it in `#if defined(KOKKOSFFT_ENABLE_TPL_<BACKEND>)` guards.
- Only one device backend can be active at a time (cuFFT, hipFFT, rocFFT, or oneMKL are mutually exclusive).

## Common Type Definitions

From `KokkosFFT_common_types.hpp`:

```cpp
template <std::size_t DIM>
using axis_type = std::array<int, DIM>;

template <std::size_t DIM>
using shape_type = std::array<std::size_t, DIM>;

enum class Normalization { forward, backward, ortho, none };
enum class Direction { forward, backward };
constexpr std::size_t MAX_FFT_DIM = 3;
```

## Things to Avoid

- Do not modify files under `tpls/` (third-party libraries).
- Do not use `std::complex` — use `Kokkos::complex` unless the backend library explicitly uses `std::complex` to represent their complex data.
- Do not use C-style casts — use C++ casts.
- Do not use `typedef` — use `using`.
- Do not use `NULL` — use `nullptr`.
- Do not use raw `new`/`delete` — use Kokkos views and RAII.
- Do not reorder includes.
- Do not add `.cpp` implementation files to the library source directories.
