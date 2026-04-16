<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Documentation Instructions for Kokkos-FFT

These instructions apply when creating, modifying, or reviewing documentation files (`.rst`, `.md`, `.py` config files) in the `docs/` directory and other documentation-related files.

## Documentation Stack

- **Sphinx**: Documentation generator for `.rst` files.
- **Breathe**: Bridge between Doxygen and Sphinx for C++ API documentation.
- **Doxygen**: Extracts API documentation from C++ source code comments.
- **Read the Docs theme** (`sphinx_rtd_theme`): HTML theme for the generated site.
- **sphinx_copybutton**: Adds copy buttons to code blocks.

## License Headers

Every documentation file must include an SPDX license header.

For reStructuredText (`.rst`) files:

```rst
.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
```

For Markdown (`.md`) files:

```markdown
<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->
```

For Python configuration files (`.py`):

```python
# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
```

CI enforces REUSE compliance — omitting this header causes build failure.

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── Doxyfile.in           # Doxygen configuration template
├── index.rst             # Root document
├── api_reference.rst     # API reference index
├── api/                  # API documentation by category
│   ├── standard/         # Standard FFT functions (fft, ifft, fft2, etc.)
│   ├── real/             # Real FFT functions (rfft, irfft, etc.)
│   ├── hermitian/        # Hermitian FFT functions (hfft, ihfft)
│   ├── helper/           # Helper functions (fftfreq, fftshift, etc.)
│   ├── plan/             # Plan class and execute function
│   └── enums/            # Normalization and Direction enums
├── intro/                # Introduction and getting started
│   ├── building.rst      # Build instructions
│   ├── quick_start.rst   # Quick start guide
│   └── using.rst         # Usage guide
├── developer/            # Developer documentation
└── samples/              # Code samples for inclusion
```

## reStructuredText Conventions

### Section Headings

Use consistent heading underlines:

```rst
Top-Level Heading
=================

Second-Level Heading
--------------------

Third-Level Heading
^^^^^^^^^^^^^^^^^^^
```

### Cross-References

- **Internal document references**: `:doc:\`path/to/document\``
- **API references**: `:cpp:func:\`KokkosFFT::fft\``
- **External links**: Use full URLs: `\`Link text <https://example.com>\`_`

### Code Blocks

```rst
.. code-block:: cpp

   // C++ code here
   KokkosFFT::fft(exec_space, in, out);
```

Use `literalinclude` for example code from source files:

```rst
.. literalinclude:: ../../examples/01_1DFFT/01_1DFFT.cpp
   :language: cpp
   :lines: 10-30
```

### Admonitions

```rst
.. note::
   Important information for the reader.

.. warning::
   Critical information about potential issues.
```

### Tables

Use list-table directive for complex tables:

```rst
.. list-table::
   :header-rows: 1

   * - Column 1
     - Column 2
   * - Data 1
     - Data 2
```

## C++ API Documentation (Doxygen)

### In-Source Documentation

Use `///` Doxygen comments in C++ headers for all public API functions:

```cpp
/// \brief Brief description of the function.
///
/// Detailed description if needed.
///
/// \tparam ExecutionSpace The Kokkos execution space.
/// \tparam InViewType The input Kokkos View type.
///
/// \param exec_space[in] The execution space instance.
/// \param in[in] The input view.
/// \param out[out] The output view.
///
/// \return Description of return value.
///
/// \throws std::runtime_error If preconditions are not met.
```

### Breathe Directives in RST

Reference Doxygen-documented C++ functions in `.rst` files:

```rst
.. doxygenfunction:: KokkosFFT::fft(const ExecutionSpace&, const InViewType&, const OutViewType&, KokkosFFT::Normalization, int, std::optional<std::size_t>)
```

For classes:

```rst
.. doxygenclass:: KokkosFFT::Plan
   :members:
```

For enums:

```rst
.. doxygenenum:: KokkosFFT::Normalization
```

## Sphinx Configuration

- Configuration file: `docs/conf.py`.
- Version is extracted automatically from `CMakeLists.txt` via regex parsing.
- Extensions: `breathe`, `sphinx_copybutton`.
- Breathe project name: `KokkosFFT`.
- Doxygen inputs: `common/src/` and `fft/src/`.

## Adding New API Documentation

1. Add Doxygen comments (`/// \brief`, `\tparam`, `\param`, etc.) to the C++ header.
2. Create a new `.rst` file in the appropriate `docs/api/<category>/` subdirectory.
3. Use `doxygenfunction`, `doxygenclass`, or `doxygenenum` directives to pull in the documentation.
4. Add the new `.rst` file to the appropriate `toctree` in `docs/api_reference.rst` or a parent document.
5. Include a code example if applicable, either inline or via `literalinclude`.

## Spell Checking

- The `typos` tool checks spelling in documentation files.
- Custom word exceptions are in `.typos.toml` (e.g., `iy`, `iz`, `ND`, `arange`).
- If adding a legitimate technical term that `typos` flags, add it to `.typos.toml`.

## Building Documentation Locally

Enable documentation build with CMake:

```bash
cmake -B build -DKokkosFFT_ENABLE_DOCS=ON
cmake --build build --target docs
```

Requires Doxygen and Sphinx with the `breathe` and `sphinx_rtd_theme` packages installed.

## Things to Avoid

- Do not use `#` style headings inconsistently in RST — use the project's underline convention.
- Do not add API documentation without corresponding Doxygen comments in the C++ source.
- Do not forget SPDX license headers on new `.rst` files.
- Do not reference internal (`KokkosFFT::Impl::`) functions in user-facing documentation.
- Do not hardcode version numbers — they are extracted from `CMakeLists.txt` automatically.
