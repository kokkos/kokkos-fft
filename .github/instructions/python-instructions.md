<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Python Instructions for Kokkos-FFT

These instructions apply when creating, modifying, or reviewing Python files (`.py`) in this repository, primarily in the `python_scripts/` and `examples/` directories.

## Python Version

- **Minimum supported**: Python 3.8 (as specified in `pyproject.toml`).
- **CI tested versions**: Python 3.10, 3.11, 3.12.
- Avoid features only available in Python 3.9+ without checking compatibility.

## License Headers

Every Python file must start with the SPDX license header:

```python
# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
```

CI enforces REUSE compliance — omitting this header causes build failure.

## Code Style

### Type Hints

Use type hints for all function signatures:

```python
def extract_versions(filename: str, modules: List[str]) -> dict:
    ...
```

### Docstrings

Use NumPy-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of the function.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int
        Description of param2.

    Returns
    -------
    bool
        Description of return value.

    Raises
    ------
    ValueError
        If the input is invalid.
    """
```

### Imports

- Group imports: standard library, third-party, local.
- Use absolute imports.

## Static Analysis

- **Pylint** is used for static analysis.
- Configuration is in `python_scripts/.pylintrc`.
- CI runs `pylint` on `examples/` and `python_scripts/` directories.
- Run `pylint <file>` before committing.

## Testing

- **pytest** is used for Python tests.
- Test files follow the `test_*.py` naming convention.
- CI runs pytest on `examples/09_derivative` and `python_scripts/`.
- Run `pytest <test_file>` to execute tests locally.

## Project Configuration

- Build system: `hatchling` (specified in `python_scripts/pyproject.toml`).
- Dependencies are listed in `pyproject.toml` under `[project.dependencies]`.
- Key dependencies: `pytest`, `numpy`, `xarray`, `matplotlib`, `pylint`.

## Python Scripts

### `check_versions.py`

- Purpose: Verify version consistency across README, docs, and CMakeLists.txt.
- Checks modules: CMake, Kokkos, gcc, IntelLLVM, nvcc, rocm.
- Validates versions in: `README.md`, example READMEs, `docs/intro/building.rst`, `docs/intro/quick_start.rst`, `CMakeLists.txt`.
- Uses regex patterns for version extraction.

### Adding New Scripts

1. Create the script in `python_scripts/`.
2. Add the SPDX license header.
3. Include type hints and NumPy-style docstrings.
4. Add corresponding tests in `python_scripts/test_<script_name>.py`.
5. Ensure the script passes `pylint` checks.

## Error Handling

- Use `ValueError` with descriptive messages for validation errors.
- Use `FileNotFoundError` for missing file errors.
- Provide detailed error context including file paths and expected vs. actual values.

## Things to Avoid

- Do not use Python 2 syntax.
- Do not add dependencies without updating `pyproject.toml`.
- Do not skip type hints on function signatures.
- Do not use `print()` for error reporting — raise exceptions with descriptive messages.
- Do not hardcode file paths — use `pathlib.Path` or `os.path` for path manipulation.
