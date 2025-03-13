// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef IO_UTILS_HPP
#define IO_UTILS_HPP

#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <filesystem>

namespace IO {
namespace Impl {

inline std::string trimRight(const std::string& s, const std::string& c) {
  std::string str(s);
  return str.erase(s.find_last_not_of(c) + 1);
}

inline std::string trimLeft(const std::string& s, const std::string& c) {
  std::string str(s);
  return str.erase(0, s.find_first_not_of(c));
}

inline std::string trim(const std::string& s, const std::string& c) {
  return trimLeft(trimRight(s, c), c);
}
}  // namespace Impl

inline std::string zfill(int n, int length = 4) {
  std::ostringstream ss;
  ss << std::setfill('0') << std::setw(length) << static_cast<int>(n);
  return ss.str();
}

void mkdir(const std::string& path, std::filesystem::perms mode) {
  std::filesystem::path dir(path);

  // If the final directory already exists, throw an error.
  if (std::filesystem::exists(dir)) {
    throw std::runtime_error("mkdir error: path already exists: " + path);
  }

  // Create the directory along with any intermediate directories.
  // create_directories returns true if at least one directory was created.
  if (!std::filesystem::create_directories(dir)) {
    throw std::runtime_error("mkdir error: failed to create directory: " +
                             path);
  }

  // Set the permissions for the final directory.
  std::filesystem::permissions(dir, mode);
}

using dict = std::map<std::string, std::string>;
dict parse_args(int argc, char* argv[]) {
  dict kwargs;
  const std::vector<std::string> args(argv + 1, argv + argc);

  assert(args.size() % 2 == 0);

  for (std::size_t i = 0; i < args.size(); i += 2) {
    std::string key   = Impl::trimLeft(args[i], "-");
    std::string value = args[i + 1];
    kwargs[key]       = value;
  }
  return kwargs;
}

std::string get_arg(dict& kwargs, const std::string& key,
                    const std::string& default_value = "") {
  if (kwargs.find(key) != kwargs.end()) {
    return kwargs[key];
  } else {
    return default_value;
  }
}

template <typename ViewType>
void to_binary(const std::string& filename, const ViewType& view) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  using value_type = typename ViewType::non_const_value_type;
  std::size_t size = view.size();

  // Write the entire block of data to the file.
  ofs.write(reinterpret_cast<const char*>(view.data()),
            size * sizeof(value_type));
}

template <typename ViewType>
void from_binary(const std::string& filename, const ViewType& view) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  using value_type = typename ViewType::non_const_value_type;
  std::size_t size = view.size();

  // Attempt to read the binary data into the view's buffer.
  ifs.read(reinterpret_cast<char*>(view.data()), size * sizeof(value_type));
}
}  // namespace IO

#endif
