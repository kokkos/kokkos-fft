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
#include <sys/stat.h>

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

// From https://gist.github.com/hrlou/cd440c181df5f4f2d0b61b80ca13516b
static int do_mkdir(const std::string& path, mode_t mode) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) {
    if (mkdir(path.c_str(), mode) != 0 && errno != EEXIST) {
      return -1;
    }
  } else if (!S_ISDIR(st.st_mode)) {
    errno = ENOTDIR;
    return -1;
  }
  return 0;
}

}  // namespace Impl

inline std::string zfill(int n, int length = 4) {
  std::ostringstream ss;
  ss << std::setfill('0') << std::setw(length) << static_cast<int>(n);
  return ss.str();
}

int mkdirs(std::string path, mode_t mode) {
  std::string build;
  for (std::size_t pos = 0; (pos = path.find('/')) != std::string::npos;) {
    build += path.substr(0, pos + 1);
    Impl::do_mkdir(build, mode);
    path.erase(0, pos + 1);
  }
  if (!path.empty()) {
    build += path;
    Impl::do_mkdir(build, mode);
  }
  return 0;
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

template <class ViewType>
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

template <class ViewType>
void from_binary(const std::string& filename, ViewType& view) {
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
