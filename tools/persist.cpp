// qf_persist — build an HNSW index, save it, reload it, and report build vs load time + file size.
// Demonstrates the persistence win: build once (slow), load many times (fast).
//
// Usage: qf_persist N=100000 dim=128 M=16 efc=200 metric=l2 path=/tmp/qf_demo.qfx
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include "bruteforce.hpp"
#include "queryforge/hnsw.hpp"

namespace {
std::size_t to_size(const std::string& v) { return static_cast<std::size_t>(std::stoull(v)); }
}  // namespace

int main(int argc, char** argv) {
  using clock = std::chrono::steady_clock;
  std::size_t N = 100000, dim = 128, M = 16, efc = 200;
  queryforge::Metric metric = queryforge::Metric::L2;
  std::string path = "/tmp/qf_demo.qfx";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto eq = a.find('=');
    if (eq == std::string::npos) continue;
    std::string k = a.substr(0, eq), v = a.substr(eq + 1);
    if (k == "N") N = to_size(v);
    else if (k == "dim") dim = to_size(v);
    else if (k == "M") M = to_size(v);
    else if (k == "efc") efc = to_size(v);
    else if (k == "metric") metric = (v == "cosine") ? queryforge::Metric::Cosine : queryforge::Metric::L2;
    else if (k == "path") path = v;
  }

  std::cout << "qf_persist: N=" << N << " dim=" << dim << " M=" << M << " efc=" << efc << "\n";
  const auto data = qf_tools::random_dataset(N, dim, /*seed=*/1);

  // Build.
  auto t0 = clock::now();
  queryforge::HnswIndex index(dim, M, efc, metric);
  index.reserve(N);  // pre-allocate: avoids realloc copies + the transient 2x memory spike
  for (std::size_t i = 0; i < N; ++i) index.add(&data[i * dim]);
  auto t1 = clock::now();
  const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Save.
  auto s0 = clock::now();
  index.save(path);
  auto s1 = clock::now();
  const double save_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();

  std::FILE* f = std::fopen(path.c_str(), "rb");
  std::fseek(f, 0, SEEK_END);
  const long file_bytes = std::ftell(f);
  std::fclose(f);

  // Load (mmap).
  auto l0 = clock::now();
  queryforge::HnswIndex loaded = queryforge::HnswIndex::load(path);
  auto l1 = clock::now();
  const double load_ms = std::chrono::duration<double, std::milli>(l1 - l0).count();

  // Integrity: the loaded index must search identically to the in-memory one.
  bool ok = (loaded.size() == index.size());
  for (std::size_t qi = 0; qi < 100 && ok; ++qi) {
    const auto a = index.search(&data[qi * dim], 10, 64);
    const auto b = loaded.search(&data[qi * dim], 10, 64);
    if (a.size() != b.size()) { ok = false; break; }
    for (std::size_t i = 0; i < a.size(); ++i)
      if (a[i].id != b[i].id) { ok = false; break; }
  }

  std::cout << "  build time : " << build_ms << " ms\n"
            << "  save time  : " << save_ms << " ms\n"
            << "  file size  : " << (file_bytes / (1024.0 * 1024.0)) << " MiB ("
            << (static_cast<double>(file_bytes) / N) << " bytes/vector)\n"
            << "  LOAD time  : " << load_ms << " ms  <-- vs build " << build_ms << " ms ("
            << (build_ms / load_ms) << "x faster)\n"
            << "  loaded size: " << loaded.size() << " vectors, sanity " << (ok ? "OK" : "FAIL")
            << "\n";
  return ok ? 0 : 1;
}
