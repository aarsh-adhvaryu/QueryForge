// Binary serialization for HnswIndex (stage A4).
//
// On-disk format (.qfx), little-endian / native layout:
//   [Header]  magic "QFHNSW\0\0" (8B), uint32 version, uint32 metric,
//             uint64 dim, M, ef_construction, num_nodes, entry_point, int32 max_layer, pad
//   [Vectors] float[num_nodes * dim]                      (the bulk of the file)
//   [Graph]   per node: int32 node_layer; then for each layer 0..node_layer:
//                        uint32 count, uint32[count] neighbor ids
//
// load() memory-maps the file (POSIX) so the OS maps bytes straight into our address space — no
// parse-the-whole-file step, data is paged in on demand. On non-POSIX we fall back to a plain read.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "queryforge/hnsw.hpp"

#if defined(__unix__) || defined(__APPLE__)
#define QF_HAVE_MMAP 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define QF_HAVE_MMAP 0
#endif

namespace queryforge {
namespace {

constexpr char kMagic[8] = {'Q', 'F', 'H', 'N', 'S', 'W', '\0', '\0'};
constexpr std::uint32_t kVersion = 1;

// Reads typed values sequentially from a contiguous byte buffer (the mmap'd file or a read buffer).
struct ByteReader {
  const char* p;
  const char* end;

  void require(std::size_t n) const {
    if (p + n > end) throw std::runtime_error("QueryForge: truncated index file");
  }
  template <typename T>
  T read() {
    require(sizeof(T));
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  void read_into(void* dst, std::size_t n) {
    require(n);
    std::memcpy(dst, p, n);
    p += n;
  }
};

// Owns a read-only memory mapping (POSIX) or a heap buffer (fallback). Frees on destruction.
struct FileBytes {
  const char* data = nullptr;
  std::size_t size = 0;
#if QF_HAVE_MMAP
  void* map_ = nullptr;
  std::size_t map_len_ = 0;
#endif
  std::vector<char> buf_;  // used only on the fallback path

  explicit FileBytes(const std::string& path) {
#if QF_HAVE_MMAP
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("QueryForge: cannot open " + path);
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
      ::close(fd);
      throw std::runtime_error("QueryForge: cannot stat " + path);
    }
    map_len_ = static_cast<std::size_t>(st.st_size);
    map_ = ::mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // the mapping stays valid after closing the fd
    if (map_ == MAP_FAILED) throw std::runtime_error("QueryForge: mmap failed for " + path);
    data = static_cast<const char*>(map_);
    size = map_len_;
#else
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("QueryForge: cannot open " + path);
    const std::streamsize n = in.tellg();
    in.seekg(0);
    buf_.resize(static_cast<std::size_t>(n));
    in.read(buf_.data(), n);
    data = buf_.data();
    size = buf_.size();
#endif
  }

  ~FileBytes() {
#if QF_HAVE_MMAP
    if (map_ && map_ != MAP_FAILED) ::munmap(map_, map_len_);
#endif
  }

  FileBytes(const FileBytes&) = delete;
  FileBytes& operator=(const FileBytes&) = delete;
};

template <typename T>
void write_pod(std::ostream& os, const T& v) {
  os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

}  // namespace

void HnswIndex::save(const std::string& path) const {
  std::ofstream os(path, std::ios::binary | std::ios::trunc);
  if (!os) throw std::runtime_error("QueryForge: cannot write " + path);

  os.write(kMagic, sizeof(kMagic));
  write_pod<std::uint32_t>(os, kVersion);
  write_pod<std::uint32_t>(os, metric_ == Metric::Cosine ? 1u : 0u);
  write_pod<std::uint64_t>(os, dim_);
  write_pod<std::uint64_t>(os, M_);
  write_pod<std::uint64_t>(os, ef_construction_);
  write_pod<std::uint64_t>(os, num_nodes_);
  write_pod<std::uint64_t>(os, entry_point_);
  write_pod<std::int32_t>(os, max_layer_);
  write_pod<std::uint32_t>(os, 0u);  // pad

  // Vectors block (contiguous).
  os.write(reinterpret_cast<const char*>(vectors_.data()),
           static_cast<std::streamsize>(vectors_.size() * sizeof(float)));

  // Graph block.
  for (std::size_t i = 0; i < num_nodes_; ++i) {
    const auto& layers = links_[i];
    write_pod<std::int32_t>(os, static_cast<std::int32_t>(layers.size()) - 1);  // node_layer
    for (const auto& nbrs : layers) {
      write_pod<std::uint32_t>(os, static_cast<std::uint32_t>(nbrs.size()));
      os.write(reinterpret_cast<const char*>(nbrs.data()),
               static_cast<std::streamsize>(nbrs.size() * sizeof(std::uint32_t)));
    }
  }
  if (!os) throw std::runtime_error("QueryForge: write error on " + path);
}

HnswIndex HnswIndex::load(const std::string& path) {
  FileBytes bytes(path);
  ByteReader r{bytes.data, bytes.data + bytes.size};

  char magic[8];
  r.read_into(magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
    throw std::runtime_error("QueryForge: bad magic in " + path);
  const std::uint32_t version = r.read<std::uint32_t>();
  if (version != kVersion)
    throw std::runtime_error("QueryForge: unsupported index version");

  const std::uint32_t metric_raw = r.read<std::uint32_t>();
  const std::uint64_t dim = r.read<std::uint64_t>();
  const std::uint64_t M = r.read<std::uint64_t>();
  const std::uint64_t efc = r.read<std::uint64_t>();
  const std::uint64_t num_nodes = r.read<std::uint64_t>();
  const std::uint64_t entry_point = r.read<std::uint64_t>();
  const std::int32_t max_layer = r.read<std::int32_t>();
  r.read<std::uint32_t>();  // pad

  const Metric metric = metric_raw == 1 ? Metric::Cosine : Metric::L2;
  HnswIndex idx(static_cast<std::size_t>(dim), static_cast<std::size_t>(M),
                static_cast<std::size_t>(efc), metric);

  idx.num_nodes_ = static_cast<std::size_t>(num_nodes);
  idx.entry_point_ = static_cast<std::uint32_t>(entry_point);
  idx.max_layer_ = max_layer;

  idx.vectors_.resize(static_cast<std::size_t>(num_nodes * dim));
  r.read_into(idx.vectors_.data(), idx.vectors_.size() * sizeof(float));

  idx.links_.resize(static_cast<std::size_t>(num_nodes));
  for (std::size_t i = 0; i < idx.num_nodes_; ++i) {
    const std::int32_t node_layer = r.read<std::int32_t>();
    auto& layers = idx.links_[i];
    layers.resize(static_cast<std::size_t>(node_layer) + 1);
    for (auto& nbrs : layers) {
      const std::uint32_t count = r.read<std::uint32_t>();
      nbrs.resize(count);
      if (count) r.read_into(nbrs.data(), count * sizeof(std::uint32_t));
    }
  }
  return idx;
}

}  // namespace queryforge
