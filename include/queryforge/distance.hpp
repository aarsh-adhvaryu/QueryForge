#pragma once

#include <cstddef>

// Distance / similarity kernels — the innermost hot loop of vector search.
//
// Every search computes the distance between the query and many stored vectors, so these
// functions dominate runtime. We provide three implementations of each kernel:
//   * scalar  — plain loop, the correctness reference.
//   * SSE     — 4 floats per instruction.
//   * AVX2    — 8 floats per instruction, using FMA (fused multiply-add).
//
// The public functions (l2_sqr, dot, cosine_distance) pick the fastest implementation the
// current CPU supports, decided once at startup via runtime CPU feature detection. This means
// one binary runs everywhere and simply falls back to scalar where AVX2 is unavailable.

namespace queryforge {

// Which SIMD implementation the dispatcher selected on this machine.
enum class SimdBackend { Scalar, SSE, AVX2 };

SimdBackend active_backend() noexcept;
const char* backend_name(SimdBackend b) noexcept;

// --- Public, auto-dispatched kernels ----------------------------------------------------

// Squared Euclidean (L2) distance: sum of (a[i]-b[i])^2.
// Squared on purpose — omitting sqrt preserves ordering and is faster.
float l2_sqr(const float* a, const float* b, std::size_t dim) noexcept;

// Inner (dot) product: sum of a[i]*b[i]. For unit-length vectors this is cosine similarity.
float dot(const float* a, const float* b, std::size_t dim) noexcept;

// Cosine distance = 1 - cosine similarity, range [0, 2]. Normalizes internally, so it is
// correct for non-unit vectors (at the cost of computing the norms each call). When vectors
// are pre-normalized, prefer 1 - dot(a,b) directly.
float cosine_distance(const float* a, const float* b, std::size_t dim) noexcept;

// --- Explicit implementations (exposed for tests & benchmarks) --------------------------
//
// WARNING: calling an *_avx2 / *_sse variant on a CPU that lacks the instruction set is
// undefined behaviour. Guard with active_backend() or the cpu_has_* helpers below.
namespace detail {

float l2_sqr_scalar(const float* a, const float* b, std::size_t dim) noexcept;
float l2_sqr_sse   (const float* a, const float* b, std::size_t dim) noexcept;
float l2_sqr_avx2  (const float* a, const float* b, std::size_t dim) noexcept;

float dot_scalar(const float* a, const float* b, std::size_t dim) noexcept;
float dot_sse   (const float* a, const float* b, std::size_t dim) noexcept;
float dot_avx2  (const float* a, const float* b, std::size_t dim) noexcept;

bool cpu_has_avx2() noexcept;
bool cpu_has_sse()  noexcept;  // SSE4.1

}  // namespace detail
}  // namespace queryforge
