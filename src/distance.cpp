#include "queryforge/distance.hpp"

#include <cmath>

// On x86 we use SSE/AVX2 intrinsics. The SIMD functions carry a GCC/Clang
// __attribute__((target(...))) so they compile with the right instructions even though the
// rest of the translation unit is built for a baseline CPU. On non-x86 we compile only the
// scalar path and alias the SIMD entry points to it.
#if defined(__x86_64__) || defined(__i386__)
#define QF_X86 1
#include <immintrin.h>
#else
#define QF_X86 0
#endif

namespace queryforge {
namespace detail {

// ---- Scalar reference (the source of truth for correctness) ----------------------------

float l2_sqr_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float dot_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

#if QF_X86

// ---- SSE (4 floats / instruction) ------------------------------------------------------

__attribute__((target("sse4.1")))
float l2_sqr_sse(const float* a, const float* b, std::size_t dim) noexcept {
  __m128 acc = _mm_setzero_ps();
  std::size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    const __m128 va = _mm_loadu_ps(a + i);
    const __m128 vb = _mm_loadu_ps(b + i);
    const __m128 d = _mm_sub_ps(va, vb);
    acc = _mm_add_ps(acc, _mm_mul_ps(d, d));
  }
  // Horizontal sum of the 4 lanes: two hadds collapse [x y z w] -> x+y+z+w.
  acc = _mm_hadd_ps(acc, acc);
  acc = _mm_hadd_ps(acc, acc);
  float result = _mm_cvtss_f32(acc);
  for (; i < dim; ++i) {  // tail when dim isn't a multiple of 4
    const float d = a[i] - b[i];
    result += d * d;
  }
  return result;
}

__attribute__((target("sse4.1")))
float dot_sse(const float* a, const float* b, std::size_t dim) noexcept {
  __m128 acc = _mm_setzero_ps();
  std::size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    const __m128 va = _mm_loadu_ps(a + i);
    const __m128 vb = _mm_loadu_ps(b + i);
    acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
  }
  acc = _mm_hadd_ps(acc, acc);
  acc = _mm_hadd_ps(acc, acc);
  float result = _mm_cvtss_f32(acc);
  for (; i < dim; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

// ---- AVX2 (8 floats / instruction, with FMA) -------------------------------------------

__attribute__((target("avx2,fma")))
float l2_sqr_avx2(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    const __m256 d = _mm256_sub_ps(va, vb);
    acc = _mm256_fmadd_ps(d, d, acc);  // acc += d*d in one instruction
  }
  // Reduce 8 lanes -> 1: fold the high 128 into the low 128, then two hadds.
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float result = _mm_cvtss_f32(s);
  for (; i < dim; ++i) {  // tail when dim isn't a multiple of 8
    const float d = a[i] - b[i];
    result += d * d;
  }
  return result;
}

__attribute__((target("avx2,fma")))
float dot_avx2(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    acc = _mm256_fmadd_ps(va, vb, acc);
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float result = _mm_cvtss_f32(s);
  for (; i < dim; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

bool cpu_has_avx2() noexcept {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") != 0;
}

bool cpu_has_sse() noexcept {
  __builtin_cpu_init();
  return __builtin_cpu_supports("sse4.1") != 0;
}

#else  // non-x86: no intrinsics available — alias everything to scalar.

float l2_sqr_sse (const float* a, const float* b, std::size_t dim) noexcept { return l2_sqr_scalar(a, b, dim); }
float l2_sqr_avx2(const float* a, const float* b, std::size_t dim) noexcept { return l2_sqr_scalar(a, b, dim); }
float dot_sse    (const float* a, const float* b, std::size_t dim) noexcept { return dot_scalar(a, b, dim); }
float dot_avx2   (const float* a, const float* b, std::size_t dim) noexcept { return dot_scalar(a, b, dim); }
bool  cpu_has_avx2() noexcept { return false; }
bool  cpu_has_sse()  noexcept { return false; }

#endif  // QF_X86

}  // namespace detail

// ---- Runtime dispatch ------------------------------------------------------------------
// Decide once which backend to use, then every call routes through it.

namespace {
SimdBackend resolve_backend() noexcept {
  if (detail::cpu_has_avx2()) return SimdBackend::AVX2;
  if (detail::cpu_has_sse())  return SimdBackend::SSE;
  return SimdBackend::Scalar;
}
const SimdBackend kBackend = resolve_backend();
}  // namespace

SimdBackend active_backend() noexcept { return kBackend; }

const char* backend_name(SimdBackend b) noexcept {
  switch (b) {
    case SimdBackend::AVX2:   return "AVX2";
    case SimdBackend::SSE:    return "SSE";
    case SimdBackend::Scalar: return "Scalar";
  }
  return "Unknown";
}

float l2_sqr(const float* a, const float* b, std::size_t dim) noexcept {
  switch (kBackend) {
    case SimdBackend::AVX2: return detail::l2_sqr_avx2(a, b, dim);
    case SimdBackend::SSE:  return detail::l2_sqr_sse(a, b, dim);
    default:                return detail::l2_sqr_scalar(a, b, dim);
  }
}

float dot(const float* a, const float* b, std::size_t dim) noexcept {
  switch (kBackend) {
    case SimdBackend::AVX2: return detail::dot_avx2(a, b, dim);
    case SimdBackend::SSE:  return detail::dot_sse(a, b, dim);
    default:                return detail::dot_scalar(a, b, dim);
  }
}

float cosine_distance(const float* a, const float* b, std::size_t dim) noexcept {
  const float dab = dot(a, b, dim);
  const float daa = dot(a, a, dim);
  const float dbb = dot(b, b, dim);
  const float denom = std::sqrt(daa * dbb);
  if (denom == 0.0f) return 1.0f;  // undefined for a zero vector; define as "maximally unrelated"
  return 1.0f - dab / denom;
}

}  // namespace queryforge
