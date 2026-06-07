#pragma once

// Public API surface for QueryForge.
//
// This header is intentionally tiny for the A0 scaffold — it exists so the build,
// the test runner, and the benchmark runner all have something real to compile and
// link against. The actual engine (distance math, NSW/HNSW graph, persistence) lands
// in later stages under include/queryforge/ and src/.

namespace queryforge {

// Returns the library version string, e.g. "0.1.0".
const char* version() noexcept;

}  // namespace queryforge
