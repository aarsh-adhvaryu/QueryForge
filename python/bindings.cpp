// Pybind11 bindings: expose the C++ HnswIndex to Python as `queryforge.HnswIndex`.
//
// Vectors cross the boundary as NumPy float32 arrays (zero extra copies for contiguous arrays).
// search() returns (ids, distances) as NumPy arrays so it drops straight into ML code.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "queryforge/hnsw.hpp"
#include "queryforge/version.hpp"

namespace py = pybind11;
using queryforge::HnswIndex;
using queryforge::Metric;

namespace {

// A contiguous float32 array, casting from other dtypes/layouts if needed.
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

}  // namespace

PYBIND11_MODULE(queryforge, m) {
  m.doc() = "QueryForge — a from-scratch C++ HNSW vector search engine.";
  m.attr("__version__") = queryforge::version();

  py::enum_<Metric>(m, "Metric")
      .value("L2", Metric::L2)
      .value("Cosine", Metric::Cosine);

  py::class_<HnswIndex>(m, "HnswIndex")
      .def(py::init<std::size_t, std::size_t, std::size_t, Metric, std::uint32_t>(),
           py::arg("dim"), py::arg("M") = 16, py::arg("ef_construction") = 200,
           py::arg("metric") = Metric::L2, py::arg("seed") = 100,
           "Create an empty index. dim is the vector dimensionality.")

      .def(
          "add",
          [](HnswIndex& self, FloatArray vec) {
            if (vec.ndim() != 1 || static_cast<std::size_t>(vec.shape(0)) != self.dim())
              throw std::invalid_argument("add() expects a 1-D array of length dim");
            return self.add(vec.data());
          },
          py::arg("vector"), "Insert one vector; returns its assigned id.")

      .def(
          "add_batch",
          [](HnswIndex& self, FloatArray mat) {
            if (mat.ndim() != 2 || static_cast<std::size_t>(mat.shape(1)) != self.dim())
              throw std::invalid_argument("add_batch() expects a 2-D array of shape (n, dim)");
            const std::size_t n = static_cast<std::size_t>(mat.shape(0));
            const float* base = mat.data();
            self.reserve(self.size() + n);  // avoid repeated reallocation during the bulk insert
            std::vector<std::uint32_t> ids;
            ids.reserve(n);
            for (std::size_t i = 0; i < n; ++i) ids.push_back(self.add(base + i * self.dim()));
            return ids;
          },
          py::arg("vectors"), "Insert n vectors from an (n, dim) array; returns the list of ids.")

      .def("reserve", &HnswIndex::reserve, py::arg("n"),
           "Pre-allocate capacity for n total vectors before a bulk build (optional, faster).")

      .def(
          "search",
          [](const HnswIndex& self, FloatArray query, std::size_t k, std::size_t ef) {
            if (query.ndim() != 1 || static_cast<std::size_t>(query.shape(0)) != self.dim())
              throw std::invalid_argument("search() expects a 1-D array of length dim");
            const auto res = self.search(query.data(), k, ef);
            py::array_t<std::uint32_t> ids(res.size());
            py::array_t<float> dists(res.size());
            auto* pid = ids.mutable_data();
            auto* pd = dists.mutable_data();
            for (std::size_t i = 0; i < res.size(); ++i) {
              pid[i] = res[i].id;
              pd[i] = res[i].distance;
            }
            return py::make_tuple(ids, dists);
          },
          py::arg("query"), py::arg("k") = 10, py::arg("ef") = 50,
          "Return (ids, distances) of the k nearest neighbors. ef is the search beam width.")

      .def("save", &HnswIndex::save, py::arg("path"), "Write the index to disk (.qfx).")
      .def_static("load", &HnswIndex::load, py::arg("path"),
                  "Load an index written by save(); uses mmap for a fast load.")

      .def_property_readonly("size", &HnswIndex::size)
      .def_property_readonly("dim", &HnswIndex::dim)
      .def_property_readonly("metric", &HnswIndex::metric)
      .def_property_readonly("max_layer", &HnswIndex::max_layer)
      .def("__len__", &HnswIndex::size)
      .def("__repr__", [](const HnswIndex& self) {
        return "<queryforge.HnswIndex size=" + std::to_string(self.size()) +
               " dim=" + std::to_string(self.dim()) + ">";
      });
}
