#include <pybind11/pybind11.h>
#include "extern.h"

namespace py = pybind11;

PYBIND11_MODULE(python_bindings, m) {
    m.def("test_just_copy", &test_just_copy, "A function that performs just copy",
        py::arg("blockSize"), py::arg("numElements"));
    m.def("test_random_index_copy", &test_random_index_copy, "A function that performs random index copy",
        py::arg("blockSize"), py::arg("numElements"));
}