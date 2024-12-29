#include <pybind11/pybind11.h>
#include "extern.h"

namespace py = pybind11;

PYBIND11_MODULE(shuffle_copy, m) {
    m.def("test_copy_allkinds", &test_copy_allkinds, "A function that performs just copy",
        py::arg("blockSize"), py::arg("numElements"), py::arg("numElements"), py::arg("numElements"));
}