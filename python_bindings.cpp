#include <pybind11/pybind11.h>
#include "extern.h"

namespace py = pybind11;

PYBIND11_MODULE(shuffle_copy, m) {
    m.def("inititalize", &inititalize, "Initialize test data to copy on host and device",
        py::arg("blockSize"), py::arg("numElements"));
    m.def("clear_all", &clear_all, "Clear test data to copy on host and device");
    m.def("test_copy_allkinds", &test_copy_gpu, "A function that performs just copy",
        py::arg("blockSize"), py::arg("numElements"), py::arg("useShared"));
    m.def("test_random_index_copy", &test_random_index_copy, "A function that performs copy at random places",
        py::arg("blockSize"), py::arg("numElements"));
    m.def("test_copy_cpu", &test_copy_cpu, "A function that performs just copy on CPU",
        py::arg("numElements"));
    m.def("test_random_copy_cpu", &test_random_copy_cpu, "A function that performs copy on CPU at random places",
        py::arg("numElements"));
}