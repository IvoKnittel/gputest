#include <../pybind11/pybind11.h>
#include "extern.h"

namespace py = pybind11;

PYBIND11_MODULE(shuffle_copy, m) {
    m.def("enableDevice", &enableDevice, "Enable device");
    m.def("inititalize", &inititalize, "Initialize image  size to copy on host and device", py::arg("numElements"));
    m.def("MoveImageToDevice", &MoveImageToDevice, "Move image to device", py::arg("image"));
    m.def("image_copy", &image_copy, "Copy image within device");
    m.def("GetImageFromDevice", &GetImageFromDevice, "Move image to host", py::arg("image"));
    m.def("clear_all", &clear_all, "Clear test data to copy on host and device");
}
